import os
import re
import sys
import gc
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import chromadb
from PIL import Image
import pypdfium2 as pdfium
from transformers import ColPaliForRetrieval, ColPaliProcessor
from pydantic import BaseModel, Field

from src.extraction import LlamaVisionESGExtractor
from src.temporal_auditor import TemporalAuditor, TemporalAuditSummary
from src.strategic_ambiguity import StrategicAmbiguityEngine, AmbiguityAuditResult


class ComprehensiveAuditReport(BaseModel):
    company: str
    target_years: List[int] = Field(default_factory=list)
    extracted_pages: List[Any] = Field(default_factory=list)
    temporal_summary: Any
    ambiguity_evaluations: List[Any] = Field(default_factory=list)

    @property
    def restatements(self):
        if hasattr(self.temporal_summary, "flagged_anomalies"):
            return self.temporal_summary.flagged_anomalies
        return []


class AutoMultimodalTemporalAuditor:
    def __init__(
        self,
        chroma_db_path: str = "./data/chromadb",
        collection_name: str = "esg_visual_reports",
        colpali_model_id: str = "vidore/colpali-v1.3-hf",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.colpali_model_id = colpali_model_id
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)

        self.temporal_auditor = TemporalAuditor(restatement_threshold=0.05)
        self.ambiguity_engine = StrategicAmbiguityEngine()
        self.selected_pages = []
        self.last_extracted_dataset = []

    def discover_all_companies_and_years(self) -> Dict[str, List[int]]:
        records = self.collection.get(include=["metadatas"])
        if not records["ids"]:
            return {}

        company_year_map: Dict[str, set] = {}
        for m in records["metadatas"]:
            company = m.get("company", "Unknown Company")
            year = m.get("fiscal_year")
            if company and year:
                company_year_map.setdefault(company, set()).add(int(year))

        return {comp: sorted(list(years)) for comp, years in company_year_map.items()}

    @staticmethod
    def _score_page_for_emissions_table(pdf: pdfium.PdfDocument, page_number: int) -> float:
        """
        Scores a PDF page based on whether it contains an actual quantitative Scope 1/2/3
        emissions appendix table rather than narrative prose.
        """
        try:
            total_pages = len(pdf)
            text = pdf[page_number - 1].get_textpage().get_text_range()
        except Exception:
            return -10.0

        if not text:
            return -10.0

        tl = text.lower()
        score = 0.0

        # 1. Reward explicit Scope 1, 2, 3 labels OR Amazon's direct/indirect row terminology
        has_s1 = "scope 1" in tl or "fossil fuels" in tl or "direct emissions" in tl
        has_s2 = "scope 2" in tl or "purchased electricity" in tl or "market-based" in tl
        has_s3 = "scope 3" in tl or "value chain" in tl or "indirect emissions" in tl

        if has_s1:
            score += 3.5
        if has_s2:
            score += 3.5
        if has_s3:
            score += 3.5
        if has_s1 and has_s2 and has_s3:
            score += 5.0

        # 2. Strong bonus for appendix / data table headers
        if any(
            k in tl
            for k in [
                "appendix",
                "data table",
                "environmental data",
                "footprint data",
                "carbon footprint",
                "ghg emissions",
                "million metric tons",
                "mmt co2e",
                "tco2e",
            ]
        ):
            score += 4.0

        # 3. Check for multi-year column headers (e.g., 2019, 2020, 2021, 2022, 2023, 2024)
        years_found = set(re.findall(r"\b(2019|2020|2021|2022|2023|2024)\b", tl))
        if len(years_found) >= 3:
            score += 6.0
        elif len(years_found) == 2:
            score += 3.0

        # 4. Count decimal/comma numbers (real tables have dense numbers; narrative prose does not)
        nums = re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b", text)
        if len(nums) >= 12:
            score += 7.0
        elif len(nums) >= 6:
            score += 3.5
        else:
            score -= 5.0

        # 5. Position heuristic: ESG appendix tables sit toward the back of long reports
        if page_number <= 18 and total_pages > 30:
            score -= 6.0
        elif page_number >= int(total_pages * 0.65):
            score += 4.0

        if "table of contents" in tl or page_number <= 5:
            score -= 12.0

        return score

    @staticmethod
    def _count_quantitative_targets(statement: str) -> int:
        """
        Counts substantive quantitative metrics/targets in narrative text (e.g., '50%', '100%',
        '10 million', '5.2 GW', '25,000 MT') while ignoring calendar years (2015-2050)
        and branding terms like '1-Click' or 'Scope 1'.
        """
        if not statement:
            return 0
        # Strip calendar years (e.g. 2020, 2025, 2030, 2040, 2050)
        cleaned = re.sub(r"\b20[1-5][0-9]\b", "", statement)
        # Strip 'Scope 1', 'Scope 2', 'Scope 3', and '1-Click'
        cleaned = re.sub(r"scope\s*[123]\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"1-click", "", cleaned, flags=re.IGNORECASE)

        # Match percentages (e.g. 50%, 100%), decimals/comma numbers, or numbers followed by units
        quant_matches = re.findall(
            r"\b\d+(?:\.\d+)?%|\b\d{1,3}(?:,\d{3})+|\b\d+\.\d+\b|\b\d+\s*(?:million|billion|thousand|gw|mw|mt|tons|tonnes|kwh)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        return len(quant_matches)

    def retrieve_pages_for_company(
        self, query: str, company: str, available_years: List[int], top_k_per_year: int = 2
    ):
        """Stage 1: Uses ColPali visual embeddings + Full-PDF Table Verification to guarantee the best data pages per year."""
        print(f"\n[+] Stage 1: Retrieving Visual Pages for '{company}' across years {available_years}...")
        result_limit = max(1, int(top_k_per_year))

        processor = ColPaliProcessor.from_pretrained(self.colpali_model_id)
        retriever = ColPaliForRetrieval.from_pretrained(
            self.colpali_model_id,
            torch_dtype=torch.float16,
            device_map={"": 0} if self.device == "cuda" else "cpu",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).eval()

        inputs = processor(text=query, return_tensors="pt")
        inputs = {
            k: v.to(device="cuda:0" if self.device == "cuda" else "cpu")
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = retriever(**inputs)
            embeddings = outputs.embeddings if hasattr(outputs, "embeddings") else outputs[0]
            pooled_query = torch.mean(embeddings, dim=1).cpu().to(torch.float32).numpy().tolist()

        selected_pages = []
        for year in available_years:
            res = self.collection.query(
                query_embeddings=pooled_query,
                n_results=15,
                where={"$and": [{"company": company}, {"fiscal_year": year}]},
                include=["metadatas", "distances"],
            )

            candidates = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            if not candidates:
                continue

            source_file = candidates[0].get("source_file")
            pdf_path = os.path.join("./data/raw_pdfs", source_file) if source_file else None
            pdf_doc = pdfium.PdfDocument(pdf_path) if (pdf_path and os.path.exists(pdf_path)) else None

            scored_candidates = []
            seen_pages = set()

            for meta, dist in zip(candidates, dists):
                pg = meta.get("page_number", 0)
                seen_pages.add(pg)
                table_bonus = self._score_page_for_emissions_table(pdf_doc, pg) if pdf_doc else 0.0
                combined_rank = float(dist) - (table_bonus * 0.25)
                scored_candidates.append((combined_rank, table_bonus, meta))

            if pdf_doc is not None:
                for p_idx in range(len(pdf_doc)):
                    pg_num = p_idx + 1
                    if pg_num in seen_pages:
                        continue
                    t_score = self._score_page_for_emissions_table(pdf_doc, pg_num)
                    if t_score >= 10.0:
                        syn_meta = {
                            "company": company,
                            "fiscal_year": year,
                            "page_number": pg_num,
                            "source_file": source_file,
                        }
                        scored_candidates.append((0.4 - (t_score * 0.25), t_score, syn_meta))

                pdf_doc.close()

            scored_candidates.sort(key=lambda x: x[0])
            for _, _, meta in scored_candidates[:result_limit]:
                selected_pages.append(meta)

        retriever = None
        for name in ("processor", "inputs", "outputs", "embeddings", "pooled_query"):
            if name in locals():
                del locals()[name]
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[✓] Stage 1 Complete: ColPali evicted from GPU VRAM.")

        return selected_pages

    def render_page_with_text(self, source_file: str, page_number: int) -> Tuple[Image.Image, str]:
        pdf_path = os.path.join("./data/raw_pdfs", source_file)
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[page_number - 1]
        image = page.render(scale=2.0).to_pil().convert("RGB")
        try:
            page_text = page.get_textpage().get_text_range()
        except Exception:
            page_text = ""
        pdf.close()
        return image, page_text

    def render_page(self, source_file: str, page_number: int) -> Image.Image:
        img, _ = self.render_page_with_text(source_file, page_number)
        return img

    def audit_company(
        self, company: str, available_years: List[int], query: str
    ) -> Optional[ComprehensiveAuditReport]:
        if len(available_years) < 2:
            print(f"[-] Skipping '{company}': Found only {len(available_years)} year ({available_years}).")
            return None

        matched_pages = self.retrieve_pages_for_company(
            query, company, available_years, top_k_per_year=2
        )

        print(f"\n[+] Stage 2: Extracting Visual Data for '{company}' ({len(matched_pages)} pages)...")
        extractor = LlamaVisionESGExtractor()
        extracted_numerical_dataset: Dict[str, List[Dict[str, Any]]] = {}
        narratives_by_year: Dict[int, List[str]] = {}

        try:
            for page_meta in matched_pages:
                year = page_meta["fiscal_year"]
                page_num = page_meta["page_number"]
                source_file = page_meta["source_file"]

                print(f"    ├─ Extracting FY{year} (Page {page_num})...")
                image, page_text = self.render_page_with_text(source_file, page_num)

                extraction = extractor.extract_metrics_from_image(
                    image=image,
                    company_hint=company,
                    year_hint=year,
                    page_text=page_text,
                )
                page_meta["extraction"] = extraction
                page_meta["report_year"] = year
                page_meta["metrics"] = extraction.metrics

                for m in extraction.metrics:
                    if m.value is None:
                        continue

                    norm_val, norm_unit = self.temporal_auditor.normalize_value(m.value, m.unit)
                    if norm_val is None or norm_val <= 0:
                        continue

                    extracted_numerical_dataset.setdefault(m.metric_name, []).append({
                        "metric_name": m.metric_name,
                        "fiscal_year": int(m.fiscal_year or year),
                        "report_year": int(year),
                        "raw_value": m.value,
                        "value": norm_val,
                        "unit": norm_unit,
                        "source_element": getattr(m, "source_element", "table"),
                    })
        except Exception as exc:
            print(f"    [!] Error while extracting visual metrics for '{company}': {exc}")

        self.selected_pages = matched_pages
        self.last_extracted_dataset = matched_pages

        pages_by_year = {}
        for page_info in matched_pages:
            pages_by_year.setdefault(page_info["fiscal_year"], []).append(page_info)

        for year in available_years:
            pages_for_year = pages_by_year.get(year, [])
            year_narratives = []

            for p_info in pages_for_year:
                ext = p_info.get("extraction")
                if ext and ext.qualitative_commitments:
                    for commit in ext.qualitative_commitments:
                        if len(commit.strip()) > 30 and "maintain" not in commit.lower():
                            year_narratives.append(commit.strip())

            if len(year_narratives) < 2 and pages_for_year:
                source_file = pages_for_year[0]["source_file"]
                pdf_path = os.path.join("./data/raw_pdfs", source_file)
                try:
                    pdf = pdfium.PdfDocument(pdf_path)
                    scan_limit = min(14, len(pdf))
                    harvested_sentences = []
                    target_keywords = [
                        "net-zero", "reduce", "carbon neutral", "target", "renewable",
                        "climate pledge", "by 2030", "by 2040", "roadmap", "emissions", "recycled"
                    ]
                    for p_i in range(1, scan_limit):
                        text = pdf[p_i].get_textpage().get_text_range()
                        sentences = [s.strip() for s in text.split(".") if len(s.strip().split()) > 7]
                        for s in sentences:
                            s_clean = " ".join(s.split())
                            if any(kw in s_clean.lower() for kw in target_keywords):
                                if not any(d in s_clean.lower() for d in ["safe harbor", "forward-looking", "table of contents"]):
                                    harvested_sentences.append(s_clean)
                        if len(harvested_sentences) >= 4:
                            break
                    pdf.close()
                    if harvested_sentences:
                        year_narratives.extend(harvested_sentences[:3])
                except Exception as err:
                    print(f"    [!] Error reading narrative from {source_file}: {err}")

            if not year_narratives:
                year_narratives = [f"{company} announced environmental targets and carbon initiatives for fiscal cycle {year}."]

            narratives_by_year[year] = year_narratives
            print(f"    ├─ FY{year} Extracted {len(year_narratives)} narrative commitment(s)")

        print(f"\n[+] Stage 3A: Running Longitudinal Restatement Audit for {company}...")
        temporal_summary = self.temporal_auditor.generate_company_audit_summary(
            company, extracted_numerical_dataset
        )

        print(f"[+] Stage 3B: Running Strategic Ambiguity Scoring for {company}...")
        ambiguity_results: List[AmbiguityAuditResult] = []
        sorted_years = sorted(narratives_by_year.keys())

        for i in range(len(sorted_years) - 1):
            base_yr = sorted_years[i]
            target_yr = sorted_years[i + 1]

            base_combined = " ".join(narratives_by_year.get(base_yr, []))
            target_combined = " ".join(narratives_by_year.get(target_yr, []))

            # Check if substantive numerical targets (%, GW, MT, etc.) were dropped or reduced
            q_base = self._count_quantitative_targets(base_combined)
            q_target = self._count_quantitative_targets(target_combined)
            number_omitted = (q_base > 0 and q_target == 0) or (q_base >= 2 and q_target < q_base)

            sas_res = self.ambiguity_engine.calculate_sas(
                company=company,
                base_year=base_yr,
                target_year=target_yr,
                base_statement=base_combined,
                target_statement=target_combined,
                has_number_omission=number_omitted,
            )
            ambiguity_results.append(sas_res)

        return ComprehensiveAuditReport(
            company=company,
            target_years=available_years,
            extracted_pages=matched_pages,
            temporal_summary=temporal_summary,
            ambiguity_evaluations=ambiguity_results,
        )


def main():
    print("==================================================================")
    print(" Multimodal Temporal Auditor - Autonomous Multi-Company Audit")
    print("==================================================================\n")

    auditor = AutoMultimodalTemporalAuditor()
    discovered_companies = auditor.discover_all_companies_and_years()

    if not discovered_companies:
        print("[!] No indexed companies found in ChromaDB. Please run:")
        print("    .\\venv\\Scripts\\python.exe -m src.ingestion")
        return

    print(f"[+] Discovered {len(discovered_companies)} company/companies in Vector DB:")
    for comp, years in discovered_companies.items():
        print(f"    • {comp}: Available Fiscal Years -> {years}")

    audit_query = (
        "Environmental Data Appendix: Greenhouse Gas Emissions historical summary table, "
        "Scope 1 direct, Scope 2 market-based, Scope 3 metric tons CO2e breakdown"
    )

    for company, years in discovered_companies.items():
        report = auditor.audit_company(company=company, available_years=years, query=audit_query)
        if not report:
            continue

        print("\n" + "=" * 30 + f" AUDIT REPORT: {report.company.upper()} " + "=" * 30)
        print(f"Total Metrics Assessed:      {report.temporal_summary.total_metrics_evaluated}")
        print(f"Longitudinal Consistency:    {report.temporal_summary.consistency_score * 100:.1f}%")
        print(f"Discrepancies Flagged:       {len(report.temporal_summary.flagged_anomalies)}")

        if report.temporal_summary.flagged_anomalies:
            print("\n[!] NUMERICAL DISCREPANCIES:")
            for idx, flag in enumerate(report.temporal_summary.flagged_anomalies, 1):
                print(f"  {idx}. [{flag.flag_type}] {flag.metric_name}")
                print(f"     - Base (FY{flag.base_year}): {flag.normalized_value_base:,.1f} {flag.unit}")
                print(f"     - Comparison (FY{flag.comparison_year}): {flag.normalized_value_comparison:,.1f} {flag.unit}")
                print(f"     - Variance: {flag.percentage_drift:+.2f}%")
                print(f"     - Note: {flag.explanation}")
        else:
            print("\n[✓] No baseline shifts or anomalies detected across the audited numerical metrics.")

        print("\n[!] STRATEGIC AMBIGUITY EVALUATIONS (SAS):")
        for idx, sas in enumerate(report.ambiguity_evaluations, 1):
            print(f"  {idx}. Cycle Transition: FY{sas.base_year} -> FY{sas.target_year}")
            print(f"     - Composite SAS:        {sas.strategic_ambiguity_score} [{sas.flag_level}]")
            print(f"     - Semantic Drift:       {sas.semantic_drift}")
            print(f"     - Linguistic Vagueness: {sas.vagueness_score}")
            print(f"     - Numerical Omission:   {sas.numerical_omission}\n")

        print("=" * 75 + "\n")


if __name__ == "__main__":
    main()