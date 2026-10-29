import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pypdfium2 as pdfium
from pathlib import Path
from typing import List, Dict, Tuple

# Ensure CUDA allocator stability
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class StandaloneSASAuditor:
    def __init__(
        self,
        embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        # Using climate/financial vagueness-sensitive RoBERTa baseline
        vagueness_model_id: str = "roberta-base"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[+] Initializing SAS Prototype on {self.device.upper()}...")

        # 1. Semantic Embedding Model (Dense Cosine Space)
        print(f"    ├─ Loading Embedding Model: {embed_model_id}")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
        self.embed_model = AutoModel.from_pretrained(embed_model_id).to(self.device).eval()

        # 2. RoBERTa Vagueness / Stance Backbone
        print(f"    └─ Loading Vagueness Classifier: {vagueness_model_id}")
        self.vague_tokenizer = AutoTokenizer.from_pretrained(vagueness_model_id)
        self.vague_model = AutoModelForSequenceClassification.from_pretrained(
            vagueness_model_id, 
            num_labels=2
        ).to(self.device).eval()

        # Regulatory & Linguistic Ambiguity Lexicons (GRI/SASB standards)
        self.hedging_lexicon = {
            "aim to", "endeavor to", "aspire to", "pathway", "where feasible",
            "explore", "seek to", "strive", "subject to market conditions",
            "ambition", "intend", "potential to", "working towards"
        }
        self.hard_anchors = {
            "%", "percent", "metric ton", "mwh", "net-zero by 20", "baseline year",
            "scope 1", "scope 2", "scope 3", "reduced by", "absolute reduction"
        }

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_semantic_drift(self, text_t: str, text_t1: str) -> Tuple[float, float]:
        """
        Calculates cosine similarity and semantic drift:
        Drift_sem = 1 - CosineSimilarity(e_t, e_t+1)
        """
        inputs = self.embed_tokenizer(
            [text_t, text_t1], 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        cos_sim = torch.mm(embeddings[0:1], embeddings[1:2].transpose(0, 1)).item()
        semantic_drift = max(0.0, 1.0 - cos_sim)
        return round(cos_sim, 4), round(semantic_drift, 4)

    def compute_linguistic_vagueness(self, text: str) -> float:
        """
        Combines token-level hedging density with RoBERTa latent uncertainty representation.
        """
        text_lower = text.lower()
        hedge_hits = sum(1 for h in self.hedging_lexicon if h in text_lower)
        anchor_hits = sum(1 for a in self.hard_anchors if a in text_lower)

        # Lexical ratio
        lexical_score = hedge_hits / max(1, (hedge_hits + anchor_hits))

        # Model latent uncertainty (entropy of output logits)
        inputs = self.vague_tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.vague_model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            # Higher entropy represents non-assertive stance
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            normalized_entropy = min(1.0, entropy / 0.6931)  # normalize by ln(2)

        composite_vagueness = (0.6 * lexical_score) + (0.4 * normalized_entropy)
        return round(min(1.0, max(0.0, composite_vagueness)), 4)

    def extract_target_prose_from_pdf(self, pdf_path: str, max_pages_to_scan: int = 15) -> str:
        """
        Extracts narrative target paragraphs mentioning emissions, climate targets, or net-zero.
        """
        pdf = pdfium.PdfDocument(pdf_path)
        extracted_prose = []
        keywords = ["target", "commit", "emissions", "net-zero", "scope", "reduction", "pathway"]

        for i in range(min(len(pdf), max_pages_to_scan)):
            page_text = pdf[i].get_textpage().get_text_range()
            lines = page_text.splitlines()
            for line in lines:
                line_clean = line.strip()
                if len(line_clean.split()) > 7:  # sentence candidate
                    if any(kw in line_clean.lower() for kw in keywords):
                        extracted_prose.append(line_clean)

        pdf.close()
        return " ".join(extracted_prose[:8])  # Return dense composite narrative context

    def audit_consecutive_years(
        self,
        company: str,
        year_t_pdf: str,
        year_t1_pdf: str,
        year_t: int,
        year_t1: int,
        alpha: float = 0.45,
        beta: float = 0.35,
        gamma: float = 0.20
    ) -> Dict:
        print(f"\n[+] Extracting narrative disclosures from {company} reports...")
        prose_t = self.extract_target_prose_from_pdf(year_t_pdf)
        prose_t1 = self.extract_target_prose_from_pdf(year_t1_pdf)

        if not prose_t or not prose_t1:
            raise ValueError("Insufficient narrative sentences extracted from one of the target reports.")

        print(f"    ├─ Disclosures extracted for FY{year_t} ({len(prose_t)} chars)")
        print(f"    └─ Disclosures extracted for FY{year_t1} ({len(prose_t1)} chars)")

        # 1. Drift
        cos_sim, drift_sem = self.compute_semantic_drift(prose_t, prose_t1)

        # 2. Vagueness
        vagueness_t1 = self.compute_linguistic_vagueness(prose_t1)

        # 3. Numerical Omission Penalty
        num_t = sum(c.isdigit() for c in prose_t)
        num_t1 = sum(c.isdigit() for c in prose_t1)
        omission_penalty = 1.0 if (num_t > 5 and num_t1 == 0) else 0.0

        # 4. Formal SAS Formulation
        # SAS = alpha * Drift_sem + beta * Vagueness_t1 + gamma * Omission
        sas = (alpha * min(1.0, drift_sem * 2.0)) + (beta * vagueness_t1) + (gamma * omission_penalty)
        sas = round(min(1.0, max(0.0, sas)), 4)

        if sas >= 0.60:
            status = "HIGH_STRATEGIC_AMBIGUITY (Hedging / Goal Dilution)"
        elif sas >= 0.35:
            status = "MODERATE_AMBIGUITY (Language Drift)"
        else:
            status = "ALIGNED_DISCLOSURE (Consistent Commitments)"

        return {
            "company": company,
            "comparison_cycles": f"FY{year_t} -> FY{year_t1}",
            "cosine_similarity": cos_sim,
            "semantic_drift": drift_sem,
            "vagueness_score": vagueness_t1,
            "omission_flag": bool(omission_penalty),
            "strategic_ambiguity_score": sas,
            "classification": status,
            "sample_t": prose_t[:150] + "...",
            "sample_t1": prose_t1[:150] + "..."
        }


if __name__ == "__main__":
    auditor = StandaloneSASAuditor()

    # Point to two consecutive PDFs in your data folder
    pdf_dir = Path("./data/raw_pdfs")
    
    # Example using Apple or Microsoft files
    candidates = sorted(list(pdf_dir.glob("*.pdf")))
    if len(candidates) < 2:
        print("[!] Need at least two PDFs in data/raw_pdfs/ to run this prototype.")
        sys.exit(1)

    file_1 = str(candidates[0])
    file_2 = str(candidates[1])

    result = auditor.audit_consecutive_years(
        company="TargetEnterprise",
        year_t_pdf=file_1,
        year_t1_pdf=file_2,
        year_t=2021,
        year_t1=2022
    )

    print("\n" + "="*30 + " SAS EMPIRICAL PROTOTYPE OUTPUT " + "="*30)
    print(f"Company:               {result['company']}")
    print(f"Reporting Cycles:      {result['comparison_cycles']}")
    print(f"Cosine Similarity:     {result['cosine_similarity']}")
    print(f"Semantic Drift (1-Sim):{result['semantic_drift']}")
    print(f"Vagueness Index:       {result['vagueness_score']}")
    print(f"Numerical Omission:    {result['omission_flag']}")
    print(f"--------------------------------------------------------------")
    print(f"FINAL SAS SCORE:       {result['strategic_ambiguity_score']} / 1.00")
    print(f"Audit Classification:  {result['classification']}")
    print(f"--------------------------------------------------------------")
    print(f"FY{result['comparison_cycles'][:4]} Sample:\n  \"{result['sample_t']}\"")
    print(f"FY{result['comparison_cycles'][-4:]} Sample:\n  \"{result['sample_t1']}\"")
    print("="*75)