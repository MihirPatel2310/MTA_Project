import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score

# Ensure root directory is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.temporal_auditor import TemporalAuditor, AuditFlag


# ==========================================
# 1. Evaluation Metric Data Models
# ==========================================

class CompanyBenchmarkResult(BaseModel):
    company: str
    total_ground_truth_cases: int
    system_flags_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    inconsistency_detection_rate: float = Field(..., description="IDR = TP / Total GT Inconsistencies")


class GlobalBenchmarkSummary(BaseModel):
    evaluated_companies: List[str]
    total_test_instances: int
    overall_precision: float
    overall_recall: float
    overall_f1: float
    mean_idR: float
    cohens_kappa: float = Field(..., description="Inter-annotator agreement between system and human ground-truth")
    company_breakdowns: List[CompanyBenchmarkResult]


# ==========================================
# 2. Benchmark Evaluator Engine
# ==========================================

class BenchmarkEvaluator:
    """
    Evaluates the MTA pipeline against labeled ground truth disclosures
    to compute Precision, Recall, F1, IDR, and Cohen's Kappa.
    """
    def __init__(
        self,
        ground_truth_path: str = "./data/benchmark/ground_truth_anomalies.json",
        restatement_threshold: float = 0.05
    ):
        self.gt_path = Path(ground_truth_path)
        if not self.gt_path.exists():
            raise FileNotFoundError(f"[!] Ground truth file not found at: {self.gt_path}")

        with open(self.gt_path, "r", encoding="utf-8") as f:
            self.ground_truth: List[Dict[str, Any]] = json.load(f)

        self.auditor = TemporalAuditor(restatement_threshold=restatement_threshold)
        print(f"[+] Loaded {len(self.ground_truth)} benchmark ground-truth test cases.")

    def _fuzzy_metric_match(self, predicted_name: str, gt_name: str) -> bool:
        """Determines if extracted metric matches ground truth label semantically."""
        pred = predicted_name.lower()
        gt = gt_name.lower()

        if pred == gt or gt in pred or pred in gt:
            return True

        keywords = ["scope 1", "scope 2", "scope 3", "ghg", "carbon", "water", "energy", "emissions"]
        for kw in keywords:
            if kw in pred and kw in gt:
                return True
        return False

    def evaluate_predictions(
        self,
        company: str,
        extracted_dataset: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[CompanyBenchmarkResult, List[int], List[int]]:
        """
        Evaluates a single company's extracted metrics against ground truth.
        Returns the company result along with binary vectors (y_true, y_pred) for Kappa calculation.
        """
        # 1. Run Auditor
        summary = self.auditor.generate_company_audit_summary(company, extracted_dataset)
        detected_flags: List[AuditFlag] = summary.flagged_anomalies

        # 2. Filter Ground Truth for this company
        gt_cases = [
            gt for gt in self.ground_truth 
            if gt["company"].lower() in company.lower() or company.lower() in gt["company"].lower()
        ]

        matched_gt_indices = set()
        tp = 0
        fp = 0

        # Verify each detected flag
        for flag in detected_flags:
            match_found = False
            for idx, gt in enumerate(gt_cases):
                target_yr = gt.get("target_fiscal_year", gt.get("base_year"))
                if self._fuzzy_metric_match(flag.metric_name, gt["metric_name"]) and flag.base_year == target_yr:
                    tp += 1
                    matched_gt_indices.add(idx)
                    match_found = True
                    break

            if not match_found:
                fp += 1

        fn = len(gt_cases) - len(matched_gt_indices)

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        idr = round(tp / len(gt_cases), 4) if len(gt_cases) > 0 else 0.0

        # Construct binary vectors for Cohen's Kappa
        # Form pairs of all candidate evaluations
        y_true = [1] * len(gt_cases)
        y_pred = [1 if i in matched_gt_indices else 0 for i in range(len(gt_cases))]

        # Append false positive noise
        for _ in range(fp):
            y_true.append(0)
            y_pred.append(1)

        # Baseline alignment true negatives (consistent metrics correctly untouched)
        consistent_baseline_count = max(0, summary.total_metrics_evaluated - (tp + fp + fn))
        for _ in range(consistent_baseline_count):
            y_true.append(0)
            y_pred.append(0)

        result = CompanyBenchmarkResult(
            company=company,
            total_ground_truth_cases=len(gt_cases),
            system_flags_count=len(detected_flags),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            inconsistency_detection_rate=idr
        )

        return result, y_true, y_pred

    def run_full_benchmark(
        self,
        companies_data: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> GlobalBenchmarkSummary:
        """
        Executes benchmark evaluation across all target companies in the dataset.
        """
        all_breakdowns = []
        global_y_true = []
        global_y_pred = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0

        for company, extracted_data in companies_data.items():
            breakdown, y_t, y_p = self.evaluate_predictions(company, extracted_data)
            all_breakdowns.append(breakdown)
            global_y_true.extend(y_t)
            global_y_pred.extend(y_p)

            total_tp += breakdown.true_positives
            total_fp += breakdown.false_positives
            total_fn += breakdown.false_negatives
            total_gt += breakdown.total_ground_truth_cases

        overall_prec = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = round(2 * (overall_prec * overall_rec) / (overall_prec + overall_rec), 4) if (overall_prec + overall_rec) > 0 else 0.0
        mean_idr = round(total_tp / total_gt, 4) if total_gt > 0 else 0.0

        # Calculate Cohen's Kappa
        try:
            kappa = round(float(cohen_kappa_score(global_y_true, global_y_pred)), 4)
        except Exception:
            kappa = 0.0

        return GlobalBenchmarkSummary(
            evaluated_companies=list(companies_data.keys()),
            total_test_instances=len(global_y_true),
            overall_precision=overall_prec,
            overall_recall=overall_rec,
            overall_f1=overall_f1,
            mean_idR=mean_idr,
            cohens_kappa=kappa,
            company_breakdowns=all_breakdowns
        )


if __name__ == "__main__":
    print("==========================================================")
    print(" Multimodal Temporal Auditor - Academic Benchmark Suite")
    print("==========================================================\n")

    # Mock multi-year extracted dataset reflecting real test cases (Microsoft & BP)
    mock_benchmark_data = {
        "Microsoft": {
            "Scope 3 GHG Emissions": [
                {"fiscal_year": 2020, "report_year": 2021, "value": 11.2, "unit": "MMT CO2e"},
                {"fiscal_year": 2020, "report_year": 2024, "value": 13.8, "unit": "million metric tons co2e"},
                {"fiscal_year": 2023, "report_year": 2023, "value": 15.4, "unit": "MMT CO2e"}
            ],
            "Scope 1 GHG Emissions": [
                {"fiscal_year": 2022, "report_year": 2022, "value": 118000.0, "unit": "MT CO2e"},
                {"fiscal_year": 2023, "report_year": 2023, "value": 121000.0, "unit": "MT CO2e"}
            ]
        },
        "BP": {
            "Scope 1 Operational Emissions": [
                {"fiscal_year": 2022, "report_year": 2022, "value": 31.8, "unit": "MT CO2e"},
                {"fiscal_year": 2022, "report_year": 2024, "value": 30.2, "unit": "MT CO2e"}
            ],
            "Total Fresh Water Consumption": [
                {"fiscal_year": 2022, "report_year": 2022, "value": 240.0, "unit": "m3"},
                {"fiscal_year": 2023, "report_year": 2023, "value": 242.0, "unit": "m3"}
            ]
        }
    }

    evaluator = BenchmarkEvaluator(ground_truth_path="./data/benchmark/ground_truth_anomalies.json")
    summary = evaluator.run_full_benchmark(mock_benchmark_data)

    print("\n" + "="*28 + " EMPIRICAL RESULTS " + "="*28)
    print(f"Evaluated Companies:               {', '.join(summary.evaluated_companies)}")
    print(f"Total Evaluated Test Pairs:        {summary.total_test_instances}")
    print(f"Micro Precision:                   {summary.overall_precision * 100:.1f}%")
    print(f"Micro Recall:                      {summary.overall_recall * 100:.1f}%")
    print(f"Micro F1-Score:                    {summary.overall_f1 * 100:.1f}%")
    print(f"Inconsistency Detection Rate (IDR): {summary.mean_idR * 100:.1f}%")
    print(f"Cohen's Kappa (κ Agreement):       {summary.cohens_kappa:.4f}")
    print("="*75)

    print("\n[+] Company-Level Breakdown:")
    for b in summary.company_breakdowns:
        print(f"  • {b.company}:")
        print(f"      - Ground Truth Cases: {b.total_ground_truth_cases} | Detected: {b.system_flags_count}")
        print(f"      - (TP={b.true_positives}, FP={b.false_positives}, FN={b.false_negatives})")
        print(f"      - Precision: {b.precision:.2f} | Recall: {b.recall:.2f} | F1: {b.f1_score:.2f} | IDR: {b.inconsistency_detection_rate * 100:.1f}%")
    print("="*75)