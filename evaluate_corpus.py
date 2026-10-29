import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    cohen_kappa_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import main
from main import AutoMultimodalTemporalAuditor as MultimodalTemporalAuditor
# --------------------------------------------------------------------------
# 1. GROUND TRUTH BENCHMARK CORPUS (Human Auditor Baseline: FY21 - FY24)
# --------------------------------------------------------------------------
# Hand-annotated baseline from published corporate reports:
# - Amazon: Market-based carbon accounting transition in FY22; Scope 3 restatement in FY23.
# - Apple: Methodology refinement for Scope 3 emissions in FY22.
# - Google: Grid emissions factor updates in FY22; AI data center load baseline adjustment in FY23-FY24.
GROUND_TRUTH_DATA = {
    # Key: (Company, BaseYear, TargetYear)
    ("Amazon", 2021, 2022): {
        "human_ambiguity_flag": "HIGH_STRATEGIC_AMBIGUITY",
        "known_restatement": True,
        "notes": "Market-based accounting transition shifted emissions trajectory."
    },
    ("Amazon", 2022, 2023): {
        "human_ambiguity_flag": "MODERATE_AMBIGUITY",
        "known_restatement": True,
        "notes": "Scope 3 transportation baseline restated."
    },
    ("Amazon", 2023, 2024): {
        "human_ambiguity_flag": "ALIGNED_DISCLOSURE",
        "known_restatement": False,
        "notes": "Consistent disclosure with granular facility breakdown."
    },
    ("Apple", 2021, 2022): {
        "human_ambiguity_flag": "ALIGNED_DISCLOSURE",
        "known_restatement": False,
        "notes": "Consistent 2030 roadmap metrics."
    },
    ("Apple", 2022, 2023): {
        "human_ambiguity_flag": "MODERATE_AMBIGUITY",
        "known_restatement": True,
        "notes": "Supply chain clean electricity methodology update."
    },
    ("Apple", 2023, 2024): {
        "human_ambiguity_flag": "ALIGNED_DISCLOSURE",
        "known_restatement": False,
        "notes": "Clean transition to carbon neutral product-level metrics."
    },
    ("Google", 2021, 2022): {
        "human_ambiguity_flag": "ALIGNED_DISCLOSURE",
        "known_restatement": False,
        "notes": "Standard annual reporting continuation."
    },
    ("Google", 2022, 2023): {
        "human_ambiguity_flag": "HIGH_STRATEGIC_AMBIGUITY",
        "known_restatement": True,
        "notes": "Scope 2 market-based restatement due to PPA reallocations."
    },
    ("Google", 2023, 2024): {
        "human_ambiguity_flag": "HIGH_STRATEGIC_AMBIGUITY",
        "known_restatement": True,
        "notes": "Data center power baseline reset due to generative AI scale-up."
    }
}

VALID_CATEGORIES = ["ALIGNED_DISCLOSURE", "MODERATE_AMBIGUITY", "HIGH_STRATEGIC_AMBIGUITY"]


def normalize_flag_string(raw_flag: str) -> str:
    """Ensures consistent categorization across pipeline outputs."""
    raw = raw_flag.upper()
    if "HIGH" in raw:
        return "HIGH_STRATEGIC_AMBIGUITY"
    elif "MODERATE" in raw:
        return "MODERATE_AMBIGUITY"
    return "ALIGNED_DISCLOSURE"


def run_live_corpus_evaluation():
    print("=" * 75)
    print("      MULTIMODAL TEMPORAL AUDITOR (MTA) - CORPUS BENCHMARK EVALUATION")
    print("=" * 75)

    auditor = MultimodalTemporalAuditor()
    live_results = []
    total_metrics_extracted = 0
    companies = sorted({key[0] for key in GROUND_TRUTH_DATA})

    for company in companies:
        available_years = sorted({year for key in GROUND_TRUTH_DATA if key[0] == company for year in key[1:]})
        try:
            report = auditor.audit_company(
                company=company,
                available_years=available_years,
                query="Scope 1, Scope 2, and Scope 3 greenhouse gas emissions, renewable energy"
            )

            if not report:
                print(f"    [X] Audit returned empty report for {company}.")
                continue

            # Convert Pydantic report model to dict
            rep_dict = report.model_dump() if hasattr(report, "model_dump") else (report.dict() if hasattr(report, "dict") else vars(report))
            print(f"    [i] Report keys available: {list(rep_dict.keys())}")

            # 1. Harvest Restatement Transitions
            restatement_transitions = set()
            for r_key in ["restatements", "restatement_records", "detected_restatements"]:
                if r_key in rep_dict and rep_dict[r_key]:
                    for r in rep_dict[r_key]:
                        # Handle both dict items and objects
                        b_yr = r.get("base_year") if isinstance(r, dict) else getattr(r, "base_year", None)
                        t_yr = r.get("target_year") if isinstance(r, dict) else getattr(r, "target_year", None)
                        if b_yr and t_yr:
                            restatement_transitions.add((int(b_yr), int(t_yr)))

            # 2. Harvest Ambiguity Scores
            sas_items = []
            for a_key in ["ambiguity_results", "ambiguity_evaluations", "ambiguity_scores", "sas_results", "strategic_ambiguity_scores"]:
                if a_key in rep_dict and rep_dict[a_key]:
                    sas_items = rep_dict[a_key]
                    break

            # Canonical company name matching (Amazon, Apple, Google)
            matched_gt_name = company
            for canonical in ["Amazon", "Apple", "Google"]:
                if canonical.lower() in company.lower():
                    matched_gt_name = canonical
                    break

            for item in sas_items:
                total_metrics_extracted += 1
                d = item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else vars(item))
                base_y = int(d.get("base_year", 0))
                target_y = int(d.get("target_year", 0))
                sas_val = float(d.get("strategic_ambiguity_score", 0.0))
                flag = str(d.get("flag_level", "ALIGNED_DISCLOSURE"))
                drift = float(d.get("semantic_drift", 0.0))
                vagueness = float(d.get("vagueness_score", 0.0))

                has_restatement = (base_y, target_y) in restatement_transitions
                pred_flag = normalize_flag_string(flag)

                live_results.append({
                    "company": matched_gt_name,
                    "base_year": base_y,
                    "target_year": target_y,
                    "pred_ambiguity": pred_flag,
                    "pred_restatement": has_restatement,
                    "sas_score": sas_val,
                    "semantic_drift": drift,
                    "vagueness": vagueness
                })
                print(f"    ├─ [{matched_gt_name}] Cycle FY{base_y}->FY{target_y}: SAS={sas_val:.3f} [{pred_flag}] | Restatement={has_restatement}")

        except Exception as e:
            import traceback
            print(f"    [!] Error auditing {company}: {e}")
            traceback.print_exc()

    # --------------------------------------------------------------------------
    # 2. COMPUTE FORMAL BENCHMARK METRICS
    # --------------------------------------------------------------------------
    if not live_results:
        print("\n[!] No live results could be gathered. Check ChromaDB indexing.")
        return

    y_true_ambiguity = []
    y_pred_ambiguity = []

    y_true_restatement = []
    y_pred_restatement = []

    transition_logs = []

    for item in live_results:
        key = (item["company"], item["base_year"], item["target_year"])
        if key in GROUND_TRUTH_DATA:
            gt = GROUND_TRUTH_DATA[key]
            
            y_true_ambiguity.append(gt["human_ambiguity_flag"])
            y_pred_ambiguity.append(item["pred_ambiguity"])

            y_true_restatement.append(1 if gt["known_restatement"] else 0)
            y_pred_restatement.append(1 if item["pred_restatement"] else 0)

            transition_logs.append({
                "transition": f"{item['company']} (FY{item['base_year']} -> FY{item['target_year']})",
                "ground_truth_ambiguity": gt["human_ambiguity_flag"],
                "mta_ambiguity": item["pred_ambiguity"],
                "ground_truth_restatement": gt["known_restatement"],
                "mta_restatement": item["pred_restatement"],
                "sas_score": item["sas_score"]
            })

    # Metric 1: Cohen's Kappa (Inter-annotator agreement on Strategic Ambiguity)
    kappa = cohen_kappa_score(y_true_ambiguity, y_pred_ambiguity, labels=VALID_CATEGORIES)

    # Metric 2: Restatement Inconsistency Detection Rate (IDR / Recall)
    # IDR = True Positives / (True Positives + False Negatives)
    idr_recall = recall_score(y_true_restatement, y_pred_restatement, zero_division=0)
    precision = precision_score(y_true_restatement, y_pred_restatement, zero_division=0)
    restatement_f1 = f1_score(y_true_restatement, y_pred_restatement, zero_division=0)

    # Metric 3: Multi-class Macro F1 on Ambiguity Detection
    ambiguity_macro_f1 = f1_score(y_true_ambiguity, y_pred_ambiguity, labels=VALID_CATEGORIES, average="macro", zero_division=0)

    # --------------------------------------------------------------------------
    # 3. PRINT FORMATTED RESULTS (READY FOR CHAPTER 5 TABLES)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 32 + " CHAPTER 5 EMPIRICAL RESULTS " + "=" * 32)
    print(f"Total Evaluated Reporting Transitions: {len(y_true_ambiguity)} / {len(GROUND_TRUTH_DATA)}")
    print(f"Total Multi-Year Metrics Extracted:    {total_metrics_extracted}")
    print("-" * 75)
    
    print("1. STRATEGIC AMBIGUITY (SAS) CLASSIFICATION:")
    print(f"   • Cohen's Kappa (κ):                 {kappa:.4f}")
    if kappa >= 0.70:
        strength = "Substantial / Excellent Agreement"
    elif kappa >= 0.40:
        strength = "Moderate / Defensible Agreement"
    else:
        strength = "Fair Agreement"
    print(f"     └─ Agreement Strength:             {strength}")
    print(f"   • Ambiguity Classification Macro F1: {ambiguity_macro_f1:.4f}")

    print("\n2. LONGITUDINAL RESTATEMENT DETECTION (IDR):")
    print(f"   • Inconsistency Detection Rate (IDR): {idr_recall * 100:.1f}% (Sensitivity / Recall)")
    print(f"   • Restatement Precision:             {precision * 100:.1f}%")
    print(f"   • Restatement Detection F1-Score:    {restatement_f1:.4f}")
    print("-" * 75)

    print("\n3. TRANSITION-BY-TRANSITION AUDIT BREAKDOWN:")
    header = f"{'Reporting Cycle':<26} | {'SAS':<6} | {'Pred Flag':<23} | {'GT Flag':<23} | {'Restatement?'}"
    print(header)
    print("-" * len(header))
    for t in transition_logs:
        rest_str = f"MTA:{t['mta_restatement']} / GT:{t['ground_truth_restatement']}"
        print(f"{t['transition']:<26} | {t['sas_score']:<6.3f} | {t['mta_ambiguity']:<23} | {t['ground_truth_ambiguity']:<23} | {rest_str}")

    print("=" * 75)

    # Save output to JSON for easy citation in thesis write-up
    results_payload = {
        "cohens_kappa": round(kappa, 4),
        "inconsistency_detection_rate": round(idr_recall, 4),
        "restatement_precision": round(precision, 4),
        "restatement_f1": round(restatement_f1, 4),
        "ambiguity_macro_f1": round(ambiguity_macro_f1, 4),
        "detailed_transitions": transition_logs
    }
    with open("benchmark_evaluation_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)
    print("\n[✓] Results successfully exported to 'benchmark_evaluation_results.json' for Chapter 5.")


if __name__ == "__main__":
    run_live_corpus_evaluation()