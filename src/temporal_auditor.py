from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import re


class AuditFlag(BaseModel):
    metric_name: str
    company: str
    base_year: int
    comparison_year: int
    reported_value_base: float
    reported_value_comparison: float
    normalized_value_base: float
    normalized_value_comparison: float
    unit: str
    percentage_drift: float
    flag_type: str = Field(..., description="'Baseline Restatement' or 'YoY Discrepancy'")
    explanation: str

    @property
    def base_value(self) -> float:
        return self.normalized_value_base

    @property
    def comparison_value(self) -> float:
        return self.normalized_value_comparison

    @property
    def drift_pct(self) -> float:
        return self.percentage_drift


class TemporalAuditSummary(BaseModel):
    company: str
    total_metrics_evaluated: int
    flagged_anomalies: List[AuditFlag]
    consistency_score: float = Field(..., description="Proportion of consistent metrics (0.0 to 1.0)")


class RestatementRecord(BaseModel):
    metric_name: str
    base_year: int
    target_year: int
    original_value: float
    reported_value: float
    delta_pct: float
    is_restatement: bool


def clean_unit_string(unit: str) -> str:
    if not unit:
        return ""
    text = str(unit).lower().strip()
    text = text.replace("₂", "2").replace("â‚‚", "2").replace("co2-e", "co2e").replace("co2 e", "co2e")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_value(value: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], str]:
    """
    Normalizes any emission figure to canonical Metric Tons (MT CO2e) without double-scaling.
    - Values in the 0.01 to 500 range (e.g., Amazon's 12.11 or 55.36) are in Million Metric Tons (MMT) -> * 1,000,000.
    - Values >= 1,000 (e.g., Apple's 55,200 or Google's 5,865,100) are already in Metric Tons -> * 1.0.
    """
    if value is None:
        return None, "MT CO2e"
    try:
        val_flt = float(value)
    except (ValueError, TypeError):
        return None, "MT CO2e"

    if val_flt <= 0:
        return None, "MT CO2e"

    cleaned_unit = clean_unit_string(unit)

    # 1. If the raw number is small (< 600), it represents Million Metric Tons (MMT)
    if val_flt < 600.0:
        if any(k in cleaned_unit for k in ["thousand", "kt", "kmt"]):
            return val_flt * 1_000.0, "MT CO2e"
        return val_flt * 1_000_000.0, "MT CO2e"

    # 2. If the raw number is between 600 and 10,000 and explicitly labeled in thousands/kilotonnes
    if 600.0 <= val_flt < 10_000.0 and any(k in cleaned_unit for k in ["thousand", "kt", "kmt"]):
        return val_flt * 1_000.0, "MT CO2e"

    # 3. If the raw number is already >= 1,000 (e.g. 52,730 or 5,865,100), it is already in MT CO2e.
    # Never multiply a number >= 1,000 by 1,000,000 (prevents 52 Billion MT hallucinations).
    return val_flt, "MT CO2e"


def canonicalize_metric_key(name: str) -> str:
    n = str(name).lower().strip()
    if "scope 1" in n:
        return "Scope 1 Direct GHG"
    if "scope 2" in n:
        return "Scope 2 Indirect GHG"
    if "scope 3" in n:
        return "Scope 3 Value Chain GHG"
    if any(k in n for k in ["renewable", "electricity", "clean energy"]):
        return "Renewable Energy Percentage"
    if "carbon intensity" in n:
        return "Carbon Intensity"
    return name.strip()


class TemporalAuditor:
    """
    Evaluates multi-year metric consistency and detects silent baseline adjustments
    and year-over-year emissions trajectory anomalies.
    """
    def __init__(self, threshold: float = 0.05, restatement_threshold: Optional[float] = None):
        self.threshold = threshold if restatement_threshold is None else restatement_threshold
        self.restatement_threshold = self.threshold

    @staticmethod
    def normalize_value(value: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], str]:
        return normalize_value(value, unit)

    def audit_metric_history(
        self,
        company: str,
        metric_name: str,
        yearly_records: List[Dict[str, Any]]
    ) -> Tuple[List[AuditFlag], int]:
        flags: List[AuditFlag] = []
        comparisons_made = 0
        if not yearly_records:
            return flags, comparisons_made

        active_threshold = getattr(self, "threshold", getattr(self, "restatement_threshold", 0.05))

        cleaned_records = []
        for rec in yearly_records:
            fy = rec.get("fiscal_year")
            ry = rec.get("report_year", fy)
            raw_v = rec.get("raw_value", rec.get("value"))
            norm_v, norm_u = normalize_value(raw_v, rec.get("unit"))
            if fy is None or norm_v is None or norm_v <= 0:
                continue
            # Filter to valid historical audit window
            if int(fy) < 2019 or int(fy) > 2024:
                continue
            cleaned_records.append({
                "fiscal_year": int(fy),
                "report_year": int(ry) if ry is not None else int(fy),
                "raw_value": float(raw_v) if raw_v is not None else float(norm_v),
                "norm_value": float(norm_v),
                "unit": norm_u,
            })

        if len(cleaned_records) < 2:
            return flags, comparisons_made

        # Deduplicate entries from the exact same (fiscal_year, report_year), keeping the primary total (largest valid value)
        dedup_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for r in cleaned_records:
            key = (r["fiscal_year"], r["report_year"])
            if key not in dedup_map or r["norm_value"] > dedup_map[key]["norm_value"]:
                dedup_map[key] = r

        unique_records = sorted(dedup_map.values(), key=lambda x: (x["fiscal_year"], x["report_year"]))

        by_fiscal_year: Dict[int, List[Dict[str, Any]]] = {}
        for r in unique_records:
            by_fiscal_year.setdefault(r["fiscal_year"], []).append(r)

        # PASS 1: Same-Fiscal-Year Baseline Restatements across different Report Years
        for fy, recs in sorted(by_fiscal_year.items()):
            if len(recs) >= 2:
                recs_sorted = sorted(recs, key=lambda x: x["report_year"])
                for i in range(len(recs_sorted) - 1):
                    curr = recs_sorted[i]
                    nxt = recs_sorted[i + 1]
                    if curr["report_year"] == nxt["report_year"]:
                        continue

                    ratio = nxt["norm_value"] / curr["norm_value"]
                    # Guard against sub-category row mismatches (> 3x or < 0.33x scale jumps)
                    if not (0.33 <= ratio <= 3.0):
                        continue

                    comparisons_made += 1
                    delta = (nxt["norm_value"] - curr["norm_value"]) / curr["norm_value"]
                    if abs(delta) >= active_threshold:
                        drift_pct = round(delta * 100, 2)
                        flags.append(
                            AuditFlag(
                                company=company,
                                metric_name=metric_name,
                                base_year=curr["report_year"],
                                comparison_year=nxt["report_year"],
                                reported_value_base=curr["raw_value"],
                                reported_value_comparison=nxt["raw_value"],
                                normalized_value_base=curr["norm_value"],
                                normalized_value_comparison=nxt["norm_value"],
                                unit="MT CO2e",
                                percentage_drift=drift_pct,
                                flag_type="Baseline Restatement",
                                explanation=(
                                    f"Retroactive recalculation of FY{fy} {metric_name} between the "
                                    f"{curr['report_year']} Report and {nxt['report_year']} Report ({drift_pct:+.1f}% shift)."
                                ),
                            )
                        )

        # PASS 2: Chronological Year-over-Year (YoY) Trajectory Discrepancies (2021-2024)
        canonical_yearly: List[Dict[str, Any]] = []
        for fy in sorted(by_fiscal_year.keys()):
            if fy < 2021:
                continue
            recs_sorted = sorted(by_fiscal_year[fy], key=lambda x: x["report_year"])
            canonical_yearly.append(recs_sorted[-1])

        for i in range(len(canonical_yearly) - 1):
            curr = canonical_yearly[i]
            nxt = canonical_yearly[i + 1]
            ratio = nxt["norm_value"] / curr["norm_value"]
            if not (0.33 <= ratio <= 3.0):
                continue

            comparisons_made += 1
            delta = (nxt["norm_value"] - curr["norm_value"]) / curr["norm_value"]
            if abs(delta) >= active_threshold:
                drift_pct = round(delta * 100, 2)
                direction = "increase" if delta > 0 else "reduction"
                flags.append(
                    AuditFlag(
                        company=company,
                        metric_name=metric_name,
                        base_year=curr["fiscal_year"],
                        comparison_year=nxt["fiscal_year"],
                        reported_value_base=curr["raw_value"],
                        reported_value_comparison=nxt["raw_value"],
                        normalized_value_base=curr["norm_value"],
                        normalized_value_comparison=nxt["norm_value"],
                        unit="MT CO2e",
                        percentage_drift=drift_pct,
                        flag_type="YoY Discrepancy",
                        explanation=(
                            f"Longitudinal YoY {direction} of {abs(drift_pct):.1f}% in {metric_name} "
                            f"from FY{curr['fiscal_year']} to FY{nxt['fiscal_year']}."
                        ),
                    )
                )

        return flags, comparisons_made

    def generate_company_audit_summary(self, company: str, extracted_dataset: Any) -> TemporalAuditSummary:
        grouped_data: Dict[str, List[Dict[str, Any]]] = {}

        if isinstance(extracted_dataset, list):
            for item in extracted_dataset:
                record_dict = (
                    item if isinstance(item, dict)
                    else (item.model_dump() if hasattr(item, "model_dump") else vars(item))
                )
                m_name = record_dict.get("metric_name")
                if not m_name:
                    continue
                ckey = canonicalize_metric_key(m_name)
                grouped_data.setdefault(ckey, []).append(record_dict)

        elif isinstance(extracted_dataset, dict):
            for raw_key, records in extracted_dataset.items():
                ckey = canonicalize_metric_key(raw_key)
                for item in records:
                    record_dict = (
                        item if isinstance(item, dict)
                        else (item.model_dump() if hasattr(item, "model_dump") else vars(item))
                    )
                    grouped_data.setdefault(ckey, []).append(record_dict)

        flagged_anomalies: List[AuditFlag] = []
        total_metrics = len(grouped_data)
        total_comparisons = 0

        for metric_name, records in grouped_data.items():
            anomalies, comps = self.audit_metric_history(
                company=company,
                metric_name=metric_name,
                yearly_records=records,
            )
            flagged_anomalies.extend(anomalies)
            total_comparisons += comps

        # Consistency score reflects the proportion of historical baselines that were NOT silently restated
        restatement_count = sum(1 for a in flagged_anomalies if a.flag_type == "Baseline Restatement")
        denom = max(total_comparisons, total_metrics * 3, 1)
        consistency_score = 1.0 - (restatement_count / denom)

        return TemporalAuditSummary(
            company=company,
            total_metrics_evaluated=total_metrics,
            flagged_anomalies=flagged_anomalies,
            consistency_score=max(0.0, min(1.0, round(consistency_score, 4))),
        )