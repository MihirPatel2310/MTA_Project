# -*- coding: utf-8 -*-
import main

target_years = [2021, 2022, 2023, 2024]
query = (
    "Environmental Data Appendix: Greenhouse gas emissions inventory summary table, "
    "Scope 1 direct, Scope 2 market-based, Scope 3 metric tons CO2e"
)

auditor = main.AutoMultimodalTemporalAuditor()
rep = auditor.audit_company("Amazon", target_years, query)

print("\n" + "=" * 60)
print("STAGE 3A AUDIT EVALUATION SUMMARY")
print("=" * 60)

if hasattr(rep, "temporal_summary") and rep.temporal_summary:
    ts = rep.temporal_summary
    print(f"Metrics evaluated: {ts.total_metrics_evaluated}")
    print(f"Anomalies flagged: {len(ts.flagged_anomalies)}")
    for a in ts.flagged_anomalies:
        print(f"  - {a.metric_name} ({a.base_year}->{a.comparison_year}): {a.percentage_drift:+}% | Type: {a.flag_type}")
        print(f"    Base: {a.base_value} | Comparison: {a.comparison_value}")

print("\n" + "=" * 60)
print("EXTRACTED PAGE METRICS")
print("=" * 60)

pages = getattr(rep, "extracted_pages", [])
if not pages and hasattr(auditor, "last_extracted_dataset"):
    pages = auditor.last_extracted_dataset

print("Total Page Results:", len(pages))
for p in pages:
    ext = p.get("extraction")
    metrics = getattr(ext, "metrics", []) if ext else p.get("metrics", [])
    rep_yr = p.get("fiscal_year") or p.get("report_year")
    pg = p.get("page_number")
    print(f"\n[+] FY{rep_yr} (Page {pg}): {len(metrics)} metrics extracted")
    for m in metrics:
        # Convert Pydantic model to dict if applicable
        m_dict = m.model_dump() if hasattr(m, "model_dump") else (m if isinstance(m, dict) else vars(m))
        name = m_dict.get("metric_name")
        val = m_dict.get("value")
        unit = m_dict.get("unit")
        fy = m_dict.get("fiscal_year") or rep_yr
        print(f"    - FY{fy} | {name}: {val} {unit}")
