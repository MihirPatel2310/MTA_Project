"""
export_audit_results.py

Exports empirical audit results for Amazon, Apple, and Google (FY2021-FY2024)
produced by the Multimodal Temporal Auditor (MTA) into structured CSV and multi-tab Excel files.

Usage:
    .\venv\Scripts\python.exe export_audit_results.py
"""

import os
import json
import pandas as pd
from openpyxl import Workbook  # type: ignore[reportMissingModuleSource]
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side  # type: ignore[reportMissingModuleSource]
from openpyxl.utils import get_column_letter  # type: ignore[reportMissingModuleSource]

# ==============================================================================
# 1. Empirical Audit Dataset (from main.py runtime outputs)
# ==============================================================================

EXECUTIVE_SUMMARY = [
    {
        "Enterprise": "Amazon",
        "Audited Fiscal Cycles": "FY2021 – FY2024",
        "Evaluated Metrics": 3,
        "Longitudinal Consistency Score (%)": 88.5,
        "Total Discrepancies Flagged": 11,
        "Baseline Restatements": 3,
        "YoY Trajectory Shifts": 8,
        "Peak Composite SAS": 0.3135,
        "SAS Risk Classification": "MODERATE_AMBIGUITY",
        "Key Analytical Takeaway": "Continuous Scope 1 growth (12.11M -> 15.13M MT); retroactive Scope 3 adjustments (-10.8% in FY22, +18.8% in FY23)."
    },
    {
        "Enterprise": "Apple",
        "Audited Fiscal Cycles": "FY2021 – FY2024",
        "Evaluated Metrics": 3,
        "Longitudinal Consistency Score (%)": 76.2,
        "Total Discrepancies Flagged": 9,
        "Baseline Restatements": 5,
        "YoY Trajectory Shifts": 4,
        "Peak Composite SAS": 0.3639,
        "SAS Risk Classification": "MODERATE_AMBIGUITY -> ALIGNED",
        "Key Analytical Takeaway": "Methodological recalculations across Scope 1 (+16.4%) and Scope 2; progressive improvement to aligned disclosure (SAS 0.1650)."
    },
    {
        "Enterprise": "Google",
        "Audited Fiscal Cycles": "FY2021 – FY2024",
        "Evaluated Metrics": 3,
        "Longitudinal Consistency Score (%)": 43.8,
        "Total Discrepancies Flagged": 14,
        "Baseline Restatements": 8,
        "YoY Trajectory Shifts": 6,
        "Peak Composite SAS": 0.5914,
        "SAS Risk Classification": "HIGH_STRATEGIC_AMBIGUITY",
        "Key Analytical Takeaway": "Surge in Scope 2 (+39.65% in FY22->23) directly triggers spike to 0.5914 SAS with Numerical Omission (1.0)."
    }
]

NUMERICAL_DISCREPANCIES = [
    # Amazon
    {"Company": "Amazon", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 39910000.0, "Comparison Value (MT CO2e)": 14220000.0, "Variance (%)": -64.37, "Explanation": "Retroactive recalculation of FY2023 Scope 1 Direct GHG between the 2023 Report and 2024 Report (-64.4% shift)."},
    {"Company": "Amazon", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 12110000.0, "Comparison Value (MT CO2e)": 13020000.0, "Variance (%)": 7.51, "Explanation": "Longitudinal YoY increase of 7.5% in Scope 1 Direct GHG from FY2021 to FY2022."},
    {"Company": "Amazon", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 13020000.0, "Comparison Value (MT CO2e)": 14220000.0, "Variance (%)": 9.22, "Explanation": "Longitudinal YoY increase of 9.2% in Scope 1 Direct GHG from FY2022 to FY2023."},
    {"Company": "Amazon", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 14220000.0, "Comparison Value (MT CO2e)": 15130000.0, "Variance (%)": 6.40, "Explanation": "Longitudinal YoY increase of 6.4% in Scope 1 Direct GHG from FY2023 to FY2024."},
    {"Company": "Amazon", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 4070000.0, "Comparison Value (MT CO2e)": 3060000.0, "Variance (%)": -24.82, "Explanation": "Longitudinal YoY reduction of 24.8% in Scope 2 Indirect GHG from FY2021 to FY2022."},
    {"Company": "Amazon", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 3060000.0, "Comparison Value (MT CO2e)": 2760000.0, "Variance (%)": -9.80, "Explanation": "Longitudinal YoY reduction of 9.8% in Scope 2 Indirect GHG from FY2022 to FY2023."},
    {"Company": "Amazon", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 2760000.0, "Comparison Value (MT CO2e)": 5760000.0, "Variance (%)": 108.70, "Explanation": "Longitudinal YoY increase of 108.7% in Scope 2 Indirect GHG from FY2023 to FY2024."},
    {"Company": "Amazon", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 3 Value Chain GHG", "Base Year": 2022, "Comparison Year": 2024, "Base Value (MT CO2e)": 54980000.0, "Comparison Value (MT CO2e)": 49020000.0, "Variance (%)": -10.84, "Explanation": "Retroactive recalculation of FY2022 Scope 3 Value Chain GHG between the 2022 Report and 2024 Report (-10.8% shift)."},
    {"Company": "Amazon", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 3 Value Chain GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 39910000.0, "Comparison Value (MT CO2e)": 47400000.0, "Variance (%)": 18.77, "Explanation": "Retroactive recalculation of FY2023 Scope 3 Value Chain GHG between the 2023 Report and 2024 Report (+18.8% shift)."},
    {"Company": "Amazon", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 3 Value Chain GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 55360000.0, "Comparison Value (MT CO2e)": 49020000.0, "Variance (%)": -11.45, "Explanation": "Longitudinal YoY reduction of 11.4% in Scope 3 Value Chain GHG from FY2021 to FY2022."},
    {"Company": "Amazon", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 3 Value Chain GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 47400000.0, "Comparison Value (MT CO2e)": 50320000.0, "Variance (%)": 6.16, "Explanation": "Longitudinal YoY increase of 6.2% in Scope 3 Value Chain GHG from FY2023 to FY2024."},

    # Apple
    {"Company": "Apple", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 47430.0, "Comparison Value (MT CO2e)": 55200.0, "Variance (%)": 16.38, "Explanation": "Retroactive recalculation of FY2021 Scope 1 Direct GHG between the 2021 Report and 2022 Report (+16.4% shift)."},
    {"Company": "Apple", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 166380.0, "Comparison Value (MT CO2e)": 55200.0, "Variance (%)": -66.82, "Explanation": "Retroactive recalculation of FY2021 Scope 1 Direct GHG between the 2023 Report and 2024 Report (-66.8% shift)."},
    {"Company": "Apple", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 2780.0, "Comparison Value (MT CO2e)": 3000.0, "Variance (%)": 7.91, "Explanation": "Retroactive recalculation of FY2020 Scope 2 Indirect GHG between the 2023 Report and 2024 Report (+7.9% shift)."},
    {"Company": "Apple", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 2780.0, "Comparison Value (MT CO2e)": 3000.0, "Variance (%)": 7.91, "Explanation": "Retroactive recalculation of FY2021 Scope 2 Indirect GHG between the 2022 Report and 2023 Report (+7.9% shift)."},
    {"Company": "Apple", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 3000.0, "Comparison Value (MT CO2e)": 3400.0, "Variance (%)": 13.33, "Explanation": "Retroactive recalculation of FY2021 Scope 2 Indirect GHG between the 2023 Report and 2024 Report (+13.3% shift)."},
    {"Company": "Apple", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 1003246.0, "Comparison Value (MT CO2e)": 1065405.0, "Variance (%)": 6.20, "Explanation": "Longitudinal YoY increase of 6.2% in Scope 2 Indirect GHG from FY2022 to FY2023."},
    {"Company": "Apple", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 1065405.0, "Comparison Value (MT CO2e)": 1206700.0, "Variance (%)": 13.26, "Explanation": "Longitudinal YoY increase of 13.3% in Scope 2 Indirect GHG from FY2023 to FY2024."},
    {"Company": "Apple", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 3 Value Chain GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 108400.0, "Comparison Value (MT CO2e)": 265800.0, "Variance (%)": 145.20, "Explanation": "Longitudinal YoY increase of 145.2% in Scope 3 Value Chain GHG from FY2021 to FY2022."},
    {"Company": "Apple", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 3 Value Chain GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 265800.0, "Comparison Value (MT CO2e)": 412800.0, "Variance (%)": 55.30, "Explanation": "Longitudinal YoY increase of 55.3% in Scope 3 Value Chain GHG from FY2022 to FY2023."},

    # Google
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 66549.0, "Comparison Value (MT CO2e)": 38700.0, "Variance (%)": -41.85, "Explanation": "Retroactive recalculation of FY2020 Scope 1 Direct GHG between the 2022 Report and 2023 Report (-41.9% shift)."},
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 63521.0, "Comparison Value (MT CO2e)": 45100.0, "Variance (%)": -29.00, "Explanation": "Retroactive recalculation of FY2021 Scope 1 Direct GHG between the 2022 Report and 2023 Report (-29.0% shift)."},
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 10200000.0, "Comparison Value (MT CO2e)": 9183600.0, "Variance (%)": -9.96, "Explanation": "Retroactive recalculation of FY2023 Scope 1 Direct GHG between the 2023 Report and 2024 Report (-10.0% shift)."},
    {"Company": "Google", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 1 Direct GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 3388000.0, "Comparison Value (MT CO2e)": 9183600.0, "Variance (%)": 171.06, "Explanation": "Longitudinal YoY increase of 171.1% in Scope 1 Direct GHG from FY2022 to FY2023."},
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 22000000.0, "Comparison Value (MT CO2e)": 29000000.0, "Variance (%)": 31.82, "Explanation": "Retroactive recalculation of FY2020 Scope 2 Indirect GHG between the 2021 Report and 2022 Report (+31.8% shift)."},
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 5865100.0, "Comparison Value (MT CO2e)": 5116900.0, "Variance (%)": -12.76, "Explanation": "Retroactive recalculation of FY2020 Scope 2 Indirect GHG between the 2023 Report and 2024 Report (-12.8% shift)."},
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 1518643.0, "Comparison Value (MT CO2e)": 509334.0, "Variance (%)": -66.46, "Explanation": "Retroactive recalculation of FY2021 Scope 2 Indirect GHG between the 2021 Report and 2022 Report (-66.5% shift)."},
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 6576200.0, "Comparison Value (MT CO2e)": 5865100.0, "Variance (%)": -10.81, "Explanation": "Retroactive recalculation of FY2021 Scope 2 Indirect GHG between the 2023 Report and 2024 Report (-10.8% shift)."},
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 8045400.0, "Comparison Value (MT CO2e)": 6576200.0, "Variance (%)": -18.26, "Explanation": "Retroactive recalculation of FY2022 Scope 2 Indirect GHG between the 2023 Report and 2024 Report (-18.3% shift)."},
    {"Company": "Google", "Flag Type": "Baseline Restatement", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2023, "Comparison Year": 2024, "Base Value (MT CO2e)": 10200000.0, "Comparison Value (MT CO2e)": 9183600.0, "Variance (%)": -9.96, "Explanation": "Retroactive recalculation of FY2023 Scope 2 Indirect GHG between the 2023 Report and 2024 Report (-10.0% shift)."},
    {"Company": "Google", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 5865100.0, "Comparison Value (MT CO2e)": 6576200.0, "Variance (%)": 12.12, "Explanation": "Longitudinal YoY increase of 12.1% in Scope 2 Indirect GHG from FY2021 to FY2022."},
    {"Company": "Google", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 2 Indirect GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 6576200.0, "Comparison Value (MT CO2e)": 9183600.0, "Variance (%)": 39.65, "Explanation": "Longitudinal YoY increase of 39.6% in Scope 2 Indirect GHG from FY2022 to FY2023."},
    {"Company": "Google", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 3 Value Chain GHG", "Base Year": 2021, "Comparison Year": 2022, "Base Value (MT CO2e)": 2249000.0, "Comparison Value (MT CO2e)": 2828000.0, "Variance (%)": 25.74, "Explanation": "Longitudinal YoY increase of 25.7% in Scope 3 Value Chain GHG from FY2021 to FY2022."},
    {"Company": "Google", "Flag Type": "YoY Discrepancy", "Metric Name": "Scope 3 Value Chain GHG", "Base Year": 2022, "Comparison Year": 2023, "Base Value (MT CO2e)": 2828000.0, "Comparison Value (MT CO2e)": 3762000.0, "Variance (%)": 33.03, "Explanation": "Longitudinal YoY increase of 33.0% in Scope 3 Value Chain GHG from FY2022 to FY2023."}
]

STRATEGIC_AMBIGUITY_RESULTS = [
    # Amazon
    {"Company": "Amazon", "Cycle Transition": "FY2021 -> FY2022", "Base Year": 2021, "Target Year": 2022, "Composite SAS": 0.2781, "Risk Classification": "MODERATE_AMBIGUITY", "Semantic Target Drift": 0.1773, "Linguistic Vagueness Score": 0.5667, "Numerical Omission (0/1)": 0.0},
    {"Company": "Amazon", "Cycle Transition": "FY2022 -> FY2023", "Base Year": 2022, "Target Year": 2023, "Composite SAS": 0.2871, "Risk Classification": "MODERATE_AMBIGUITY", "Semantic Target Drift": 0.3696, "Linguistic Vagueness Score": 0.3451, "Numerical Omission (0/1)": 0.0},
    {"Company": "Amazon", "Cycle Transition": "FY2023 -> FY2024", "Base Year": 2023, "Target Year": 2024, "Composite SAS": 0.3135, "Risk Classification": "MODERATE_AMBIGUITY", "Semantic Target Drift": 0.2558, "Linguistic Vagueness Score": 0.5667, "Numerical Omission (0/1)": 0.0},

    # Apple
    {"Company": "Apple", "Cycle Transition": "FY2021 -> FY2022", "Base Year": 2021, "Target Year": 2022, "Composite SAS": 0.3639, "Risk Classification": "MODERATE_AMBIGUITY", "Semantic Target Drift": 0.3679, "Linguistic Vagueness Score": 0.5667, "Numerical Omission (0/1)": 0.0},
    {"Company": "Apple", "Cycle Transition": "FY2022 -> FY2023", "Base Year": 2022, "Target Year": 2023, "Composite SAS": 0.3556, "Risk Classification": "MODERATE_AMBIGUITY", "Semantic Target Drift": 0.2652, "Linguistic Vagueness Score": 0.6750, "Numerical Omission (0/1)": 0.0},
    {"Company": "Apple", "Cycle Transition": "FY2023 -> FY2024", "Base Year": 2023, "Target Year": 2024, "Composite SAS": 0.1650, "Risk Classification": "ALIGNED_DISCLOSURE", "Semantic Target Drift": 0.0944, "Linguistic Vagueness Score": 0.3500, "Numerical Omission (0/1)": 0.0},

    # Google
    {"Company": "Google", "Cycle Transition": "FY2021 -> FY2022", "Base Year": 2021, "Target Year": 2022, "Composite SAS": 0.1998, "Risk Classification": "ALIGNED_DISCLOSURE", "Semantic Target Drift": 0.1973, "Linguistic Vagueness Score": 0.3171, "Numerical Omission (0/1)": 0.0},
    {"Company": "Google", "Cycle Transition": "FY2022 -> FY2023", "Base Year": 2022, "Target Year": 2023, "Composite SAS": 0.5914, "Risk Classification": "HIGH_STRATEGIC_AMBIGUITY", "Semantic Target Drift": 0.4291, "Linguistic Vagueness Score": 0.5667, "Numerical Omission (0/1)": 1.0},
    {"Company": "Google", "Cycle Transition": "FY2023 -> FY2024", "Base Year": 2023, "Target Year": 2024, "Composite SAS": 0.2359, "Risk Classification": "ALIGNED_DISCLOSURE", "Semantic Target Drift": 0.2796, "Linguistic Vagueness Score": 0.3144, "Numerical Omission (0/1)": 0.0}
]

# ==============================================================================
# 2. Excel & CSV Export Implementation
# ==============================================================================

def export_to_csv():
    """Exports each analytical table to a standalone, publication-ready CSV."""
    df_summary = pd.DataFrame(EXECUTIVE_SUMMARY)
    df_discrepancies = pd.DataFrame(NUMERICAL_DISCREPANCIES)
    df_sas = pd.DataFrame(STRATEGIC_AMBIGUITY_RESULTS)

    df_summary.to_csv("MTA_Audit_Executive_Summary.csv", index=False)
    df_discrepancies.to_csv("MTA_Numerical_Discrepancies.csv", index=False)
    df_sas.to_csv("MTA_Strategic_Ambiguity_Scores.csv", index=False)
    print("[✓] Generated standalone CSV files:")
    print("    - MTA_Audit_Executive_Summary.csv")
    print("    - MTA_Numerical_Discrepancies.csv")
    print("    - MTA_Strategic_Ambiguity_Scores.csv")


def export_to_excel(filename="MTA_MultiCompany_Audit_Results.xlsx"):
    """
    Creates a styled, multi-tab Excel workbook formatted for academic reporting.
    Tabs:
      1. Executive Summary
      2. Numerical Discrepancies
      3. Strategic Ambiguity (SAS)
    """
    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styles
    navy_header_fill = PatternFill(start_color="1A2B4C", end_color="1A2B4C", fill_type="solid")
    steel_sub_fill   = PatternFill(start_color="2C4268", end_color="2C4268", fill_type="solid")
    zebra_light_fill = PatternFill(start_color="F8F9FA", end_color="F8F9FA", fill_type="solid")
    alert_red_fill   = PatternFill(start_color="FDE8E8", end_color="FDE8E8", fill_type="solid")
    alert_amber_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    alert_green_fill = PatternFill(start_color="DEF7EC", end_color="DEF7EC", fill_type="solid")

    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold   = Font(name="Calibri", size=10, bold=True)
    font_body   = Font(name="Calibri", size=10)
    font_title  = Font(name="Calibri", size=16, bold=True, color="1A2B4C")
    font_sub    = Font(name="Calibri", size=10, italic=True, color="6B7280")

    thin_border_side = Side(border_style="thin", color="E5E7EB")
    border_standard  = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)

    # --------------------------------------------------------------------------
    # TAB 1: Executive Summary
    # --------------------------------------------------------------------------
    ws1 = wb.create_sheet(title="Executive Summary")
    ws1.views.sheetView[0].showGridLines = True

    # Title block
    ws1["A1"] = "Multimodal Temporal Auditor (MTA) — Executive Summary"
    ws1["A1"].font = font_title
    ws1["A2"] = "Longitudinal ESG Audit & Strategic Ambiguity Analysis across Amazon, Apple, and Google (FY2021-FY2024)"
    ws1["A2"].font = font_sub

    df_sum = pd.DataFrame(EXECUTIVE_SUMMARY)
    start_row = 4
    
    # Headers
    for col_idx, col_name in enumerate(df_sum.columns, start=1):
        cell = ws1.cell(row=start_row, column=col_idx, value=col_name)
        cell.fill = navy_header_fill
        cell.font = font_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border_standard

    # Rows
    for row_idx, row_data in enumerate(df_sum.values, start=start_row+1):
        fill = zebra_light_fill if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for col_idx, val in enumerate(row_data, start=1):
            cell = ws1.cell(row=row_idx, column=col_idx, value=val)
            cell.font = font_body
            cell.border = border_standard
            cell.fill = fill
            
            # Alignments & formatting
            if col_idx in [1, 2, 10]:
                cell.alignment = Alignment(horizontal="left", vertical="center")
            elif col_idx in [3, 5, 6, 7]:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.number_format = "#,##0"
            elif col_idx == 4:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = '0.0%'
                if val >= 80.0:
                    cell.fill = alert_green_fill
                elif val >= 60.0:
                    cell.fill = alert_amber_fill
                else:
                    cell.fill = alert_red_fill
            elif col_idx == 8:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = '0.0%'
            elif col_idx == 9:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if "HIGH" in str(val):
                    cell.fill = alert_red_fill
                    cell.font = Font(name="Calibri", size=10, bold=True, color="991B1B")
                elif "MODERATE" in str(val):
                    cell.fill = alert_amber_fill
                else:
                    cell.fill = alert_green_fill

    # --------------------------------------------------------------------------
    # TAB 2: Numerical Discrepancies
    # --------------------------------------------------------------------------
    ws2 = wb.create_sheet(title="Numerical Discrepancies")
    ws2.views.sheetView[0].showGridLines = True

    ws2["A1"] = "Stage 3A: Longitudinal Restatements & Trajectory Discrepancies"
    ws2["A1"].font = font_title
    ws2["A2"] = "Detailed registry of flagged historical baseline recalculations and YoY emissions shifts (|Δ| >= 5.0%)"
    ws2["A2"].font = font_sub

    df_disc = pd.DataFrame(NUMERICAL_DISCREPANCIES)
    start_row = 4

    for col_idx, col_name in enumerate(df_disc.columns, start=1):
        cell = ws2.cell(row=start_row, column=col_idx, value=col_name)
        cell.fill = navy_header_fill
        cell.font = font_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_standard

    for row_idx, row_data in enumerate(df_disc.values, start=start_row+1):
        fill = zebra_light_fill if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for col_idx, val in enumerate(row_data, start=1):
            cell = ws2.cell(row=row_idx, column=col_idx, value=val)
            cell.font = font_body
            cell.border = border_standard
            cell.fill = fill

            if col_idx in [1, 2, 3, 9]:
                cell.alignment = Alignment(horizontal="left", vertical="center")
            elif col_idx in [4, 5]:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col_idx in [6, 7]:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = "#,##0.0"
            elif col_idx == 8:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = '+0.00%;-0.00%;0.00%'
                # Highlight significant baseline restatements
                if row_data[1] == "Baseline Restatement":
                    cell.fill = alert_amber_fill
                    cell.font = Font(name="Calibri", size=10, bold=True, color="B45309")

    # --------------------------------------------------------------------------
    # TAB 3: Strategic Ambiguity Scores (SAS)
    # --------------------------------------------------------------------------
    ws3 = wb.create_sheet(title="Strategic Ambiguity (SAS)")
    ws3.views.sheetView[0].showGridLines = True

    ws3["A1"] = "Stage 3B: Strategic Ambiguity & Greenwashing Evaluation"
    ws3["A1"].font = font_title
    ws3["A2"] = "Composite SAS index derived from Semantic Target Drift, Linguistic Vagueness, and Quantitative Numerical Omission"
    ws3["A2"].font = font_sub

    df_sas = pd.DataFrame(STRATEGIC_AMBIGUITY_RESULTS)
    start_row = 4

    for col_idx, col_name in enumerate(df_sas.columns, start=1):
        cell = ws3.cell(row=start_row, column=col_idx, value=col_name)
        cell.fill = navy_header_fill
        cell.font = font_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_standard

    for row_idx, row_data in enumerate(df_sas.values, start=start_row+1):
        fill = zebra_light_fill if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for col_idx, val in enumerate(row_data, start=1):
            cell = ws3.cell(row=row_idx, column=col_idx, value=val)
            cell.font = font_body
            cell.border = border_standard
            cell.fill = fill

            if col_idx in [1, 2]:
                cell.alignment = Alignment(horizontal="left", vertical="center")
            elif col_idx in [3, 4, 9]:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col_idx in [5, 7, 8]:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = '0.0%'
            elif col_idx == 6:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if "HIGH" in str(val):
                    cell.fill = alert_red_fill
                    cell.font = Font(name="Calibri", size=10, bold=True, color="991B1B")
                elif "MODERATE" in str(val):
                    cell.fill = alert_amber_fill
                else:
                    cell.fill = alert_green_fill

    # Auto-adjust column widths across all sheets
    for ws in [ws1, ws2, ws3]:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                # ignore header title strings on row 1 and 2
                if cell.row in [1, 2]:
                    continue
                val_str = str(cell.value or "")
                if len(val_str) > max_len:
                    max_len = len(val_str)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)
        # Freeze top rows
        ws.freeze_panes = "A5"

    wb.save(filename)
    print(f"[✓] Generated formatted Excel workbook: {filename}")


if __name__ == "__main__":
    print("=" * 65)
    print(" MTA Empirical Audit Exporter — Generating Excel & CSV Datasets")
    print("=" * 65)
    export_to_csv()
    export_to_excel()
    print("=" * 65)
    print(" Export complete! All files ready for supervisor review and thesis figures.")
