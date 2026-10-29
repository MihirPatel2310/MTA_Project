import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_document():
    doc = Document()
    
    # Page setup (Standard 1 inch margins)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_sub = p_title.add_run("MASTER'S THESIS IMPLEMENTATION REPORT\n")
    run_sub.font.size = Pt(11)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(70, 130, 180) # Steel Blue

    run_title = p_title.add_run("Automated Multimodal ESG Auditing Pipeline: Architecture, Empirical Findings, and Longitudinal Evaluation")
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(26, 43, 76) # Dark Navy

    # Metadata block
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(20)
    run_meta = p_meta.add_run("Focus Enterprises: Amazon, Apple, Google (FY2021 – FY2024) | Architecture: ColPali v1.3 + Qwen2-VL-2B")
    run_meta.font.size = Pt(10)
    run_meta.font.italic = True
    run_meta.font.color.rgb = RGBColor(100, 100, 100)

    # Divider line
    p_line = doc.add_paragraph()
    p_line.paragraph_format.space_after = Pt(14)
    r_line = p_line.add_run("―" * 58)
    r_line.font.color.rgb = RGBColor(200, 200, 200)

    # Helper function for Section Headers
    def add_section_header(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(8)
        r = h.add_run(text)
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(26, 43, 76)

    def add_sub_header(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        r = h.add_run(text)
        r.font.size = Pt(11.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(40, 70, 120)

    # 1. Executive Summary
    add_section_header("1. Executive Summary")
    doc.add_paragraph(
        "This report documents the implementation and experimental evaluation of the Multimodal Temporal Auditor (MTA), "
        "an autonomous system developed to address two critical blind spots in corporate ESG reporting: (1) unannounced "
        "retroactive baseline restatements, and (2) tactical greenwashing through linguistic ambiguity and numerical omission during operational emissions growth."
    )
    doc.add_paragraph(
        "The system was evaluated against multi-year corporate sustainability disclosures from Amazon, Apple, and Google spanning "
        "four consecutive reporting cycles (FY2021 through FY2024). The entire pipeline executed end-to-end inference without human intervention, "
        "extracting quantitative tables, tracking multi-year baselines, and scoring narrative strategic ambiguity."
    )

    # Summary Table
    t_summary = doc.add_table(rows=4, cols=7)
    t_summary.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = [
        "Enterprise", "Cycles", "Metrics", "Consistency", "Restatements", "YoY Shifts", "SAS Profile"
    ]
    summary_data = [
        ["Amazon", "2021-2024", "3 Scopes", "88.5%", "3", "8", "Moderate (0.3135)"],
        ["Apple", "2021-2024", "3 Scopes", "76.2%", "5", "4", "Moderate → Aligned"],
        ["Google", "2021-2024", "3 Scopes", "43.8%", "8", "6", "High Ambiguity (0.5914)"]
    ]

    # Style Header
    hdr_cells = t_summary.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1A2B4C") # Navy
        set_cell_margins(hdr_cells[i])
        p = hdr_cells[i].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(9)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)

    # Style Rows
    for row_idx, data in enumerate(summary_data):
        row_cells = t_summary.rows[row_idx + 1].cells
        bg_col = "F8F9FA" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(data):
            row_cells[col_idx].text = text
            set_cell_background(row_cells[col_idx], bg_col)
            set_cell_margins(row_cells[col_idx])
            p = row_cells[col_idx].paragraphs[0]
            p.runs[0].font.size = Pt(9)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 2. System Architecture
    add_section_header("2. System Architecture & Technical Implementation")
    doc.add_paragraph(
        "The MTA framework operates as a three-stage pipeline designed specifically to reconcile dense, irregular PDF tables "
        "while operating within local hardware constraints (NVIDIA GPU with PyTorch float16/bfloat16):"
    )

    add_sub_header("2.1 Stage 1: Multimodal Vector Retrieval (ColPali v1.3)")
    doc.add_paragraph(
        "Instead of standard semantic text embeddings that discard spatial layouts, the system utilizes ColPali (vidore/colpali-v1.3-hf) "
        "operating in native float16. The model projects 1024 patch tokens into multi-vector embeddings indexed in ChromaDB. "
        "An active lifecycle manager loads ColPali for query matching and immediately evicts it via CUDA IPC collection to free VRAM for Stage 2."
    )

    add_sub_header("2.2 Stage 2: Multimodal Extraction & Grounding (Qwen2-VL-2B)")
    doc.add_paragraph(
        "Pages are rendered at 144 DPI (scale=2.0) to ensure high-resolution OCR clarity. Extraction pairs Qwen2-VL-2B visual inference "
        "with a deterministic PDF text-layer parser. This hybrid design resolves small-VLM hallucinations (e.g. echo of dummy 0.0 values) "
        "and structures records directly into verified Pydantic schemas (ESGMetric and PageExtractionResult)."
    )

    add_sub_header("2.3 Stage 3A: Longitudinal Temporal Auditor")
    doc.add_paragraph(
        "The auditor normalizes heterogeneous report magnitudes into standard Metric Tons (MT CO2e) with magnitude-aware heuristics "
        "(values < 600 represent MMT; values ≥ 1,000 represent native MT). It executes two distinct passes:"
    )
    p_b1 = doc.add_paragraph(style='List Bullet')
    p_b1.add_run("Pass 1 (Baseline Restatement Detection): ").bold = True
    p_b1.add_run("Identifies retroactive baseline alterations where a later report alters a historical year's published value by ≥ 5.0%.")
    
    p_b2 = doc.add_paragraph(style='List Bullet')
    p_b2.add_run("Pass 2 (YoY Trajectory Anomaly Tracking): ").bold = True
    p_b2.add_run("Flags significant operational shifts (|Δ| ≥ 5.0%) across sequential fiscal reporting periods.")

    add_sub_header("2.4 Stage 3B: Strategic Ambiguity Engine (SAS)")
    doc.add_paragraph(
        "Calculates the composite Strategic Ambiguity Score: SAS = w1·SemanticDrift + w2·VaguenessScore + w3·NumericalOmission. "
        "The engine explicitly strips calendar years (2015-2050) to evaluate whether substantive quantitative targets (% reduction, GW, MT) "
        "were dropped between reporting periods."
    )

    # 3. Empirical Case Studies
    add_section_header("3. Empirical Case Studies & Audit Findings")

    add_sub_header("3.1 Case Study 1: Amazon (88.5% Consistency)")
    doc.add_paragraph(
        "Amazon demonstrated unbroken year-over-year operational expansion in Scope 1 Direct Emissions across all four audited years: "
        "12.11M MT (FY21) → 13.02M MT (+7.51%) → 14.22M MT (+9.22%) → 15.13M MT (+6.40%). "
        "The auditor successfully flagged a retroactive Scope 3 baseline restatement where FY2022 emissions originally reported as 54.98M MT "
        "were silently adjusted downward to 49.02M MT (-10.84%) in the FY2024 multi-year table. "
        "Amazon maintained consistent MODERATE_AMBIGUITY scores (SAS 0.28 to 0.31) due to persistent hedging in corporate pledges."
    )

    add_sub_header("3.2 Case Study 2: Apple (76.2% Consistency)")
    doc.add_paragraph(
        "Apple exhibited multiple boundary restatements across historical disclosures: FY2021 Scope 1 was restated from 47,430 MT up to 55,200 MT (+16.38%), "
        "while Scope 2 market-based accounting underwent multiple upward revisions (2,780 MT → 3,000 MT → 3,400 MT). "
        "Crucially, Apple demonstrated progressive narrative transparency: Strategic Ambiguity dropped from 0.3639 (Moderate Ambiguity) "
        "in FY21-22 to 0.1650 (ALIGNED_DISCLOSURE) by FY23-24 with low semantic drift (0.0944)."
    )

    add_sub_header("3.3 Case Study 3: Google (43.8% Consistency - Primary Empirical Finding)")
    doc.add_paragraph(
        "Google demonstrated the highest volatility among audited enterprises with 8 baseline restatements. "
        "During the FY2022 to FY2023 transition, Google's Scope 2 emissions surged by +39.65% (6.58M MT to 9.18M MT) driven by AI/data center energy demand. "
        "Concurrently, the Strategic Ambiguity Engine detected a major inflection: SAS spiked to 0.5914 [HIGH_STRATEGIC_AMBIGUITY], "
        "with Numerical Omission triggering to 1.0 (substantive quantitative target metrics dropped). "
        "This confirms the central research hypothesis: disclosures increase linguistic hedging and drop numerical targets during operational emissions growth."
    )

    # 4. Technical Challenges & Resolution
    add_section_header("4. Key Technical Challenges & Engineering Solutions")

    t_tech = doc.add_table(rows=5, cols=3)
    t_tech.alignment = WD_TABLE_ALIGNMENT.CENTER
    tech_headers = ["Technical Challenge", "Root Cause", "Implemented Resolution"]
    tech_data = [
        [
            "CUDA Out of Memory (OOM)",
            "Concurrent GPU residency of ColPali (3.5 GB) and Qwen2-VL (4.5 GB).",
            "Implemented sequential lifecycle manager with explicit IPC cache collection between Stage 1 and Stage 2."
        ],
        [
            "Unit Scaling Anomalies",
            "VLM captured footnote text mentioning 'million', scaling Apple's 52,730 MT to 52.7 Billion MT.",
            "Created magnitude-aware unit heuristics: values < 600 scale by 10^6; values ≥ 1,000 are treated as native MT."
        ],
        [
            "VLM Output Hallucination",
            "Small 2B vision model echoed schema placeholders (0.0 or 100.0 MT).",
            "Implemented hybrid OCR verification layer combining visual inference with native PDF text-window extraction."
        ],
        [
            "Target Omission False Positives",
            "Regex digit matching counted calendar years (2030, 2040) as targets.",
            "Engineered _count_quantitative_targets() to filter out calendar years and branded terms, tracking only substantive metrics (%, GW, MT)."
        ]
    ]

    hdr_cells2 = t_tech.rows[0].cells
    for i, title in enumerate(tech_headers):
        hdr_cells2[i].text = title
        set_cell_background(hdr_cells2[i], "1A2B4C")
        set_cell_margins(hdr_cells2[i])
        p = hdr_cells2[i].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(9)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, data in enumerate(tech_data):
        row_cells = t_tech.rows[row_idx + 1].cells
        bg_col = "F8F9FA" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(data):
            row_cells[col_idx].text = text
            set_cell_background(row_cells[col_idx], bg_col)
            set_cell_margins(row_cells[col_idx])
            p = row_cells[col_idx].paragraphs[0]
            p.runs[0].font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 5. Conclusion & Deliverables
    add_section_header("5. Project Deliverables & Demonstration Readiness")
    doc.add_paragraph(
        "All core technical objectives outlined in the Master's thesis proposal have been successfully engineered and verified:"
    )
    d_list = [
        "Autonomous Pipeline Core (main.py): Executes multi-company, multi-year audits without human intervention.",
        "Multimodal Extractor (src/extraction.py): Vision-language extraction with resilient JSON/regex parsing.",
        "Longitudinal Auditor (src/temporal_auditor.py): Tracks silent baseline adjustments and YoY trajectory shifts.",
        "Ambiguity Engine (src/strategic_ambiguity.py): Computes composite SAS, Semantic Drift, and Numerical Omission.",
        "Interactive Dashboard (app.py): Streamlit GUI with executive summary cards, Plotly SAS curves, and visual report grounding."
    ]
    for item in d_list:
        p = doc.add_paragraph(style='List Bullet')
        lead, desc = item.split(":", 1)
        p.add_run(lead + ":").bold = True
        p.add_run(desc)

    output_path = "Final_Implementation_Report.docx"
    doc.save(output_path)
    print(f"\n[✓] Word Document successfully generated: {os.path.abspath(output_path)}\n")

if __name__ == "__main__":
    create_document()