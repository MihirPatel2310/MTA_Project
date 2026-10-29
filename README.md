# Multimodal Temporal Auditor (MTA)
> **Autonomous Multi-Modal Audit Pipeline for Detecting Historical Baseline Restatements & Strategic Ambiguity in Corporate ESG Disclosures**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Corporate sustainability reporting frequently exhibits two major audit blind spots:
1. **Unannounced Historical Baseline Restatements:** Unnotified retroactive revisions to previously published Scope 1, Scope 2, and Scope 3 greenhouse gas (GHG) figures in subsequent disclosure cycles.
2. **Strategic Linguistic Ambiguity (Greenwashing):** Increased narrative hedging and the selective omission of quantitative reduction targets during periods of operational carbon emissions surges.

The **Multimodal Temporal Auditor (MTA)** provides an automated, end-to-end framework combining Vision-Language Models (VLMs), multi-vector visual retrieval, and temporal audit algorithms to extract, verify, and cross-examine multi-year disclosures directly from raw corporate PDF reports.

---

## 🏗️ System Architecture

The pipeline implements a three-stage decoupled architecture engineered for execution within consumer GPU memory constraints:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: VISUAL RETRIEVAL                       │
│   PDF Reports ──► ColPali v1.3 (Native FP16) ──► ChromaDB Vector Store │
│                   + Table Density Scoring Heuristic                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Eager VRAM Eviction
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: VISUAL EXTRACTION                      │
│   High-Res Render (144 DPI) ──► Qwen2-VL-2B-Instruct                   │
│                                 + Native PDF Table Verification        │
│                                 ──► Validated ESGMetric Schemas        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 3: TEMPORAL AUDITING                      │
│   ┌─────────────────────────────┐   ┌──────────────────────────────┐   │
│   │   Stage 3A: TemporalAuditor │   │ Stage 3B: Ambiguity Engine   │   │
│   │   • Unit Normalization (MT) │   │ • Semantic Drift (Cosine)    │   │
│   │   • Baseline Restatements   │   │ • Linguistic Vagueness       │   │
│   │   • YoY Trajectory Shifts   │   │ • Target Numerical Omission  │   │
│   └─────────────────────────────┘   └──────────────────────────────┘   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION & INTERACTIVE UI                      │
│   Streamlit Web Dashboard (app.py) ──► Plotly SAS Curves & Audit Logs   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Findings (FY2021 – FY2024)

Empirical evaluation across annual corporate sustainability reports from **Amazon**, **Apple**, and **Google** produced the following results:

| Enterprise | Audited Cycles | Longitudinal Consistency | Baseline Restatements | YoY Trajectory Shifts | Strategic Ambiguity Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Amazon** | FY2021–2024 | **88.5%** | 3 | 8 | Moderate Ambiguity (`SAS: 0.28–0.31`) |
| **Apple** | FY2021–2024 | **76.2%** | 5 | 4 | Moderate $\to$ Aligned (`0.3639` $\to$ `0.1650`) |
| **Google** | FY2021–2024 | **43.8%** | 8 | 6 | High Ambiguity Peak (**`SAS: 0.5914`**) |

### Key Analytical Takeaways
* **Google (Primary Empirical Finding):** In the FY2022 $\to$ FY2023 transition, Google's Scope 2 indirect emissions grew by **`+39.65%`** (from $6.58\text{M}$ to $9.18\text{M MT CO}_2\text{e}$), driven by data center AI compute scaling. Concurrently, the Strategic Ambiguity Engine detected an inflection: **`SAS spiked to 0.5914 [HIGH_STRATEGIC_AMBIGUITY]`** with **`Numerical Omission triggering to 1.0`** (substantive quantitative target metrics dropped).
* **Amazon:** Maintained steady Scope 1 growth ($12.11\text{M} \to 15.13\text{M MT CO}_2\text{e}$), with retroactive Scope 3 adjustments ($-10.84\%$ in FY22; $+18.77\%$ in FY23).
* **Apple:** Methodological boundary revisions in Scope 1 ($+16.38\%$) and Scope 2 were accompanied by progressive narrative transparency down to `0.1650 [ALIGNED_DISCLOSURE]`.

---

## ⚙️ Repository Structure

```text
MTA_Project/
├── app.py                         # Streamlit interactive dashboard
├── main.py                        # Autonomous multi-company CLI execution core
├── export_audit_results.py        # Exporter to formatted multi-tab Excel & CSVs
├── generate_all_reports.py        # All-in-one Word, Excel & CSV report generator
├── src/
│   ├── extraction.py              # Qwen2-VL-2B vision-language extractor & OCR parser
│   ├── temporal_auditor.py        # Longitudinal restatement & YoY drift detector
│   ├── strategic_ambiguity.py     # SAS engine (semantic drift, vagueness, omission)
│   └── ingestion.py               # ColPali-based document ingestion and indexing
├── data/
│   ├── raw_pdfs/                  # Input corporate sustainability PDFs (gitignored)
│   └── chromadb/                  # Local vector database storage (gitignored)
├── requirements.txt               # Project dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
* **OS:** Windows 10/11 or Linux
* **Python:** 3.10 or 3.11
* **Hardware:** NVIDIA GPU with $\ge 8\text{ GB}$ VRAM recommended (CUDA-enabled)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
cd <YOUR-REPO-NAME>

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Execution
Place target corporate sustainability reports in `./data/raw_pdfs/` named in the format `<company>_<year>.pdf` (e.g., `Amazon_2022.pdf`), then execute:

```bash
# Run full autonomous audit across all indexed enterprises
python main.py

# Launch the interactive Streamlit dashboard
streamlit run app.py
```

### 4. Exporting Deliverables
```bash
# Generate Word document, styled Excel workbook, and CSV datasets
python generate_all_reports.py
```

---

## 📐 Methodological Formulations

### Strategic Ambiguity Score (SAS)
The composite SAS quantifies tactical linguistic hedging and disclosure obfuscation across consecutive reporting cycles:

$$\text{SAS} = w_1 \cdot \text{SemanticDrift} + w_2 \cdot \text{VaguenessScore} + w_3 \cdot \text{NumericalOmission}$$

Where:
* **$\text{SemanticDrift}$:** $1 - \text{CosineSimilarity}(\vec{e}_{t}, \vec{e}_{t+1})$ calculated using dense contextual embeddings of corporate commitments.
* **$\text{VaguenessScore}$:** Normalized frequency of modal verbs and non-committal hedging constructs (`strive to`, `aim to`, `seek to`, `explore`).
* **$\text{NumericalOmission}$:** Binary indicator triggered ($1.0$) when substantive quantitative targets (`%`, $\text{GW}$, $\text{MT}$) present in cycle $t$ are dropped in cycle $t+1$.

---

## 📄 License & Attribution

Developed as part of a Master's Thesis in Applied Technologies (Computing) at Unitec, New Zealand.  
Licensed under the [MIT License](LICENSE).