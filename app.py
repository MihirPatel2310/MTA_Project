import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import AutoMultimodalTemporalAuditor

st.set_page_config(
    page_title="Multimodal Temporal Auditor (MTA)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #dee2e6;
    }
    .flag-high { color: #d90429; font-weight: bold; }
    .flag-moderate { color: #f77f00; font-weight: bold; }
    .flag-aligned { color: #2b9348; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_auditor():
    return AutoMultimodalTemporalAuditor()


def main():
    st.title("🔍 Multimodal Temporal Auditor (MTA)")
    st.caption("Detecting Silent Restatements and Strategic Ambiguity in Multi-Year ESG Disclosures")

    with st.spinner("Connecting to Vector DB and Analytical Engines..."):
        auditor = load_auditor()

    discovered_companies = auditor.discover_all_companies_and_years()

    if not discovered_companies:
        st.error("No indexed reports found in ChromaDB! Please run `python -m src.ingestion` first.")
        return

    st.sidebar.header("Audit Configuration")
    company_options = list(discovered_companies.keys())
    selected_company = st.sidebar.selectbox("Select Target Enterprise", company_options)

    available_years = discovered_companies[selected_company]
    st.sidebar.info(f"**Discovered Disclosure Years:** {', '.join(map(str, available_years))}")

    query = st.sidebar.text_area(
        "Audit Query Target",
        value="Environmental Data Appendix: Greenhouse Gas Emissions historical summary table, Scope 1 direct, Scope 2 market-based, Scope 3 metric tons CO2e breakdown",
        height=100,
    )

    run_audit_btn = st.sidebar.button("Run Multi-Modal Audit", type="primary", use_container_width=True)

    cache_key = f"report_{selected_company}"

    if run_audit_btn or cache_key in st.session_state:
        if run_audit_btn:
            with st.spinner(f"Running Multi-Modal Temporal Audit for {selected_company}..."):
                report = auditor.audit_company(
                    company=selected_company, available_years=available_years, query=query
                )
                st.session_state[cache_key] = report
        else:
            report = st.session_state[cache_key]

        if not report:
            st.warning(f"Audit could not be completed for {selected_company}. Ensure at least two fiscal years are indexed.")
            return

        st.markdown("### Executive Audit Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Audited Entity", report.company)
        with col2:
            st.metric("Evaluated Metrics", report.temporal_summary.total_metrics_evaluated)
        with col3:
            st.metric("Longitudinal Consistency", f"{report.temporal_summary.consistency_score * 100:.1f}%")
        with col4:
            st.metric("Flagged Anomalies", len(report.temporal_summary.flagged_anomalies))

        st.divider()

        tab1, tab2, tab3 = st.tabs([
            "📊 Numerical Restatements & Drift",
            "📈 Strategic Ambiguity (SAS)",
            "📄 Visual Report Grounding",
        ])

        with tab1:
            st.subheader("Longitudinal Restatement & Baseline Discrepancies")
            anomalies = getattr(report.temporal_summary, "flagged_anomalies", [])

            if anomalies:
                table_data = []
                for item in anomalies:
                    r = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else vars(item))
                    m_name = r.get("metric_name", "Emissions Metric")
                    b_yr = r.get("base_year")
                    c_yr = r.get("comparison_year")
                    base_val = r.get("normalized_value_base", r.get("reported_value_base"))
                    comp_val = r.get("normalized_value_comparison", r.get("reported_value_comparison"))
                    drift = r.get("percentage_drift", 0.0)
                    unit_str = r.get("unit", "MT CO2e")
                    flag_type = r.get("flag_type", "Restatement")
                    exp = r.get("explanation", "")

                    if base_val is None or comp_val is None:
                        continue

                    table_data.append({
                        "Metric": m_name,
                        "Base Year": f"FY{b_yr}" if b_yr else "N/A",
                        "Comp Year": f"FY{c_yr}" if c_yr else "N/A",
                        "Base Value (Normalized)": f"{base_val:,.1f} {unit_str}".strip(),
                        "Comparison Value (Normalized)": f"{comp_val:,.1f} {unit_str}".strip(),
                        "Drift (%)": f"{drift:+.2f}%",
                        "Flag Type": flag_type,
                        "Explanation": exp,
                    })

                if table_data:
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                else:
                    st.info("No significant longitudinal restatements detected.")
            else:
                st.info("No longitudinal restatements detected across reported metrics.")

        with tab2:
            st.subheader("Strategic Ambiguity & Linguistic Hedging Analysis")
            if report.ambiguity_evaluations:
                sas_data = []
                for s in report.ambiguity_evaluations:
                    sas_data.append({
                        "Cycle Transition": f"FY{s.base_year} ➔ FY{s.target_year}",
                        "Composite SAS": s.strategic_ambiguity_score,
                        "Risk Classification": s.flag_level,
                        "Semantic Target Drift": s.semantic_drift,
                        "Linguistic Vagueness": s.vagueness_score,
                        "Numerical Omission": s.numerical_omission,
                        "Base Narrative": s.base_statement,
                        "Target Narrative": s.target_statement,
                    })
                df_sas = pd.DataFrame(sas_data)

                st.dataframe(
                    df_sas[["Cycle Transition", "Composite SAS", "Risk Classification", "Semantic Target Drift", "Linguistic Vagueness", "Numerical Omission"]],
                    use_container_width=True,
                )

                fig_sas = go.Figure()
                fig_sas.add_trace(go.Bar(
                    x=df_sas["Cycle Transition"],
                    y=df_sas["Composite SAS"],
                    name="Composite SAS Index",
                    marker_color="#1d3557",
                ))
                fig_sas.add_trace(go.Scatter(
                    x=df_sas["Cycle Transition"],
                    y=df_sas["Semantic Target Drift"],
                    name="Semantic Target Drift",
                    mode="lines+markers",
                    line=dict(color="#e63946", width=3),
                ))
                fig_sas.update_layout(
                    title="Longitudinal SAS Drift Across Consecutive Reporting Cycles",
                    yaxis_title="Score / Index (0.0 to 1.0)",
                    barmode="group",
                )
                st.plotly_chart(fig_sas, use_container_width=True)

                st.markdown("#### Qualitative Narrative Drilldown")
                for item in sas_data:
                    with st.expander(f"Inspect Transition: {item['Cycle Transition']} [{item['Risk Classification']}]"):
                        st.markdown(f"**Base Cycle Narrative:**\n> {item['Base Narrative']}")
                        st.markdown(f"**Subsequent Cycle Narrative:**\n> {item['Target Narrative']}")
            else:
                st.info("Insufficient multi-year narrative text available to compute Strategic Ambiguity Scores.")

        with tab3:
            st.subheader("Retrieved Multi-Year Infographics & Tables")
            st.write("Actual ColPali + Table-Verified pages extracted during Stage 1 & 2:")

            extracted_pages = getattr(report, "extracted_pages", [])
            if extracted_pages:
                cols = st.columns(min(4, len(extracted_pages)))
                for idx, p_meta in enumerate(extracted_pages):
                    with cols[idx % len(cols)]:
                        yr = p_meta.get("fiscal_year")
                        pg_num = p_meta.get("page_number")
                        source_pdf = p_meta.get("source_file")
                        st.markdown(f"**FY{yr} — Page {pg_num}**")
                        img = auditor.render_page(source_pdf, pg_num)
                        st.image(img, caption=f"{source_pdf} (p. {pg_num})", use_container_width=True)
            else:
                st.caption("No pages rendered.")


if __name__ == "__main__":
    main()