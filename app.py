import os
import time
import pandas as pd
import streamlit as st
from AGENTS.data_understanding_agent import DataUnderstandingAgent
from AGENTS.cleaning_agent import CleaningAgent
from AGENTS.statistics_agent import StatisticsAgent
from AGENTS.visualization_agent import VisualizationAgent
from AGENTS.insight_agent import InsightAgent
from AGENTS.recommendation_agent import RecommendationAgent
from AGENTS.report_agent import ReportAgent

st.set_page_config(page_title="EDA Agent", page_icon="🔍", layout="wide")

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* Background */
.stApp { background: #070711; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0d0d1a;
    border: 1px solid #1e1e3f;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 0 20px rgba(99,102,241,0.08);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0d1a;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e1e3f;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-size: 0.82em;
    font-weight: 600;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #0d0d1a;
    border: 1px solid #1e1e3f;
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1em;
    padding: 12px 0;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35);
    transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 32px rgba(99,102,241,0.6);
    transform: translateY(-1px);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d0d1a;
    border: 2px dashed #1e1e3f;
    border-radius: 14px;
    padding: 8px;
}

/* Divider */
hr { border-color: #1e1e3f; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070711; }
::-webkit-scrollbar-thumb { background: #1e1e3f; border-radius: 3px; }

/* Agent node animations */
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(99,102,241,0.6); }
    70%  { box-shadow: 0 0 0 10px rgba(99,102,241,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}
@keyframes glow-green {
    0%, 100% { box-shadow: 0 0 8px rgba(34,197,94,0.4); }
    50%       { box-shadow: 0 0 20px rgba(34,197,94,0.8); }
}
@keyframes scan-line {
    0%   { background-position: 0% 0%; }
    100% { background-position: 0% 100%; }
}

.agent-idle {
    background: #0d0d1a;
    border: 1px solid #1e1e3f;
    border-radius: 14px;
    padding: 14px 8px;
    text-align: center;
    transition: all 0.3s;
}
.agent-running {
    background: linear-gradient(135deg, #1e1040, #0d0d2e);
    border: 1px solid #6366f1;
    border-radius: 14px;
    padding: 14px 8px;
    text-align: center;
    animation: pulse-ring 1.2s infinite;
}
.agent-done {
    background: linear-gradient(135deg, #052010, #071a0f);
    border: 1px solid #22c55e;
    border-radius: 14px;
    padding: 14px 8px;
    text-align: center;
    animation: glow-green 2s infinite;
}
.agent-icon { font-size: 1.6em; margin-bottom: 4px; }
.agent-name { font-size: 0.65em; color: #64748b; font-family: 'JetBrains Mono', monospace; line-height: 1.3; }
.agent-time { font-size: 0.7em; color: #22c55e; font-weight: 700; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

/* Arrow connector */
.arrow { color: #1e1e3f; font-size: 1.2em; display: flex; align-items: center; justify-content: center; padding-top: 28px; }
.arrow-active { color: #6366f1; }
.arrow-done   { color: #22c55e; }

/* Section headers */
.section-header {
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6366f1;
    margin: 28px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e1e3f, transparent);
}

/* Token bar */
.token-bar-bg {
    background: #0d0d1a;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 4px;
}
.token-bar-fill {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
}

/* Status badge */
.badge-success {
    display: inline-block;
    background: linear-gradient(135deg, #052010, #071a0f);
    border: 1px solid #22c55e;
    color: #22c55e;
    font-size: 0.75em;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "eda_state"   not in st.session_state: st.session_state["eda_state"]   = None
if "exec_times"  not in st.session_state: st.session_state["exec_times"]  = {}
if "pipeline_done" not in st.session_state: st.session_state["pipeline_done"] = False

AGENTS_META = [
    ("🔎", "Data\nUnderstanding", "DataUnderstandingAgent", "Shape, dtypes, nulls",        "dataset_info"),
    ("🧹", "Cleaning",            "CleaningAgent",          "Duplicates, nulls",            "cleaned_data"),
    ("📊", "Statistics",          "StatisticsAgent",        "Describe, skew, kurtosis",     "statistics"),
    ("🖼️", "Visualization",       "VisualizationAgent",     "Histograms, heatmap",          "visualizations"),
    ("💡", "Insights",            "InsightAgent",           "LLM insights",                 "insights"),
    ("✅", "Recommend-\nations",  "RecommendationAgent",    "LLM next steps",               "recommendations"),
    ("📄", "Report",              "ReportAgent",            "Markdown report",              "report"),
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:40px 0 8px'>
    <div style='font-size:0.72em;letter-spacing:4px;color:#6366f1;font-weight:700;text-transform:uppercase;margin-bottom:8px'>
        ◈ MULTI-AGENT SYSTEM ◈
    </div>
    <h1 style='font-size:3em;font-weight:700;background:linear-gradient(135deg,#e2e8f0,#6366f1,#8b5cf6);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;line-height:1.1'>
        EDA Agent
    </h1>
    <p style='color:#475569;font-size:0.95em;margin-top:10px;letter-spacing:0.5px'>
        Automated Exploratory Data Analysis &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; Groq LLaMA 3.3 70B
    </p>
</div>
""", unsafe_allow_html=True)

# ── Pipeline Diagram (always visible) ─────────────────────────────────────────
st.markdown("<div class='section-header'>⬡ &nbsp;AGENT WORKFLOW</div>", unsafe_allow_html=True)

exec_times   = st.session_state["exec_times"]
pipeline_done = st.session_state["pipeline_done"]

# Build alternating node + arrow columns
col_widths = []
for i in range(len(AGENTS_META)):
    col_widths.append(1)
    if i < len(AGENTS_META) - 1:
        col_widths.append(0.25)

cols = st.columns(col_widths)
node_placeholders = []
arrow_placeholders = []

col_idx = 0
for i, (icon, name, cls, desc, key) in enumerate(AGENTS_META):
    with cols[col_idx]:
        ph = st.empty()
        node_placeholders.append(ph)
        done = cls in exec_times
        cls_name = "agent-done" if done else "agent-idle"
        time_str = f"<div class='agent-time'>✓ {exec_times[cls]}s</div>" if done else ""
        ph.markdown(f"""
        <div class='{cls_name}'>
            <div class='agent-icon'>{icon}</div>
            <div class='agent-name'>{name}</div>
            {time_str}
        </div>""", unsafe_allow_html=True)
    col_idx += 1

    if i < len(AGENTS_META) - 1:
        with cols[col_idx]:
            aph = st.empty()
            arrow_placeholders.append(aph)
            arrow_cls = "arrow-done" if done else "arrow"
            aph.markdown(f"<div class='{arrow_cls}'>▶</div>", unsafe_allow_html=True)
        col_idx += 1

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>⬡ &nbsp;DATASET INPUT</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    temp_path    = os.path.join("FILE_input", uploaded_file.name)
    dataset_name = os.path.splitext(uploaded_file.name)[0]
    output_dir   = os.path.join("GRAPH", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    preview_df = pd.read_csv(temp_path)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           preview_df.shape[0])
    c2.metric("Columns",        preview_df.shape[1])
    c3.metric("Missing Values", int(preview_df.isnull().sum().sum()))
    c4.metric("Dataset",        dataset_name)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⬡  INITIALIZE AGENT PIPELINE", use_container_width=True, type="primary"):
        st.session_state["exec_times"]   = {}
        st.session_state["pipeline_done"] = False

        AGENT_INSTANCES = [
            DataUnderstandingAgent(), CleaningAgent(), StatisticsAgent(),
            VisualizationAgent(), InsightAgent(), RecommendationAgent(), ReportAgent(),
        ]

        state = {"file_path": temp_path, "output_dir": output_dir, "token_usage": {}}

        for i, (agent, (icon, name, cls, desc, key)) in enumerate(zip(AGENT_INSTANCES, AGENTS_META)):

            # Mark current node as running
            node_placeholders[i].markdown(f"""
            <div class='agent-running'>
                <div class='agent-icon'>{icon}</div>
                <div class='agent-name' style='color:#a5b4fc'>{name}</div>
                <div class='agent-time' style='color:#6366f1'>● running</div>
            </div>""", unsafe_allow_html=True)

            if i > 0:
                arrow_placeholders[i-1].markdown("<div class='arrow-active'>▶</div>", unsafe_allow_html=True)

            t0 = time.time()
            state.update(agent.run(state))
            elapsed = round(time.time() - t0, 2)
            st.session_state["exec_times"][cls] = elapsed

            # Mark done
            node_placeholders[i].markdown(f"""
            <div class='agent-done'>
                <div class='agent-icon'>{icon}</div>
                <div class='agent-name'>{name}</div>
                <div class='agent-time'>✓ {elapsed}s</div>
            </div>""", unsafe_allow_html=True)

            if i < len(AGENTS_META) - 1:
                arrow_placeholders[i].markdown("<div class='arrow-done'>▶</div>", unsafe_allow_html=True)

        st.session_state["eda_state"]    = state
        st.session_state["pipeline_done"] = True

        st.markdown("""
        <div style='text-align:center;margin:20px 0'>
            <span class='badge-success'>◈ &nbsp;PIPELINE COMPLETE &nbsp;◈</span>
        </div>""", unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state["pipeline_done"] and st.session_state["eda_state"]:
        state = st.session_state["eda_state"]

        # ── Token Usage ───────────────────────────────────────────────────
        token_usage = state.get("token_usage", {})
        if token_usage:
            st.markdown("<div class='section-header'>⬡ &nbsp;LLM TOKEN CONSUMPTION</div>", unsafe_allow_html=True)
            token_df  = pd.DataFrame([{"Agent": a, **t} for a, t in token_usage.items()])
            max_tokens = token_df["total_tokens"].max() if not token_df.empty else 1

            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Total Prompt Tokens",      token_df["prompt_tokens"].sum())
            tc2.metric("Total Completion Tokens",  token_df["completion_tokens"].sum())
            tc3.metric("Grand Total Tokens",        token_df["total_tokens"].sum())

            st.markdown("<br>", unsafe_allow_html=True)
            for _, row in token_df.iterrows():
                pct = int(row["total_tokens"] / max_tokens * 100)
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                    <div style='display:flex;justify-content:space-between;font-size:0.78em;color:#94a3b8;margin-bottom:3px'>
                        <span style='font-family:JetBrains Mono,monospace;color:#a5b4fc'>{row['Agent']}</span>
                        <span>{int(row['total_tokens'])} tokens &nbsp;·&nbsp;
                              {int(row['prompt_tokens'])}↑ &nbsp;
                              {int(row['completion_tokens'])}↓</span>
                    </div>
                    <div class='token-bar-bg'>
                        <div class='token-bar-fill' style='width:{pct}%'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Full Results Tabs ──────────────────────────────────────────────
        st.markdown("<div class='section-header'>⬡ &nbsp;ANALYSIS RESULTS</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "  📊  Dataset Info  ",
            "  📈  Statistics  ",
            "  🖼️  Visualizations  ",
            "  💡  Insights  ",
            "  📄  Report  "
        ])

        with tab1:
            info = state["dataset_info"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows",    info["shape"][0])
            m2.metric("Columns", info["shape"][1])
            m3.metric("Nulls",   sum(info["nulls"].values()))
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Column": list(info["dtypes"].keys()),
                "Type":   list(info["dtypes"].values()),
                "Nulls":  [info["nulls"][c] for c in info["dtypes"]]
            }), use_container_width=True, hide_index=True)

        with tab2:
            stats = state["statistics"]
            st.markdown("**Descriptive Statistics**")
            st.dataframe(pd.DataFrame(stats["describe"]), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Skewness**")
                st.dataframe(pd.DataFrame(stats["skewness"].items(), columns=["Column", "Skewness"]),
                             use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Kurtosis**")
                st.dataframe(pd.DataFrame(stats["kurtosis"].items(), columns=["Column", "Kurtosis"]),
                             use_container_width=True, hide_index=True)

        with tab3:
            plots = state["visualizations"]
            if plots:
                cols = st.columns(2)
                for i, p in enumerate(plots):
                    if os.path.exists(p):
                        cols[i % 2].image(p, use_container_width=True)
            else:
                st.info("No visualizations generated.")

        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div style='background:#0d0d1a;border:1px solid #1e1e3f;border-radius:12px;padding:20px'>
                <div style='font-size:0.7em;letter-spacing:2px;color:#6366f1;font-weight:700;margin-bottom:12px'>
                💡 INSIGHTS</div>""", unsafe_allow_html=True)
                st.markdown(state["insights"])
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div style='background:#0d0d1a;border:1px solid #1e1e3f;border-radius:12px;padding:20px'>
                <div style='font-size:0.7em;letter-spacing:2px;color:#22c55e;font-weight:700;margin-bottom:12px'>
                ✅ RECOMMENDATIONS</div>""", unsafe_allow_html=True)
                st.markdown(state["recommendations"])
                st.markdown("</div>", unsafe_allow_html=True)

        with tab5:
            st.markdown(state["report"])
            report_path = os.path.join(output_dir, "eda_report.md")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    st.download_button(
                        "⬇️  Download Full Report",
                        f.read(),
                        file_name=f"{dataset_name}_eda_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
