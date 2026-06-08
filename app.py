import os
import time
import pandas as pd
import streamlit as st
from WORKFLOW.langgraph_workflow import app
from AGENTS.data_understanding_agent import DataUnderstandingAgent
from AGENTS.cleaning_agent import CleaningAgent
from AGENTS.statistics_agent import StatisticsAgent
from AGENTS.visualization_agent import VisualizationAgent
from AGENTS.insight_agent import InsightAgent
from AGENTS.recommendation_agent import RecommendationAgent
from AGENTS.report_agent import ReportAgent

st.set_page_config(page_title="EDA Agent", page_icon="🔍", layout="wide")


# ── Session state init ─────────────────────────────────────────────────────────
if "eda_state" not in st.session_state: st.session_state["eda_state"] = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center'>🔍 EDA Agent</h1>
<p style='text-align:center;color:gray'>Automated Exploratory Data Analysis · LangGraph + Groq LLaMA 3.3 70B</p>
""", unsafe_allow_html=True)
st.divider()

# ── Upload ──────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    temp_path = os.path.join("FILE_input", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    dataset_name = os.path.splitext(uploaded_file.name)[0]
    output_dir   = os.path.join("GRAPH", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    preview_df = pd.read_csv(temp_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("📋 Rows",    preview_df.shape[0])
    c2.metric("📌 Columns", preview_df.shape[1])
    c3.metric("❗ Missing", int(preview_df.isnull().sum().sum()))
    st.divider()

    # Auto-run DataUnderstanding + Cleaning on upload
    if st.session_state["eda_state"] is None or \
       st.session_state["eda_state"].get("file_path") != temp_path:
        with st.spinner("Preparing dataset..."):
            base_state = {"file_path": temp_path, "output_dir": output_dir, "token_usage": {}}
            base_state.update(DataUnderstandingAgent().run(base_state))
            base_state.update(CleaningAgent().run(base_state))
            st.session_state["eda_state"] = base_state
            st.session_state["chat_history"] = []
            st.session_state["last_result"]  = None

    if st.button("▶️ Run EDA Pipeline", use_container_width=True, type="primary"):

        AGENTS_META = [
            ("🔎 Data Understanding", "DataUnderstandingAgent", "Shape, dtypes, nulls",          "dataset_info"),
            ("🧹 Cleaning",           "CleaningAgent",          "Drop duplicates, fill nulls",   "cleaned_data"),
            ("📊 Statistics",         "StatisticsAgent",        "Describe, skew, kurtosis",      "statistics"),
            ("🖼️ Visualization",      "VisualizationAgent",     "Histograms, heatmap, counts",   "visualizations"),
            ("💡 Insights",           "InsightAgent",           "LLM-generated insights",        "insights"),
            ("✅ Recommendations",    "RecommendationAgent",    "LLM-generated next steps",      "recommendations"),
            ("📄 Report",             "ReportAgent",            "Compile markdown report",       "report"),
        ]
        AGENT_INSTANCES = [
            DataUnderstandingAgent(), CleaningAgent(), StatisticsAgent(),
            VisualizationAgent(), InsightAgent(), RecommendationAgent(), ReportAgent(),
        ]

        st.markdown("### ⚙️ Agent Pipeline Execution")
        agent_cols   = st.columns(len(AGENTS_META))
        placeholders = []
        for i, (label, cls, desc, _) in enumerate(AGENTS_META):
            with agent_cols[i]:
                st.markdown(f"""
                <div style='text-align:center;padding:8px;border-radius:8px;
                            background:#1e1e2e;border:1px solid #444'>
                    <div style='font-size:1.3em'>{label.split()[0]}</div>
                    <div style='font-size:0.7em;color:#aaa'>{cls}</div>
                </div>""", unsafe_allow_html=True)
                placeholders.append(st.empty())

        st.divider()
        cls_list = [m[1] for m in AGENTS_META]
        state    = {"file_path": temp_path, "output_dir": output_dir, "token_usage": {}}
        exec_times = {}

        for agent, (label, cls, desc, key) in zip(AGENT_INSTANCES, AGENTS_META):
            placeholders[cls_list.index(cls)].markdown("🟡 Running...")
            t0 = time.time()
            state.update(agent.run(state))
            elapsed = round(time.time() - t0, 2)
            exec_times[cls] = elapsed
            placeholders[cls_list.index(cls)].markdown(f"✅ **{elapsed}s**")

        st.success("🎉 EDA Pipeline Complete!")
        st.session_state["eda_state"] = state

        # ── Agent Summary Cards ───────────────────────────────────────────
        st.markdown("### 📋 Agent Results Summary")
        for label, cls, desc, key in AGENTS_META:
            with st.expander(f"{label}  —  `{exec_times[cls]}s`", expanded=False):
                st.caption(desc)
                if key == "dataset_info":
                    info = state["dataset_info"]
                    st.write(f"**Shape:** {info['shape']}")
                    st.dataframe(pd.DataFrame({
                        "Column": list(info["dtypes"].keys()),
                        "Type":   list(info["dtypes"].values()),
                        "Nulls":  [info["nulls"][c] for c in info["dtypes"]]
                    }), use_container_width=True, hide_index=True)
                elif key == "cleaned_data":
                    c = state["cleaned_data"]
                    col1, col2 = st.columns(2)
                    col1.metric("Shape after cleaning", str(c["shape_after_cleaning"]))
                    col2.metric("Duplicates removed",   c["duplicates_removed"])
                elif key == "statistics":
                    st.dataframe(pd.DataFrame(state["statistics"]["skewness"].items(),
                                              columns=["Column", "Skewness"]),
                                 use_container_width=True, hide_index=True)
                elif key == "visualizations":
                    plots = state["visualizations"]
                    st.write(f"**{len(plots)} plots generated**")
                    cols = st.columns(3)
                    for j, p in enumerate(plots):
                        if os.path.exists(p): cols[j % 3].image(p, use_container_width=True)
                elif key in ("insights", "recommendations"):
                    st.markdown(state[key])
                elif key == "report":
                    st.caption("Full report compiled and saved.")

        # ── Token Usage ───────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🧠 LLM Token Usage")
        token_usage = state.get("token_usage", {})
        if token_usage:
            token_df = pd.DataFrame([{"Agent": a, **t} for a, t in token_usage.items()])
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("📥 Prompt Tokens",     token_df["prompt_tokens"].sum())
            tc2.metric("📤 Completion Tokens", token_df["completion_tokens"].sum())
            tc3.metric("🔢 Total Tokens",      token_df["total_tokens"].sum())
            st.dataframe(token_df, use_container_width=True, hide_index=True)
            st.bar_chart(token_df.set_index("Agent")[["prompt_tokens", "completion_tokens"]])
        else:
            st.info("No token usage data available.")

        # ── Full Results Tabs ─────────────────────────────────────────────
        st.divider()
        st.markdown("### 📂 Full Results")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dataset Info", "📈 Statistics", "🖼️ Visualizations",
            "💡 Insights & Recommendations", "📄 Report"
        ])
        with tab1:
            info = state["dataset_info"]
            st.dataframe(pd.DataFrame({
                "Column": list(info["dtypes"].keys()),
                "Type":   list(info["dtypes"].values()),
                "Nulls":  [info["nulls"][c] for c in info["dtypes"]]
            }), use_container_width=True, hide_index=True)
        with tab2:
            stats = state["statistics"]
            st.dataframe(pd.DataFrame(stats["describe"]), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.write("**Skewness:**")
            c1.dataframe(pd.DataFrame(stats["skewness"].items(), columns=["Col","Skew"]),
                         use_container_width=True, hide_index=True)
            c2.write("**Kurtosis:**")
            c2.dataframe(pd.DataFrame(stats["kurtosis"].items(), columns=["Col","Kurt"]),
                         use_container_width=True, hide_index=True)
        with tab3:
            cols = st.columns(2)
            for i, p in enumerate(state["visualizations"]):
                if os.path.exists(p): cols[i % 2].image(p, use_container_width=True)
        with tab4:
            st.subheader("💡 Insights")
            st.markdown(state["insights"])
            st.divider()
            st.subheader("✅ Recommendations")
            st.markdown(state["recommendations"])
        with tab5:
            st.markdown(state["report"])
            report_path = os.path.join(output_dir, "eda_report.md")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    st.download_button("⬇️ Download Report", f.read(),
                                       file_name="eda_report.md", mime="text/markdown")
