import os
import time
import pandas as pd
import streamlit as st
from WORKFLOW.langgraph_workflow import app, sql_app
from AGENTS.data_understanding_agent import DataUnderstandingAgent
from AGENTS.cleaning_agent import CleaningAgent
from AGENTS.statistics_agent import StatisticsAgent
from AGENTS.visualization_agent import VisualizationAgent
from AGENTS.insight_agent import InsightAgent
from AGENTS.recommendation_agent import RecommendationAgent
from AGENTS.report_agent import ReportAgent

st.set_page_config(page_title="EDA Agent", page_icon="🔍", layout="wide")

# ── Sidebar width CSS (wide mode) ─────────────────────────────────────────────
if st.session_state.get("chat_wide"):
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { min-width: 520px !important; max-width: 520px !important; }
    </style>""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state["chat_history"]  = []
if "chat_wide"     not in st.session_state: st.session_state["chat_wide"]     = False
if "eda_state"     not in st.session_state: st.session_state["eda_state"]     = None
if "last_result"   not in st.session_state: st.session_state["last_result"]   = None

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

    # Auto-run DataUnderstanding + Cleaning on upload so chat is always available
    if st.session_state["eda_state"] is None or \
       st.session_state["eda_state"].get("file_path") != temp_path:
        with st.spinner("Preparing dataset for chat..."):
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

# ── Sidebar Chatbot ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 4px'>
        <span style='font-size:1.5em'>🗃️</span>
        <span style='font-weight:700;font-size:1.05em'> Ask Your Data</span><br/>
        <span style='font-size:0.75em;color:#94a3b8'>Natural Language → SQL · Groq</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Toggle wide mode by controlling sidebar width via session state label trick
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔲 Wide" if not st.session_state["chat_wide"] else "🔳 Normal", use_container_width=True):
            st.session_state["chat_wide"] = not st.session_state["chat_wide"]
            st.rerun()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    # Chat history
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state["chat_history"]:
            st.markdown("<p style='color:#475569;font-size:0.82em;text-align:center;margin-top:40px'>Upload a CSV to start asking<br/>questions about your data.</p>", unsafe_allow_html=True)
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='background:#7c3aed22;border:1px solid #7c3aed55;border-radius:12px 12px 2px 12px;
                            padding:8px 12px;font-size:0.85em;margin:4px 0;color:#e2e8f0;text-align:right'>
                🧑 {msg['content']}</div>""", unsafe_allow_html=True)
            else:
                sql_block = f"<div style='background:#0f172a;border:1px solid #1e40af;border-radius:6px;padding:5px 8px;font-family:monospace;font-size:0.75em;color:#93c5fd;margin-top:4px;word-break:break-all'>⚡ {msg['sql']}</div>" if msg.get("sql") else ""
                tok_block = f"<div style='font-size:0.72em;color:#475569;margin-top:3px'>🔢 {msg['tokens']} tokens</div>" if msg.get("tokens") else ""
                st.markdown(f"""
                <div style='background:#1e293b;border:1px solid #334155;border-radius:12px 12px 12px 2px;
                            padding:8px 12px;font-size:0.85em;margin:4px 0;color:#cbd5e1'>
                🤖 {msg['content']}{sql_block}{tok_block}</div>""", unsafe_allow_html=True)

    st.divider()

    # Input form — always available after upload
    if st.session_state["eda_state"]:
        with st.form("chat_form", clear_on_submit=True):
            nl_input  = st.text_input("", placeholder="e.g. top 5 rows where age > 40",
                                      label_visibility="collapsed")
            submitted = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

        if submitted and nl_input.strip():
            st.session_state["chat_history"].append({"role": "user", "content": nl_input})
            with st.spinner("Thinking..."):
                invoke_state = {
                    **st.session_state["eda_state"],
                    "nl_query":   nl_input,
                    "sql_query":  None,
                    "sql_result": None,
                    "sql_answer": None,
                    "token_usage": st.session_state["eda_state"].get("token_usage", {}),
                }
                sql_result = sql_app.invoke(invoke_state)

            sql_query   = sql_result.get("sql_query", "")
            result_data = sql_result.get("sql_result", [])
            sql_answer  = sql_result.get("sql_answer", "")
            usage       = sql_result.get("token_usage", {}).get("nl_to_sql", {})
            total_tok   = sum(v.get("total_tokens", 0) for v in sql_result.get("token_usage", {}).values())

            if isinstance(result_data, list) and result_data:
                st.session_state["chat_history"].append({
                    "role": "bot",
                    "content": sql_answer or f"Found {len(result_data)} row(s).",
                    "sql": sql_query, "tokens": total_tok
                })
                st.session_state["last_result"] = result_data
            else:
                st.session_state["chat_history"].append({
                    "role": "bot",
                    "content": sql_answer or (str(result_data) if result_data else "No results found."),
                    "sql": sql_query, "tokens": total_tok
                })
                st.session_state["last_result"] = None
            st.rerun()
    else:
        st.info("⚠️ Upload a CSV file to start chatting.")

# Show last query result in main area if wide mode
if st.session_state.get("last_result") and st.session_state["chat_wide"]:
    st.divider()
    st.markdown("### 🗃️ Query Result")
    st.dataframe(pd.DataFrame(st.session_state["last_result"]), use_container_width=True)
