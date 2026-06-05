import os
import streamlit as st
from WORKFLOW.langgraph_workflow import app

st.set_page_config(page_title="EDA Agent", page_icon="🔍", layout="wide")
st.title("🔍 EDA Agent")
st.caption("Automated Exploratory Data Analysis powered by LangGraph & Groq")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = os.path.join("FILE_input", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    dataset_name = os.path.splitext(uploaded_file.name)[0]
    output_dir = os.path.join("GRAPH", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    st.success(f"File uploaded: `{uploaded_file.name}`")

    if st.button("▶ Run EDA"):
        with st.spinner("Running EDA pipeline..."):
            # Progress tracking
            progress = st.progress(0)
            status = st.empty()

            stages = [
                "data_understanding", "cleaning", "statistics",
                "visualization", "insight", "recommendation", "report"
            ]
            result = {}

            for i, stage in enumerate(stages):
                status.write(f"⚙️ Running: `{stage}`...")
                progress.progress(int((i + 1) / len(stages) * 100))

            result = app.invoke({"file_path": temp_path, "output_dir": output_dir})
            progress.progress(100)
            status.success("✅ EDA Complete!")

        # --- Tabs for output ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dataset Info", "📈 Statistics", "🖼️ Visualizations", "💡 Insights & Recommendations", "📄 Report"
        ])

        with tab1:
            info = result["dataset_info"]
            col1, col2 = st.columns(2)
            col1.metric("Rows", info["shape"][0])
            col2.metric("Columns", info["shape"][1])
            st.write("**Column Types:**")
            st.json(info["dtypes"])
            st.write("**Missing Values:**")
            st.json(info["nulls"])

        with tab2:
            stats = result["statistics"]
            st.write("**Descriptive Statistics:**")
            st.json(stats["describe"])
            st.write("**Skewness:**")
            st.json(stats["skewness"])
            st.write("**Kurtosis:**")
            st.json(stats["kurtosis"])

        with tab3:
            plots = result["visualizations"]
            if plots:
                cols = st.columns(2)
                for i, plot_path in enumerate(plots):
                    if os.path.exists(plot_path):
                        cols[i % 2].image(plot_path, use_container_width=True)
            else:
                st.info("No visualizations generated.")

        with tab4:
            st.subheader("💡 Insights")
            st.markdown(result["insights"])
            st.divider()
            st.subheader("✅ Recommendations")
            st.markdown(result["recommendations"])

        with tab5:
            st.markdown(result["report"])
            report_path = os.path.join(output_dir, "eda_report.md")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    st.download_button(
                        label="⬇️ Download Report",
                        data=f.read(),
                        file_name="eda_report.md",
                        mime="text/markdown"
                    )
