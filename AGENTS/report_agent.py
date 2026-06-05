import os

BASE_GRAPH_DIR = "GRAPH"

class ReportAgent:
    def run(self, state: dict) -> dict:
        dataset_info = state["dataset_info"]
        cleaned = state["cleaned_data"]
        plots = "\n".join(f"- {p}" for p in state["visualizations"])

        report = f"""# EDA Report

## Dataset Overview
- Shape: {dataset_info['shape']}
- Columns: {dataset_info['columns']}
- Missing Values: {dataset_info['nulls']}

## After Cleaning
- Shape: {cleaned['shape_after_cleaning']}
- Duplicates Removed: {cleaned['duplicates_removed']}
- Remaining Nulls: {cleaned['remaining_nulls']}

## Visualizations
{plots}

## Insights
{state['insights']}

## Recommendations
{state['recommendations']}
"""
        safe_name = os.path.basename(os.path.normpath(state["output_dir"]))
        report_path = os.path.join(BASE_GRAPH_DIR, safe_name, "eda_report.md")
        with open(report_path, "w") as f:
            f.write(report)

        return {"report": report}
