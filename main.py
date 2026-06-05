import os
from WORKFLOW.langgraph_workflow import app

def main():
    file_path = "insurance.csv"
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]  # "insurance"
    output_dir = os.path.join("GRAPH", dataset_name)                 # "GRAPH/insurance"
    os.makedirs(output_dir, exist_ok=True)

    result = app.invoke({"file_path": file_path, "output_dir": output_dir})
    print(f"EDA Complete! Report saved to {output_dir}/eda_report.md")

if __name__ == "__main__":
    main()
