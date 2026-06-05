import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_GRAPH_DIR = "GRAPH"

class VisualizationAgent:
    def run(self, state: dict) -> dict:
        df = pd.DataFrame(state["cleaned_data"]["data"])
        safe_name = os.path.basename(os.path.normpath(state["output_dir"]))
        plots_dir = os.path.join(BASE_GRAPH_DIR, safe_name, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        saved = []

        for col in df.select_dtypes(include="number").columns:
            path = f"{plots_dir}/{col}_hist.png"
            df[col].hist()
            plt.title(f"{col} Distribution")
            plt.savefig(path)
            plt.close()
            saved.append(path)

        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            path = f"{plots_dir}/correlation_heatmap.png"
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.savefig(path)
            plt.close()
            saved.append(path)

        for col in df.select_dtypes(include="object").columns:
            path = f"{plots_dir}/{col}_countplot.png"
            df[col].value_counts().plot(kind="bar")
            plt.title(f"{col} Count")
            plt.savefig(path)
            plt.close()
            saved.append(path)

        return {"visualizations": saved}
