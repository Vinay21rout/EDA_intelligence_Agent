import pandas as pd

class StatisticsAgent:
    def run(self, state: dict) -> dict:
        df = pd.DataFrame(state["cleaned_data"]["data"])

        statistics = {
            "describe": df.describe(include="all").to_dict(),
            "correlation": df.select_dtypes(include="number").corr().to_dict(),
            "skewness": df.select_dtypes(include="number").skew().to_dict(),
            "kurtosis": df.select_dtypes(include="number").kurt().to_dict(),
            "value_counts": {
                col: df[col].value_counts().to_dict()
                for col in df.select_dtypes(include="object").columns
            },
        }
        return {"statistics": statistics}
