from FILE_input.file_input import file_df

class DataUnderstandingAgent:
    def run(self, state: dict) -> dict:
        df = file_df(state["file_path"])
        dataset_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "nulls": df.isnull().sum().to_dict(),
            "head": df.head(5).to_dict(),
        }
        return {"dataset_info": dataset_info}
