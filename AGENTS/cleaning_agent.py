from FILE_input.file_input import file_df

class CleaningAgent:
    def run(self, state: dict) -> dict:
        df = file_df(state["file_path"])
        original_rows = state["dataset_info"]["shape"][0]

        df = df.drop_duplicates()

        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        cleaned_data = {
            "shape_after_cleaning": df.shape,
            "remaining_nulls": df.isnull().sum().to_dict(),
            "duplicates_removed": original_rows - df.shape[0],
            "data": df.to_dict(),
        }
        return {"cleaned_data": cleaned_data}
