import pandas as pd

def file_df(file_path: str):
    return pd.read_csv(file_path)
