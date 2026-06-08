import duckdb
import pandas as pd

class SQLAgent:
    def run(self, state: dict) -> dict:
        df = pd.DataFrame(state["cleaned_data"]["data"])
        sql_query = state.get("sql_query", "").strip()

        if not sql_query:
            return {"sql_result": "No SQL query provided."}

        try:
            result = duckdb.query(sql_query).df()
            return {"sql_result": result.to_dict(orient="records")}
        except Exception as e:
            return {"sql_result": f"Query error: {str(e)}"}
