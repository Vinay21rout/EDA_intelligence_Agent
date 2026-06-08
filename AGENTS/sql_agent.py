import duckdb
import pandas as pd

class SQLAgent:
    def run(self, state: dict) -> dict:
        df = pd.DataFrame(state["cleaned_data"]["data"])  # noqa: F841 — used by duckdb
        sql_query = state.get("sql_query", "").strip()

        if not sql_query:
            return {"sql_result": "No SQL query provided."}

        try:
            con = duckdb.connect()
            con.register("df", df)
            result = con.execute(sql_query).df()
            con.close()
            return {"sql_result": result.to_dict(orient="records")}
        except Exception as e:
            return {"sql_result": f"Query error: {str(e)}"}
