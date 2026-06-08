import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class NLtoSQLAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    def run(self, state: dict) -> dict:
        columns = state["dataset_info"]["columns"]
        dtypes  = state["dataset_info"]["dtypes"]
        schema  = ", ".join(f"{c} ({t})" for c, t in dtypes.items())
        nl_query = state["nl_query"]

        prompt = f"""You are an expert SQL writer. The table name is `df`.
Schema: {schema}

Convert this natural language question to a valid DuckDB SQL query.
Return ONLY the SQL query, no explanation, no markdown.

Question: {nl_query}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        token_usage = state.get("token_usage", {})
        usage = response.response_metadata.get("token_usage", {})
        token_usage["nl_to_sql"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        return {
            "sql_query": response.content.strip(),
            "token_usage": token_usage,
        }
