import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class SQLAnswerAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    def run(self, state: dict) -> dict:
        nl_query    = state.get("nl_query", "")
        sql_query   = state.get("sql_query", "")
        result_data = state.get("sql_result", [])

        if isinstance(result_data, str):
            return {"sql_answer": result_data}

        if not result_data:
            return {"sql_answer": "The query returned no results. Try rephrasing your question."}
        result_preview = json.dumps(result_data[:10], indent=2)

        prompt = f"""You are a helpful data analyst assistant.

The user asked: "{nl_query}"
SQL executed: {sql_query}
Query result ({len(result_data)} rows):
{result_preview}

Explain the result in 2-3 simple, friendly sentences. Be concise and focus on what the data actually shows."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        token_usage = state.get("token_usage", {})
        usage = response.response_metadata.get("token_usage", {})
        token_usage["sql_answer"] = {
            "prompt_tokens":     usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens":      usage.get("total_tokens", 0),
        }
        return {"sql_answer": response.content.strip(), "token_usage": token_usage}
