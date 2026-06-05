import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class InsightAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    def run(self, state: dict) -> dict:
        stats_summary = json.dumps(state["statistics"]["describe"], indent=2)[:3000]

        prompt = f"""You are a data analyst. Based on the following dataset statistics, provide key insights:

{stats_summary}

Give 5 concise bullet point insights about patterns, distributions, and notable observations."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"insights": response.content}
