import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class RecommendationAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    def run(self, state: dict) -> dict:
        prompt = f"""You are a data science advisor. Based on these insights from an EDA:

{state["insights"]}

Provide 5 concise actionable recommendations for feature engineering, modeling, or further analysis."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"recommendations": response.content}
