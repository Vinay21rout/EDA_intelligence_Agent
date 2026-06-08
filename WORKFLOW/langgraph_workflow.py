from langgraph.graph import START, END, StateGraph
from typing import TypedDict

from AGENTS.data_understanding_agent import DataUnderstandingAgent
from AGENTS.cleaning_agent import CleaningAgent
from AGENTS.statistics_agent import StatisticsAgent
from AGENTS.visualization_agent import VisualizationAgent
from AGENTS.insight_agent import InsightAgent
from AGENTS.recommendation_agent import RecommendationAgent
from AGENTS.report_agent import ReportAgent


class EDA_State(TypedDict):
    file_path: str
    output_dir: str
    dataset_info: dict
    cleaned_data: dict
    statistics: dict
    visualizations: list
    insights: str
    recommendations: str
    report: str
    token_usage: dict


# Instantiate agents
data_understanding = DataUnderstandingAgent()
cleaning           = CleaningAgent()
statistics         = StatisticsAgent()
visualization      = VisualizationAgent()
insight            = InsightAgent()
recommendation     = RecommendationAgent()
report             = ReportAgent()

# Build graph
graph = StateGraph(EDA_State)

graph.add_node("data_understanding", lambda s: data_understanding.run(s))
graph.add_node("cleaning",           lambda s: cleaning.run(s))
graph.add_node("statistics",         lambda s: statistics.run(s))
graph.add_node("visualization",      lambda s: visualization.run(s))
graph.add_node("insight",            lambda s: insight.run(s))
graph.add_node("recommendation",     lambda s: recommendation.run(s))
graph.add_node("report",             lambda s: report.run(s))

graph.add_edge(START,                "data_understanding")
graph.add_edge("data_understanding", "cleaning")
graph.add_edge("cleaning",           "statistics")
graph.add_edge("statistics",         "visualization")
graph.add_edge("visualization",      "insight")
graph.add_edge("insight",            "recommendation")
graph.add_edge("recommendation",     "report")
graph.add_edge("report",             END)

app = graph.compile()
