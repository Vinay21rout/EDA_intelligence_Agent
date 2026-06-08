from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Optional

from AGENTS.data_understanding_agent import DataUnderstandingAgent
from AGENTS.cleaning_agent import CleaningAgent
from AGENTS.statistics_agent import StatisticsAgent
from AGENTS.visualization_agent import VisualizationAgent
from AGENTS.insight_agent import InsightAgent
from AGENTS.recommendation_agent import RecommendationAgent
from AGENTS.report_agent import ReportAgent
from AGENTS.nl_to_sql_agent import NLtoSQLAgent
from AGENTS.sql_agent import SQLAgent
from AGENTS.sql_answer_agent import SQLAnswerAgent


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
    nl_query: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[object]
    sql_answer: Optional[str]
    token_usage: dict


# Instantiate agents
data_understanding  = DataUnderstandingAgent()
cleaning            = CleaningAgent()
statistics          = StatisticsAgent()
visualization       = VisualizationAgent()
insight             = InsightAgent()
recommendation      = RecommendationAgent()
report              = ReportAgent()
nl_to_sql   = NLtoSQLAgent()
sql         = SQLAgent()
sql_answer  = SQLAnswerAgent()

# Build graph
graph = StateGraph(EDA_State)

graph.add_node("data_understanding", lambda s: data_understanding.run(s))
graph.add_node("cleaning",           lambda s: cleaning.run(s))
graph.add_node("statistics",         lambda s: statistics.run(s))
graph.add_node("visualization",      lambda s: visualization.run(s))
graph.add_node("insight",            lambda s: insight.run(s))
graph.add_node("recommendation",     lambda s: recommendation.run(s))
graph.add_node("report",             lambda s: report.run(s))
graph.add_node("nl_to_sql",          lambda s: nl_to_sql.run(s))
graph.add_node("sql",                lambda s: sql.run(s))

# Main EDA pipeline
graph.add_edge(START,              "data_understanding")
graph.add_edge("data_understanding", "cleaning")
graph.add_edge("cleaning",           "statistics")
graph.add_edge("statistics",         "visualization")
graph.add_edge("visualization",      "insight")
graph.add_edge("insight",            "recommendation")
graph.add_edge("recommendation",     "report")
graph.add_edge("report",             END)

# SQL sub-graph (nl_query → sql_query → sql_result)
graph.add_edge("nl_to_sql", "sql")
graph.add_edge("sql", END)

app = graph.compile()

# Separate entry for SQL-only queries (after EDA state is available)
sql_graph = StateGraph(EDA_State)
sql_graph.add_node("nl_to_sql",  lambda s: nl_to_sql.run(s))
sql_graph.add_node("sql",        lambda s: sql.run(s))
sql_graph.add_node("sql_answer", lambda s: sql_answer.run(s))
sql_graph.add_edge(START,        "nl_to_sql")
sql_graph.add_edge("nl_to_sql",  "sql")
sql_graph.add_edge("sql",        "sql_answer")
sql_graph.add_edge("sql_answer", END)
sql_app = sql_graph.compile()
