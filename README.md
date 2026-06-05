# EDA Agent

An automated Exploratory Data Analysis (EDA) pipeline built with LangGraph and Groq LLM.

## Architecture

Each stage is a separate agent class wired sequentially via LangGraph:

```
DataUnderstanding → Cleaning → Statistics → Visualization → Insight → Recommendation → Report
```

## Setup

1. Clone the repo
2. Create and activate virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your Groq API key to `.env`:
   ```
   GROQ_API_KEY=your_key_here
   ```

## Run

```bash
python main.py
```

Output is saved to `GRAPH/<dataset_name>/` with plots and `eda_report.md`.

## Agents

| Agent | Responsibility |
|---|---|
| DataUnderstandingAgent | Shape, dtypes, nulls, head |
| CleaningAgent | Drop duplicates, fill nulls |
| StatisticsAgent | Describe, correlation, skewness |
| VisualizationAgent | Histograms, heatmap, countplots |
| InsightAgent | LLM-generated insights (Groq) |
| RecommendationAgent | LLM-generated recommendations (Groq) |
| ReportAgent | Compiles markdown report |
