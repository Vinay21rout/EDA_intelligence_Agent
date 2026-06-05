<div align="center">

# 🔍 EDA Agent

### Automated Exploratory Data Analysis powered by LangGraph & Groq LLM

[![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3.3_70B-orange?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

<br/>

> Upload any CSV → Get a full EDA report with plots, insights, and recommendations — automatically.

</div>

---

## ✨ Features

- 🤖 **Multi-Agent Pipeline** — 7 specialized agents, each with a single responsibility
- 🧠 **LLM-Powered Insights** — Uses Groq's `llama-3.3-70b-versatile` for deep analysis
- 📊 **Auto Visualizations** — Histograms, correlation heatmap, countplots saved automatically
- 🗂️ **Dataset Isolation** — Each dataset gets its own folder under `GRAPH/<dataset_name>/`
- 🖥️ **Streamlit UI** — Clean interactive interface with tabbed results and report download
- 📄 **Markdown Report** — Full EDA report compiled and downloadable

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Pipeline                        │
│                                                                   │
│  ┌──────────┐   ┌─────────┐   ┌────────────┐   ┌─────────────┐ │
│  │   Data   │──▶│Cleaning │──▶│ Statistics │──▶│Visualization│ │
│  │  Under-  │   │  Agent  │   │   Agent    │   │    Agent    │ │
│  │ standing │   └─────────┘   └────────────┘   └──────┬──────┘ │
│  └──────────┘                                          │        │
│                                                        ▼        │
│  ┌──────────┐   ┌────────────┐   ┌──────────────────────────┐  │
│  │  Report  │◀──│Recommend-  │◀──│      Insight Agent       │  │
│  │  Agent   │   │ation Agent │   │   (Groq LLaMA 3.3 70B)   │  │
│  └──────────┘   └────────────┘   └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agents

| Agent | Responsibility |
|---|---|
| `DataUnderstandingAgent` | Shape, dtypes, null counts, head preview |
| `CleaningAgent` | Drop duplicates, fill nulls (median/mode) |
| `StatisticsAgent` | Describe, correlation, skewness, kurtosis |
| `VisualizationAgent` | Histograms, heatmap, countplots |
| `InsightAgent` | LLM-generated insights via Groq |
| `RecommendationAgent` | LLM-generated actionable recommendations |
| `ReportAgent` | Compiles full markdown EDA report |

---

## 📁 Project Structure

```
EDA_AGENT/
├── AGENTS/
│   ├── data_understanding_agent.py
│   ├── cleaning_agent.py
│   ├── statistics_agent.py
│   ├── visualization_agent.py
│   ├── insight_agent.py
│   ├── recommendation_agent.py
│   └── report_agent.py
├── WORKFLOW/
│   └── langgraph_workflow.py
├── FILE_input/
│   └── file_input.py
├── GRAPH/
│   └── <dataset_name>/
│       ├── plots/
│       └── eda_report.md
├── app.py           ← Streamlit UI
├── main.py          ← CLI entry point
├── .env
└── requirements.txt
```

---

## ⚙️ Setup

**1. Clone the repo**
```bash
git clone https://github.com/<your_username>/eda-agent.git
cd eda-agent
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key to `.env`**
```env
GROQ_API_KEY=your_key_here
```
> Get your free API key at [console.groq.com](https://console.groq.com)

---

## 🚀 Run

### Streamlit UI _(recommended)_
```bash
streamlit run app.py
```

### CLI
```bash
python main.py
```

---

## 📤 Output

All outputs are saved to `GRAPH/<dataset_name>/`:

```
GRAPH/
└── insurance/
    ├── plots/
    │   ├── age_hist.png
    │   ├── bmi_hist.png
    │   ├── correlation_heatmap.png
    │   └── sex_countplot.png
    └── eda_report.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent orchestration |
| [Groq](https://groq.com/) | LLM inference (LLaMA 3.3 70B) |
| [Streamlit](https://streamlit.io/) | Web UI |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) | Visualizations |

---

<div align="center">
Made with ❤️ using LangGraph & Groq
</div>
