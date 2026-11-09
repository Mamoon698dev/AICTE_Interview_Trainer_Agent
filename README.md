# 🎯 IBM Interview Trainer Agent

> **Powered by IBM Granite (ibm/granite-13b-instruct-v2) + Watsonx.ai + RAG**

An AI-driven interview preparation assistant that generates tailored question sets, preparation strategies, and real-time answer feedback using Retrieval-Augmented Generation (RAG) over a curated interview knowledge base.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Personalised Question Sets** | Technical, behavioural (STAR), and HR questions tailored to role & experience level |
| **Resume Parsing** | Extracts skills, experience, and job titles from PDF / DOCX / TXT resumes via IBM Granite |
| **RAG Pipeline** | ChromaDB vector store + IBM Slate embeddings retrieves role-specific context before generation |
| **Answer Evaluation** | Scores candidate answers 0–10, identifies strengths/weaknesses, and suggests improvements |
| **Skill Gap Analysis** | Compares candidate skills against role requirements and recommends learning resources |
| **Preparation Strategy** | 7-day personalised prep plan with learning resources and confidence tips |
| **Mock Interview Mode** | Practice answering questions with real-time IBM Granite feedback |
| **Chat Interface** | Free-form Q&A with a RAG-backed interview coach |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Interview Trainer Agent               │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │Resume Parser │  │  Question    │  │   Skill      │  │
│  │(IBM Granite) │  │  Generator   │  │  Assessor    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│         └─────────────────┴──────────────────┘          │
│                           │                             │
│                  ┌────────▼────────┐                    │
│                  │  RAG Pipeline   │                    │
│                  │  (LangChain +   │                    │
│                  │   ChromaDB)     │                    │
│                  └────────┬────────┘                    │
│                           │                             │
│              ┌────────────▼────────────┐                │
│              │    IBM Watsonx.ai       │                │
│              │  • granite-13b-instruct │                │
│              │  • slate-30m-rtrvr     │                │
│              └─────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Interview_Trainer_Agent/
├── src/
│   ├── __init__.py
│   ├── config.py           # Environment configuration
│   ├── watsonx_client.py   # IBM Granite LLM + Slate embeddings
│   ├── rag_pipeline.py     # ChromaDB vector store & retrieval
│   ├── knowledge_base.py   # Built-in interview Q&A knowledge
│   ├── resume_parser.py    # Resume file parsing (PDF/DOCX/TXT)
│   ├── question_generator.py # Interview question generation
│   ├── skill_assessor.py   # Answer evaluation & skill gap analysis
│   └── agent.py            # Top-level orchestrator
├── app.py                  # Streamlit web UI
├── cli.py                  # Rich terminal CLI
├── requirements.txt        # Python dependencies
├── env.example             # Environment variable template
├── setup.py                # One-click setup script
└── data/
    ├── chroma_db/          # Vector store (auto-created)
    ├── resumes/            # Uploaded resumes
    └── knowledge_base/     # Interview knowledge documents
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- An **IBM Cloud** account (free Lite tier works)
- A **Watsonx.ai** project

### 2. Get IBM Cloud Credentials

1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (free Lite account)
2. Create a **Watsonx.ai** project at [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
3. Generate an **API key** at **Manage → Access (IAM) → API keys**
4. Copy your **Project ID** from the project settings

### 3. Configure Environment

```bash
# Copy the template
cp env.example .env   # Linux/Mac
copy env.example .env # Windows

# Edit .env and fill in:
# WATSONX_API_KEY=<your_api_key>
# WATSONX_PROJECT_ID=<your_project_id>
```

### 4. Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup (creates directories, validates config)
python setup.py

# Launch Streamlit web UI
streamlit run app.py

# OR launch terminal CLI
python cli.py
```

---

## 🖥️ Usage

### Streamlit Web UI (`app.py`)

1. Fill in your **name**, **target role**, and **experience level** in the sidebar
2. Optionally upload your **resume** (PDF / DOCX / TXT)
3. Click **Start Session**
4. Navigate between tabs:
   - **📋 Interview Plan** – generate your tailored question set
   - **🎤 Mock Interview** – practice answering questions
   - **📊 Skill Gap** – identify skills to develop
   - **💬 Chat** – ask your AI coach anything
   - **📈 Summary** – view session statistics

### Terminal CLI (`cli.py`)

```bash
python cli.py
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `WATSONX_API_KEY` | *(required)* | IBM Cloud IAM API key |
| `WATSONX_PROJECT_ID` | *(required)* | Watsonx.ai project ID |
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` | Regional endpoint |
| `GRANITE_MODEL_ID` | `ibm/granite-13b-instruct-v2` | Primary LLM |
| `GRANITE_EMBEDDING_MODEL` | `ibm/slate-30m-english-rtrvr` | Embedding model |
| `RAG_TOP_K` | `5` | Retrieved chunks per query |
| `CHUNK_SIZE` | `512` | Document chunk size (tokens) |
| `MAX_NEW_TOKENS` | `1024` | Max LLM output tokens |
| `TEMPERATURE` | `0.7` | LLM creativity (0–1) |

---

## 🔧 Adding Custom Knowledge

Place PDF, DOCX, or TXT files in `data/knowledge_base/`. They will be ingested automatically on the next run.

---

## 📄 License

MIT License – see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

Built with [IBM Watsonx.ai](https://www.ibm.com/watsonx), [IBM Granite](https://www.ibm.com/granite), [LangChain](https://www.langchain.com/), and [ChromaDB](https://www.trychroma.com/).
