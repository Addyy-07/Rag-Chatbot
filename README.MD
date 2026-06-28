# 📄 DocChat AI — Production-Grade RAG Chatbot

An AI-powered chatbot that lets you upload any PDF and have a contextual conversation with it.
Built using **Retrieval-Augmented Generation (RAG)** — combining Pinecone vector search with
Groq's Llama 3 LLM to deliver grounded, accurate answers.

🔗 **Live Demo:** [addyy-07-rag-chatbot-app-owr2pm.streamlit.app](https://addyy-07-rag-chatbot-app-owr2pm.streamlit.app)

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📤 PDF Upload | Drag-and-drop any PDF from the UI |
| 🔍 Semantic Search | Vector embeddings via HuggingFace `all-MiniLM-L6-v2` |
| 🤖 LLM Answers | Groq Llama 3.1 8B — grounded in your document |
| 💬 Conversation Memory | Rolling chat history across multiple questions |
| ⚙️ Config-driven | All tunables (chunk size, top-k, model) via env vars |
| 🐳 Docker-ready | Multi-stage production image with health check |
| 🧪 Tested | pytest unit tests with full mocking |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| LangChain | RAG pipeline orchestration |
| HuggingFace `all-MiniLM-L6-v2` | 384-dim text embeddings |
| Groq Llama 3.1 8B | Fast, free LLM inference |
| Pinecone | Serverless vector database |
| PyPDF | PDF parsing |
| Pydantic Settings | Type-safe configuration management |
| Streamlit | Web UI |
| pytest | Unit testing |
| Docker | Containerised deployment |

---

## 🧠 How It Works

```
PDF Upload → PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFace Embeddings
                                                                        ↓
                                                             Pinecone (stored)
                                                                        ↓
User Question → Embed query → Similarity Search (top-k=3) → Context chunks
                                                                        ↓
                                              Groq Llama 3 + RAG Prompt → Answer
```

---

## 📁 Project Structure

```
rag-chatbot/
│
├── backend/
│   ├── config/
│   │   └── settings.py          # Pydantic BaseSettings — all env vars
│   ├── services/
│   │   ├── embedding_service.py # HuggingFace model factory
│   │   ├── vector_store_service.py # Pinecone index + retriever
│   │   ├── ingestion_service.py # PDF → chunks → Pinecone pipeline
│   │   └── chat_service.py      # RAG Q&A pipeline
│   ├── routes/
│   │   ├── ingest_router.py     # Upload orchestration
│   │   └── chat_router.py       # Query orchestration
│   ├── prompts/
│   │   └── rag_prompt.py        # LangChain prompt templates
│   ├── utils/
│   │   ├── logger.py            # Structured logging
│   │   └── file_utils.py        # Temp file context manager
│   └── main.py                  # Streamlit UI entrypoint
│
├── frontend/
│   └── styles.py                # All CSS / HTML strings
│
├── tests/
│   ├── test_ingestion_service.py
│   ├── test_chat_service.py
│   └── test_vector_store_service.py
│
├── docs/
│   └── architecture.md
│
├── .env.example                 # Safe env template (commit this)
├── .env                         # Real secrets (never commit)
├── Dockerfile                   # Multi-stage production image
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Addyy-07/Rag-Chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your real API keys
```

### 5. Run the app
```bash
streamlit run backend/main.py
```

---

## 🐳 Docker

```bash
# Build
docker build -t docchat-ai .

# Run (pass your .env file)
docker run --env-file .env -p 8501:8501 docchat-ai
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔑 Environment Variables

All variables are documented in [`.env.example`](.env.example).

| Variable | Required | Default | Description |
|---|---|---|---|
| `PINECONE_API_KEY` | ✅ | — | Pinecone API key |
| `GROQ_API_KEY` | ✅ | — | Groq API key |
| `PINECONE_INDEX_NAME` | | `rag-chatbot-free` | Pinecone index name |
| `GROQ_MODEL` | | `llama-3.1-8b-instant` | Groq model ID |
| `CHUNK_SIZE` | | `1000` | PDF chunk size (chars) |
| `CHUNK_OVERLAP` | | `200` | Chunk overlap (chars) |
| `RETRIEVER_TOP_K` | | `3` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | | `all-MiniLM-L6-v2` | HuggingFace model |

---

## 🏗️ Architecture Principles

- **Single Responsibility** — Each module has exactly one reason to change
- **Open/Closed** — Services accept injected dependencies; swap providers without touching callers
- **Dependency Inversion** — `ChatService` depends on abstractions, not concrete implementations
- **Config-driven** — Zero hardcoded values; everything tuneable via env vars
- **Layered** — Config → Utils → Services → Routes → UI (dependencies only flow inward)

See [`docs/architecture.md`](docs/architecture.md) for full Architecture Decision Records.

---

## 👨‍💻 Author

**Adnan** — Full Stack Developer & CSE Student at NIT Allahabad

[![GitHub](https://img.shields.io/badge/GitHub-Addyy--07-black?logo=github)](https://github.com/Addyy-07)

---

## 📄 License

MIT License — feel free to use and modify this project.