# 📄 RAG Chatbot — Chat with your PDF

An AI-powered chatbot that lets you upload any PDF and ask questions about it.
Built using Retrieval-Augmented Generation (RAG) — a technique that combines
vector search with large language models to answer questions grounded in your documents.

🔗 **Live Demo:** [addyy-07-rag-chatbot-app-owr2pm.streamlit.app](https://addyy-07-rag-chatbot-app-owr2pm.streamlit.app)

---

## ✨ Features

- 📤 Upload any PDF directly from the UI
- 🔍 Semantic search using vector embeddings
- 🤖 GPT-4o-mini powered answers grounded in your document
- 💬 Conversation memory across multiple questions
- ☁️ Deployed live on Streamlit Cloud

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| LangChain | RAG pipeline orchestration |
| OpenAI `text-embedding-ada-002` | Text embeddings |
| OpenAI `gpt-4o-mini` | Answer generation |
| Pinecone | Vector database for storing & retrieving chunks |
| PyPDF | PDF parsing |
| Streamlit | Web UI and deployment |

---

## 🧠 How It Works
```
PDF Upload → Chunking → Embeddings → Pinecone (stored)
                                          ↓
User Question → Embed query → Similarity Search → Top chunks → GPT-4o-mini → Answer
```

1. **Ingestion** — The PDF is split into overlapping chunks of ~1000 characters
2. **Embedding** — Each chunk is converted into a 1536-dimension vector using OpenAI embeddings
3. **Storage** — Vectors are stored in Pinecone's vector database
4. **Retrieval** — When a user asks a question, the query is embedded and the top 4 most similar chunks are fetched
5. **Generation** — GPT-4o-mini receives the question + retrieved chunks and generates a grounded answer

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Addyy-07/Rag-Chatbot.git
cd Rag-Chatbot
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root folder:
```
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=rag-chatbot
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
rag-chatbot/
├── app.py              # Main Streamlit app
├── ingest.py           # Script to ingest PDFs via terminal
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this)
└── .gitignore
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | Name of your Pinecone index |

---

## 👨‍💻 Author

**Adnan** — Full Stack Developer & CSE Student at NIT Allahabad

[![GitHub](https://img.shields.io/badge/GitHub-Addyy--07-black?logo=github)](https://github.com/Addyy-07)

---

## 📄 License

MIT License — feel free to use and modify this project.