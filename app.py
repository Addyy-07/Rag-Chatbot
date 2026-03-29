import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

st.set_page_config(
    page_title="DocChat AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}

    .stApp { background-color: #0f1117; }

    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
        text-align: center;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #6b7280;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 28px;
    }
    .stat-card {
        background-color: #1a1d27;
        border: 1px solid #2e3047;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 2px;
    }
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
        font-size: 15px;
    }
    .stButton > button:hover {
        background-color: #5a6fd6;
    }
    [data-testid="stChatMessage"] {
        background-color: #1a1d27;
        border-radius: 12px;
        border: 1px solid #2e3047;
        margin-bottom: 8px;
    }
    .ready-badge {
        display: inline-block;
        background-color: #1a3a2a;
        color: #34d399;
        border: 1px solid #34d399;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .powered-by {
        text-align: center;
        color: #374151;
        font-size: 0.75rem;
        margin-top: 24px;
        padding-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper functions ----

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def ensure_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return index_name

def ingest_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    index_name = ensure_index()
    embeddings = get_embeddings()
    PineconeVectorStore.from_documents(
        chunks, embeddings,
        index_name=index_name
    )
    os.unlink(tmp_path)
    return len(chunks), len(documents)

def get_answer(question, chat_history):
    index_name = ensure_index()
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # reduced from 4 to 3
    docs = retriever.invoke(question)

    # Truncate each chunk to 500 chars max
    context = "\n\n".join(doc.page_content[:500] for doc in docs)

    # Only keep last 2 messages for history
    history_text = ""
    for msg in chat_history[-2:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content'][:200]}\n"

    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer based on the context below.
If the answer isn't in the context, say so honestly. Be concise.

Previous conversation:
{history}

Context:
{context}

Question: {question}

Answer:""")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=512
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "history": history_text,
        "context": context,
        "question": question
    })

# ---- Main UI ----
st.markdown('<p class="main-title">🧠 DocChat AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a PDF and have a conversation with it</p>',
            unsafe_allow_html=True)

# ---- Upload Section ----
if not st.session_state.get("pdf_ready"):

    uploaded_file = st.file_uploader(
        "📂 Choose a PDF file",
        type="pdf",
        help="Upload any PDF to start chatting"
    )

    if uploaded_file:
        st.markdown(
            f"**📄 {uploaded_file.name}** &nbsp;·&nbsp; "
            f"{uploaded_file.size / 1024:.1f} KB"
        )
        if st.button("⚡ Process & Start Chatting"):
            with st.spinner("Reading and indexing your PDF..."):
                chunks, pages = ingest_pdf(uploaded_file)
            st.session_state.pdf_ready = True
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_chunks = chunks
            st.session_state.pdf_pages = pages
            st.session_state.messages = []
            st.rerun()

# ---- Chat Section ----
else:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            f'<span class="ready-badge">✅ {st.session_state.pdf_name}</span>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.pdf_pages}</div>
            <div class="stat-label">Pages</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.pdf_chunks}</div>
            <div class="stat-label">Chunks</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 Upload New PDF"):
        st.session_state.pdf_ready = False
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:40px 20px;">
            <div style="font-size:2.5rem">💬</div>
            <div style="margin-top:10px; color:#6b7280;">
                Ask anything about your document below
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(prompt, st.session_state.messages)
            st.write(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

# ---- Footer ----
st.markdown("""
<div class="powered-by">
    🤗 HuggingFace &nbsp;|&nbsp; 📦 Pinecone &nbsp;|&nbsp; ⚡ Groq Llama 3
</div>
""", unsafe_allow_html=True)