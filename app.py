import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="DocChat AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3047;
    }

    [data-testid="stChatMessage"] {
        background-color: #1a1d27;
        border-radius: 12px;
        border: 1px solid #2e3047;
        margin-bottom: 8px;
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

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #667eea;
        margin-bottom: 0;
    }
    .sub-title {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    .sidebar-header {
        color: #667eea;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .stat-card {
        background-color: #12151e;
        border: 1px solid #2e3047;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 8px;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .empty-state {
        text-align: center;
        padding: 80px 20px;
        color: #6b7280;
    }
    .empty-state .icon {
        font-size: 5rem;
        margin-bottom: 16px;
    }
    .empty-state .title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #9ca3af;
        margin-bottom: 8px;
    }
    .empty-state .desc {
        font-size: 0.95rem;
        color: #4b5563;
    }
    .ready-badge {
        display: inline-block;
        background-color: #1a3a2a;
        color: #34d399;
        border: 1px solid #34d399;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper functions ----

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")

def ingest_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    PineconeVectorStore.from_documents(
        chunks, embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    os.unlink(tmp_path)
    return len(chunks), len(documents)

def get_answer(question, chat_history):
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    history_text = ""
    for msg in chat_history[-4:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided document context.
Be concise, clear, and friendly. If the answer isn't in the context, say so honestly.

Previous conversation:
{history}

Context from document:
{context}

Question: {question}

Answer:""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"history": history_text, "context": context, "question": question})

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<p class="sidebar-header">📂 Document</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf",
        help="Upload any PDF to start chatting with it"
    )

    if uploaded_file:
        st.markdown(f"**{uploaded_file.name}**")
        st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")

        if st.button("⚡ Process PDF"):
            with st.spinner("Analyzing document..."):
                chunks, pages = ingest_pdf(uploaded_file)
            st.session_state.pdf_ready = True
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_chunks = chunks
            st.session_state.pdf_pages = pages
            st.success("Ready to chat!")

    if st.session_state.get("pdf_ready"):
        st.divider()
        st.markdown('<p class="sidebar-header">📊 Stats</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state.pdf_pages}</div>
                <div class="stat-label">Pages</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state.pdf_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown('<p class="sidebar-header">⚙️ Powered by</p>', unsafe_allow_html=True)
    st.caption("🔗 LangChain")
    st.caption("📦 Pinecone")
    st.caption("🤖 OpenAI GPT-4o-mini")

# ---- Main Area ----
st.markdown('<p class="main-title">🧠 DocChat AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a PDF and have a conversation with it</p>', unsafe_allow_html=True)

if not st.session_state.get("pdf_ready"):
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🧠</div>
        <div class="title">No document loaded yet</div>
        <div class="desc">Upload a PDF from the sidebar on the left<br>and start having a conversation with it</div>
    </div>
    """, unsafe_allow_html=True)

else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(f"""
        <div style="text-align:center; padding:40px 20px;">
            <div class="ready-badge">✅ Document Ready</div>
            <div style="color:#9ca3af; margin-top:12px;">
                <b style="color:#667eea">{st.session_state.pdf_name}</b> has been processed.<br>
                <span style="color:#4b5563; font-size:0.9rem;">Ask anything about it below 👇</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input(f"Ask about {st.session_state.get('pdf_name', 'your document')}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(prompt, st.session_state.messages)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})