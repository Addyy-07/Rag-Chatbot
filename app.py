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
        chunks,
        embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    os.unlink(tmp_path)  
    return len(chunks)

def get_retriever():
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def get_answer(question, chat_history):
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    history_text = ""
    for msg in chat_history[-4:]:
        if msg["role"] == "user":
            history_text += f"Human: {msg['content']}\n"
        else:
            history_text += f"Assistant: {msg['content']}\n"

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided document context.
If the answer isn't in the context, say "I couldn't find that in the document."

Previous conversation:
{history}

Context from document:
{context}

Question: {question}

Answer:""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"history": history_text, "context": context, "question": question})

# ----- UI -----
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Powered by LangChain + Pinecone + OpenAI")

# Sidebar
with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Reading and indexing your PDF..."):
                num_chunks = ingest_pdf(uploaded_file)
            st.success(f"✅ Done! Created {num_chunks} chunks.")
            st.session_state.pdf_ready = True

    if st.session_state.get("pdf_ready"):
        st.info("PDF is ready! Ask questions in the chat.")

# Chat section
if not st.session_state.get("pdf_ready"):
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your document..."):
                answer = get_answer(prompt, st.session_state.messages)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})