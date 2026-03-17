import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(question, chat_history, retriever):
    # Build context string from history
    history_text = ""
    for msg in chat_history[-4:]:  # last 4 messages for context
        if msg["role"] == "user":
            history_text += f"Human: {msg['content']}\n"
        else:
            history_text += f"Assistant: {msg['content']}\n"

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build prompt
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

    return chain.invoke({
        "history": history_text,
        "context": context,
        "question": question
    })

# ----- UI -----
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Powered by LangChain + Pinecone + OpenAI")

retriever = load_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask anything about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching your document..."):
            answer = get_answer(prompt, st.session_state.messages, retriever)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})