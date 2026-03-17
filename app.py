import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )
    return chain

# ----- UI -----
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Powered by LangChain + Pinecone + OpenAI")

chain = load_chain()

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
            result = chain.invoke({"question": prompt})
            answer = result["answer"]
        st.write(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})