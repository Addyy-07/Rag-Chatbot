import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def ingest(pdf_path):
    print("Step 1: Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")

    print("Step 2: Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Step 3: Setting up Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if index_name not in [i.name for i in pc.list_indexes()]:
        print("  Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=384,       # HuggingFace all-MiniLM-L6-v2 = 384 dims
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("  Index created!")
    else:
        print("  Index already exists.")

    print("Step 4: Embedding and uploading to Pinecone...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    PineconeVectorStore.from_documents(
        chunks, embeddings,
        index_name=index_name
    )
    print(f"\n✅ Done! {len(chunks)} chunks stored in Pinecone.")

if __name__ == "__main__":
    ingest("document.pdf")