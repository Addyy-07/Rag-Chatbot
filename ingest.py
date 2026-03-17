import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load your API keys from .env
load_dotenv()

print("Step 1: Loading PDF...")
loader = PyPDFLoader("document.pdf")
documents = loader.load()
print(f"  Loaded {len(documents)} pages")

print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # each chunk = ~1000 characters
    chunk_overlap=200   # 200 chars overlap between chunks for context
)
chunks = splitter.split_documents(documents)
print(f"  Created {len(chunks)} chunks")

print("Step 3: Setting up Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

# Create the index only if it doesn't exist yet
if index_name not in [i.name for i in pc.list_indexes()]:
    print("  Creating new index...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings are 1536 numbers long
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("  Index created!")
else:
    print("  Index already exists, skipping creation.")

print("Step 4: Embedding chunks and uploading to Pinecone...")
print("  (This may take a minute...)")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

print("\n✅ Done! Your PDF is now stored in Pinecone.")