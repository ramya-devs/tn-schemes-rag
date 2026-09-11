import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# Load .env
load_dotenv()

DATA_DIR = Path("data")
COLLECTION_NAME = "tn_schemes"


print("========================================")
print("TN SCHEMES RAG - INGESTION")
print("========================================")


# --------------------------------------------------
# 1. Find PDF files
# --------------------------------------------------

pdf_files = list(DATA_DIR.glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDF files.")

if not pdf_files:
    raise ValueError("No PDF files found inside data/")


# --------------------------------------------------
# 2. Load PDFs
# --------------------------------------------------

all_documents = []

for pdf_file in pdf_files:

    print(f"\nLoading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    all_documents.extend(documents)

    print(f"  Pages loaded: {len(documents)}")


print(f"\nTotal pages: {len(all_documents)}")


# --------------------------------------------------
# 3. Create chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(all_documents)

print(f"Total chunks: {len(chunks)}")


# --------------------------------------------------
# 4. Check API keys
# --------------------------------------------------

google_api_key = os.getenv("GOOGLE_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

if not qdrant_url:
    raise ValueError("QDRANT_URL is missing from .env")

if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY is missing from .env")


print("\nEnvironment variables found.")


# --------------------------------------------------
# 5. Create Gemini embedding model
# --------------------------------------------------

print("\nCreating Gemini embedding model...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

print("Gemini embedding model ready.")


# --------------------------------------------------
# 6. Connect to Qdrant
# --------------------------------------------------

print("\nConnecting to Qdrant...")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

print("Qdrant connected.")


# --------------------------------------------------
# 7. Upload chunks to Qdrant
# --------------------------------------------------

print("\nUploading chunks to Qdrant...")
print("This may take a little while...\n")

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=qdrant_url,
    api_key=qdrant_api_key,
    collection_name=COLLECTION_NAME,
)

print("Upload completed!")


# --------------------------------------------------
# 8. Verify Qdrant
# --------------------------------------------------

print("\nQdrant collections:")

collections = client.get_collections()

for collection in collections.collections:
    print(f"  - {collection.name}")


print("\n========================================")
print("RAG INGESTION COMPLETED SUCCESSFULLY!")
print("========================================")