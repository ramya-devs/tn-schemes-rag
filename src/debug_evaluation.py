import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

COLLECTION_NAME = "tn_schemes"

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=qdrant_url,
    api_key=qdrant_api_key,
)

questions = [
    "How much is allocated for the Kuruvai Special Package 2026?",
    "How much is allocated for delta districts?",
    "How much is allocated for non-delta districts?",
    "What seasons are covered by the Kuruvai Special Package?",
    "What are the main objectives of the Kuruvai Special Package?",
    "What incentives are provided for paddy cultivation?",
    "What is the subsidy under the Permanent Pandal scheme?",
    "How much subsidy is provided per hectare for the pandal scheme?",
    "What is the total cost considered for one hectare under the pandal scheme?",
]

for number, question in enumerate(questions, start=1):

    print("\n" + "=" * 80)
    print(f"TEST {number}")
    print("=" * 80)
    print("QUESTION:")
    print(question)

    results = vector_store.similarity_search(
        question,
        k=6,
    )

    for i, document in enumerate(results, start=1):

        source = Path(
            document.metadata.get("source", "Unknown")
        ).name

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page, int):
            page = page + 1

        print("\n" + "-" * 80)
        print(f"RESULT {i}")
        print(f"SOURCE: {source}")
        print(f"PAGE: {page}")
        print("-" * 80)

        print(document.page_content)