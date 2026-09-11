import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

COLLECTION_NAME = "tn_schemes"

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


if not qdrant_url:
    raise ValueError("QDRANT_URL is missing from .env")

if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY is missing from .env")


# ============================================================
# 2. CREATE EMBEDDING MODEL
# ============================================================

print("Creating Gemini embedding model...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

print("Embedding model ready.")


# ============================================================
# 3. CONNECT TO QDRANT
# ============================================================

print("Connecting to Qdrant...")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=qdrant_url,
    api_key=qdrant_api_key,
)

print("Connected to tn_schemes.")


# ============================================================
# 4. TEST QUESTIONS
# ============================================================

test_questions = [

    {
        "question": "குறுவை சிறப்புத் தொகுப்புத் திட்டம் 2026 என்றால் என்ன?",
        "expected_source": "kuruvai_special_package_2026.pdf",
    },

    {
        "question": "How much is allocated for the Kuruvai Special Package 2026?",
        "expected_source": "kuruvai_special_package_2026.pdf",
    },

    {
        "question": "What is the total allocation for the Kuruvai scheme?",
        "expected_source": "kuruvai_special_package_2026.pdf",
    },

    {
        "question": "How much is allocated for delta districts?",
        "expected_source": "kuruvai_special_package_2026.pdf",
    },

    {
        "question": "How much is allocated for non-delta districts?",
        "expected_source": "kuruvai_special_package_2026.pdf",
    },

    {
        "question": "What are the main objectives of the Kuruvai Special Package?",
        "expected_source": "kuruvai_special_package_2026.pdf",
    },

    {
        "question": "What seasons are covered by the Kuruvai Special Package?",
        "expected_source": "kuruvai_special_package_2026.pdf",
    },

    {
        "question": "What incentives are provided for paddy cultivation?",
        "expected_source": "kuruvai_special_package_2026.pdf",
    },

    {
        "question": "What is the Permanent Pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What is the subsidy under the Permanent Pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "How much subsidy is provided per hectare for the pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What is the total cost considered for one hectare under the pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What is the subsidy for land preparation and pandal structures?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What is the cost of GI wires under the pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What is the subsidy for planting material and labour?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "Is there an additional subsidy for SC and ST small and marginal farmers?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What horticulture projects are mentioned in the documents?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What is PM RKVY?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What supporting structures for horticultural crops are mentioned?",
        "expected_source": "pandal_go_117.pdf",
    },

    {
        "question": "What is the scholarship amount for engineering students?",
        "expected_source": None,
    },
]


# ============================================================
# 5. RUN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TN GOVERNMENT SCHEMES RAG EVALUATION")
print("=" * 70)

print(f"\nTotal test questions: {len(test_questions)}")


correct = 0
incorrect = 0


# ============================================================
# 6. TEST EACH QUESTION
# ============================================================

for index, test in enumerate(test_questions, start=1):

    question = test["question"]
    expected_source = test["expected_source"]


    print("\n" + "-" * 70)

    print(f"TEST {index}/{len(test_questions)}")

    print(f"Question: {question}")

    print(f"Expected source: {expected_source}")


    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    try:

        results = vector_store.similarity_search(
            question,
            k=6,
        )

    except Exception as e:

        print(f"❌ Retrieval error: {e}")

        incorrect += 1

        continue


    # --------------------------------------------------------
    # Extract source filenames
    # --------------------------------------------------------

    retrieved_sources = []

    for document in results:

        source = Path(
            document.metadata.get(
                "source",
                "Unknown",
            )
        ).name

        if source not in retrieved_sources:

            retrieved_sources.append(source)


    print(
        f"Retrieved sources: {retrieved_sources}"
    )


    # --------------------------------------------------------
    # Evaluate retrieval
    # --------------------------------------------------------

    if expected_source is None:

        # This is an out-of-scope question.
        # We cannot expect a particular source.

        print(
            "⚠️ Out-of-scope question — "
            "retrieval source not required."
        )

        correct += 1

        continue


    if expected_source in retrieved_sources:

        print("✅ PASS — Expected source retrieved.")

        correct += 1

    else:

        print("❌ FAIL — Expected source NOT retrieved.")

        incorrect += 1


# ============================================================
# 7. CALCULATE SCORE
# ============================================================

total = len(test_questions)

accuracy = (correct / total) * 100


# ============================================================
# 8. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print(f"Total tests : {total}")

print(f"Passed      : {correct}")

print(f"Failed      : {incorrect}")

print(f"Retrieval score: {accuracy:.2f}%")


print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)