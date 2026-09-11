import os
import re
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore


load_dotenv()

COLLECTION_NAME = "tn_schemes"


# ============================================================
# CONNECT TO QDRANT
# ============================================================

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url:
    raise ValueError("QDRANT_URL is missing from .env")

if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY is missing from .env")


print("Creating Gemini embedding model...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

print("Embedding model ready.")


print("Connecting to Qdrant...")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=qdrant_url,
    api_key=qdrant_api_key,
)

print("Connected to tn_schemes.")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text so that small OCR/formatting differences
    do not incorrectly cause an evaluation failure.
    """

    text = text.lower()

    # OCR commonly reads letter O instead of zero.
    text = re.sub(r"(?<=\d)o(?=[\d.,])", "0", text)

    # Remove currency symbols.
    text = text.replace("₹", " ")

    # Normalize common punctuation.
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("-", " ")
    text = text.replace("/", " ")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def fact_matches(fact, retrieved_text):
    """
    Check a fact using several representations.

    This avoids false failures caused by:
    - OCR
    - punctuation
    - currency symbols
    - spacing
    - lakh/crore formatting
    """

    fact_normalized = normalize_text(fact)
    text_normalized = normalize_text(retrieved_text)

    # Direct normalized match.
    if fact_normalized in text_normalized:
        return True

    # Special handling for common numeric expressions.
    fact_number = re.search(
        r"\d+(?:\.\d+)?",
        fact_normalized
    )

    if fact_number:
        number = fact_number.group()

        # Check number even when surrounding formatting differs.
        if number in text_normalized:

            # crore
            if "crore" in fact_normalized:
                if "கைாடி" in retrieved_text.lower():
                    return True

            # lakh
            if "lakh" in fact_normalized:
                if "lakh" in text_normalized:
                    return True

            # hectare / ha
            if "hectare" in fact_normalized:
                if "ha" in text_normalized:
                    return True

    return False


# ============================================================
# TEST DATA
# ============================================================

tests = [

    # --------------------------------------------------------
    # KURUVAI
    # --------------------------------------------------------

    {
        "question": "How much is allocated for the Kuruvai Special Package 2026?",
        "expected_source": "kuruvai_special_package_2026.pdf",
        "expected_facts": [
            "134.83 crore",
        ],
    },

    {
        "question": "How much is allocated for delta districts?",
        "expected_source": "kuruvai_special_package_2026.pdf",
        "expected_facts": [
            "77.50 crore",
        ],
    },

    {
        "question": "How much is allocated for non-delta districts?",
        "expected_source": "kuruvai_special_package_2026.pdf",
        "expected_facts": [
            "57.33 crore",
        ],
    },

    {
        "question": "What seasons are covered by the Kuruvai Special Package?",
        "expected_source": "kuruvai_special_package_2026.pdf",
        "expected_facts": [
            "Kuruvai",
            "Kar",
            "Sornavari",
        ],
    },

    {
        "question": "What are the main objectives of the Kuruvai Special Package?",
        "expected_source": "kuruvai_special_package_2026.pdf",
        "expected_facts": [
            "paddy cultivation",
            "paddy production",
            "productivity",
        ],
    },

    {
        "question": "What incentives are provided for paddy cultivation?",
        "expected_source": "kuruvai_special_package_2026.pdf",
        "expected_facts": [
            "mechanical transplantation",
            "direct sowing",
            "bio-fertilizers",
        ],
    },


    # --------------------------------------------------------
    # PANDAL
    # --------------------------------------------------------

    {
        "question": "What is the subsidy under the Permanent Pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "3.00 Lakh",
            "Ha",
        ],
    },

    {
        "question": "How much subsidy is provided per hectare for the pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "3.00 Lakh",
            "Ha",
        ],
    },

    {
        "question": "What is the total cost considered for one hectare under the pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "6,00,000",
            "1Ha",
        ],
    },

    {
        "question": "What is the subsidy for land preparation and pandal structures?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "land preparation",
            "pandal",
        ],
    },

    {
        "question": "What is the cost of GI wires under the pandal scheme?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "GI Wires",
        ],
    },

    {
        "question": "What is the subsidy for planting material and labour?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "planting material",
            "labour",
        ],
    },

    {
        "question": "Is there an additional subsidy for SC and ST small and marginal farmers?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "SC",
            "ST",
            "small",
            "marginal farmers",
        ],
    },

    {
        "question": "What supporting structures for horticultural crops are mentioned?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "supporting structures",
            "horticultural crops",
        ],
    },

    {
        "question": "What is PM RKVY?",
        "expected_source": "pandal_go_117.pdf",
        "expected_facts": [
            "PM-RKVY",
        ],
    },
]


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TN GOVERNMENT SCHEMES - IMPROVED RAG EVALUATION")
print("=" * 70)

print(f"\nTotal tests: {len(tests)}")


total_facts = 0
matched_facts = 0

passed_tests = 0
failed_tests = 0

source_hits = 0


for index, test in enumerate(tests, start=1):

    question = test["question"]
    expected_source = test["expected_source"]
    expected_facts = test["expected_facts"]


    print("\n" + "-" * 70)
    print(f"TEST {index}/{len(tests)}")
    print(f"Question: {question}")


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    results = vector_store.similarity_search(
        question,
        k=6,
    )


    retrieved_text = ""

    retrieved_sources = set()


    for document in results:

        source = Path(
            document.metadata.get(
                "source",
                "Unknown"
            )
        ).name

        retrieved_sources.add(source)

        retrieved_text += "\n" + document.page_content


    # --------------------------------------------------------
    # SOURCE CHECK
    # --------------------------------------------------------

    source_pass = expected_source in retrieved_sources


    if source_pass:
        print("✅ Expected source retrieved.")
        source_hits += 1
    else:
        print("❌ Expected source NOT retrieved.")


    # --------------------------------------------------------
    # FACT CHECK
    # --------------------------------------------------------

    test_pass = source_pass


    print("\nFact checks:")


    for fact in expected_facts:

        total_facts += 1


        if fact_matches(
            fact,
            retrieved_text
        ):

            print(f"  ✅ {fact}")

            matched_facts += 1

        else:

            print(f"  ❌ {fact}")

            test_pass = False


    # --------------------------------------------------------
    # TEST RESULT
    # --------------------------------------------------------

    if test_pass:

        print("\n✅ TEST PASSED")

        passed_tests += 1

    else:

        print("\n❌ TEST FAILED")

        failed_tests += 1


# ============================================================
# FINAL SCORES
# ============================================================

source_score = (
    source_hits / len(tests)
) * 100


test_score = (
    passed_tests / len(tests)
) * 100


if total_facts > 0:

    fact_score = (
        matched_facts / total_facts
    ) * 100

else:

    fact_score = 0


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RAG EVALUATION RESULTS")
print("=" * 70)

print(f"Total tests        : {len(tests)}")

print(f"Source hits        : {source_hits}")

print(f"Source retrieval   : {source_score:.2f}%")

print()

print(f"Tests passed       : {passed_tests}")

print(f"Tests failed       : {failed_tests}")

print(f"Test pass rate     : {test_score:.2f}%")

print()

print(f"Expected facts     : {total_facts}")

print(f"Matched facts      : {matched_facts}")

print(f"Context fact score : {fact_score:.2f}%")

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)