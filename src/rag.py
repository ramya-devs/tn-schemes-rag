import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_qdrant import QdrantVectorStore


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

COLLECTION_NAME = "tn_schemes"


# ============================================================
# 2. GET QDRANT SETTINGS
# ============================================================

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url:
    raise ValueError("QDRANT_URL is missing from .env")

if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY is missing from .env")


# ============================================================
# 3. CREATE GEMINI EMBEDDING MODEL
# ============================================================

print("Creating Gemini embedding model...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

print("Embedding model ready.")


# ============================================================
# 4. CONNECT TO QDRANT
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
# 5. CREATE GEMINI LANGUAGE MODEL
# ============================================================

print("Creating Gemini language model...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

print("Gemini language model ready.")


# ============================================================
# 6. ASK USER FOR A QUESTION
# ============================================================

question = input("\nAsk your question: ")


# ============================================================
# 7. SEARCH QDRANT
# ============================================================

print("\nSearching Qdrant...")

results = vector_store.similarity_search(
    question,
    k=3,
)

print(f"Retrieved {len(results)} relevant documents.")


# ============================================================
# 8. BUILD CONTEXT
# ============================================================

context_parts = []

for i, document in enumerate(results):

    source = document.metadata.get(
        "source",
        "Unknown source"
    )

    page = document.metadata.get(
        "page",
        "Unknown page"
    )

    context_parts.append(
        f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{document.page_content}
"""
    )


context = "\n".join(context_parts)


# ============================================================
# 9. CREATE RAG PROMPT
# ============================================================

prompt = f"""
You are a Tamil Nadu government scheme information assistant.

Your job is to answer the user's question ONLY using the
government-document CONTEXT provided below.

STRICT RULES:

1. Use only facts explicitly present in the CONTEXT.
2. Do not use your own general knowledge.
3. Do not guess, assume, or fill missing information.
4. If the CONTEXT does not contain the answer, say:

"இந்த தகவல் வழங்கப்பட்ட அரசு ஆவணங்களில் இல்லை."

5. Do not invent:
   - names
   - dates
   - subsidy amounts
   - eligibility
   - districts
   - application procedures
   - benefits

6. If the user asks in Tamil, answer in simple Tamil.

7. If the user asks in English, answer in English.

8. Keep the answer concise and easy to understand.

9. Do not mention information that is not supported by
   the provided CONTEXT.

10. At the end, mention the source document and page number
    used for the answer.

USER QUESTION:
{question}

CONTEXT:
{context}

Now answer the question using ONLY the CONTEXT.
"""


# ============================================================
# 10. GENERATE ANSWER
# ============================================================

print("\nGenerating answer...")

response = llm.invoke(prompt)

answer = response.content

if isinstance(answer, list):
    answer = "".join(
        item.get("text", "") if isinstance(item, dict) else str(item)
        for item in answer
    )


# ============================================================
# 11. DISPLAY ANSWER
# ============================================================

print("\n" + "=" * 60)
print("AI ANSWER")
print("=" * 60)

print(answer)


# ============================================================
# 12. DISPLAY SOURCES
# ============================================================

print("\n" + "=" * 60)
print("📄 SOURCES USED")
print("=" * 60)

for document in results:

    source = Path(
        document.metadata.get(
            "source",
            "Unknown"
        )
    ).name

    page = document.metadata.get(
        "page",
        "Unknown"
    )

    if isinstance(page, int):
        print(f"- {source} — Page {page + 1}")
    else:
        print(f"- {source} — Page {page}")


# ============================================================
# 13. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("RAG TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)