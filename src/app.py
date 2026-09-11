import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_qdrant import QdrantVectorStore


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TN Government Schemes Assistant",
    page_icon="🌾",
    layout="centered",
)


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

COLLECTION_NAME = "tn_schemes"

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


if not qdrant_url:
    st.error("QDRANT_URL is missing from .env")
    st.stop()


if not qdrant_api_key:
    st.error("QDRANT_API_KEY is missing from .env")
    st.stop()


# ============================================================
# 3. APPLICATION TITLE
# ============================================================

st.title("🌾 TN Government Schemes Assistant")

st.write(
    "Ask questions about Tamil Nadu government agricultural "
    "schemes using the available official government documents."
)


# ============================================================
# 4. LOAD GEMINI EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )


# ============================================================
# 5. LOAD QDRANT VECTOR STORE
# ============================================================

@st.cache_resource
def load_vector_store():

    embeddings = load_embeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    return vector_store


# ============================================================
# 6. LOAD GEMINI LANGUAGE MODEL
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
    )


# ============================================================
# 7. LOAD RAG COMPONENTS
# ============================================================

try:

    vector_store = load_vector_store()
    llm = load_llm()

except Exception:

    st.error(
        "⚠️ Unable to connect to the RAG system. "
        "Please check your Qdrant configuration and API keys."
    )

    st.stop()


# ============================================================
# 8. EXAMPLE QUESTIONS
# ============================================================

st.subheader("💡 Example Questions")

example_questions = [
    "குறுவை சிறப்புத் தொகுப்புத் திட்டம் 2026 என்றால் என்ன?",
    "How much is allocated for the Kuruvai Special Package 2026?",
    "What is the subsidy provided under the horticulture pandal scheme?",
    "What schemes are available in the documents?",
]


for example in example_questions:

    if st.button(
        example,
        use_container_width=True,
    ):

        st.session_state["question"] = example


# ============================================================
# 9. QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask your question",
    value=st.session_state.get("question", ""),
    placeholder=(
        "Example: குறுவை சிறப்புத் தொகுப்புத் திட்டம் "
        "2026 என்றால் என்ன?"
    ),
)


# ============================================================
# 10. BUTTONS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    ask_button = st.button(
        "🔍 Ask Question",
        use_container_width=True,
    )


with col2:

    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True,
    )


# ============================================================
# 11. CLEAR BUTTON
# ============================================================

if clear_button:

    st.session_state.pop("question", None)

    st.rerun()


# ============================================================
# 12. PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning("Please enter a question.")

        st.stop()


    # ========================================================
    # 13. RETRIEVE RELEVANT DOCUMENTS
    # ========================================================

    with st.spinner("🔎 Searching government documents..."):

        try:

            results = vector_store.similarity_search(
                question,
                k=6,
            )

        except Exception:

            st.error(
                "⚠️ Error while searching the government "
                "documents. Please try again."
            )

            st.stop()


    if not results:

        st.warning(
            "No relevant information was found in the "
            "available government documents."
        )

        st.stop()


    # ========================================================
    # 14. BUILD CONTEXT
    # ========================================================

    context_parts = []


    for document in results:

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )


        if isinstance(page, int):

            page_number = page + 1

        else:

            page_number = page


        context_parts.append(
            f"""
SOURCE: {source}

PAGE: {page_number}

CONTENT:
{document.page_content}
"""
        )


    context = "\n".join(context_parts)


    # ========================================================
    # 15. RAG PROMPT
    # ========================================================

    prompt = f"""
You are a Tamil Nadu government scheme information assistant.

Your job is to answer the user's question ONLY using the
government-document CONTEXT provided below.

STRICT RULES:

1. Use only facts explicitly present in the CONTEXT.

2. Do not use your own general knowledge.

3. Do not guess, assume, or fill missing information.

4. If the CONTEXT does not contain enough information
   to answer the question, say exactly:

"இந்த தகவல் வழங்கப்பட்ட அரசு ஆவணங்களில் இல்லை."

5. Do not invent:
   - names
   - dates
   - subsidy amounts
   - eligibility
   - districts
   - application procedures
   - benefits
   - scheme details

6. If the user asks in Tamil, answer in simple Tamil.

7. If the user asks in English, answer in clear English.

8. Keep the answer concise and directly related to
   the user's question.

9. Do not add unnecessary information.

10. Do not make claims that are not supported by the
    provided CONTEXT.

11. Do not include a source section in your answer.
    The application will display sources separately.

12. If the question asks for a list of schemes or projects,
    identify the relevant schemes or projects mentioned
    across the provided CONTEXT.

USER QUESTION:
{question}

CONTEXT:
{context}

Now answer the user's question using ONLY the CONTEXT.
"""


    # ========================================================
    # 16. GENERATE ANSWER
    # ========================================================

    with st.spinner("🤖 Generating answer..."):

        try:

            response = llm.invoke(prompt)

            answer = response.content


        except Exception as e:

            error_message = str(e)

            # ------------------------------------------------
            # GEMINI QUOTA ERROR
            # ------------------------------------------------

            if (
                "RESOURCE_EXHAUSTED" in error_message
                or "429" in error_message
                or "quota" in error_message.lower()
            ):

                st.error(
                    "⚠️ Gemini API quota has been reached."
                )

                st.info(
                    "The document search is working, but "
                    "Gemini's answer-generation quota is "
                    "temporarily unavailable. Please try "
                    "again after the quota resets."
                )

            # ------------------------------------------------
            # OTHER GEMINI ERROR
            # ------------------------------------------------

            else:

                st.error(
                    "⚠️ An error occurred while generating "
                    "the answer. Please try again."
                )

            st.stop()


    # ========================================================
    # 17. CLEAN GEMINI RESPONSE
    # ========================================================

    if isinstance(answer, list):

        answer = "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )


    answer = str(answer).strip()


    # ========================================================
    # 18. DISPLAY ANSWER
    # ========================================================

    st.subheader("🤖 Answer")

    st.markdown(answer)


    # ========================================================
    # 19. CHECK WHETHER ANSWER IS "NOT FOUND"
    # ========================================================

    not_found_message = (
        "இந்த தகவல் வழங்கப்பட்ட அரசு ஆவணங்களில் இல்லை."
    )

    answer_is_not_found = (
        not_found_message in answer
    )


    # ========================================================
    # 20. DISPLAY SOURCES
    # ========================================================

    if not answer_is_not_found:

        st.subheader("📄 Sources")

        sources = []


        for document in results:

            source = Path(
                document.metadata.get(
                    "source",
                    "Unknown",
                )
            ).name


            page = document.metadata.get(
                "page",
                "Unknown",
            )


            if isinstance(page, int):

                page_number = page + 1

            else:

                page_number = page


            sources.append(
                (
                    source,
                    page_number,
                )
            )


        # ----------------------------------------------------
        # Remove duplicates
        # ----------------------------------------------------

        sources = list(set(sources))


        # ----------------------------------------------------
        # Sort sources
        # ----------------------------------------------------

        def sort_key(item):

            source, page = item

            if isinstance(page, int):

                return (
                    source.lower(),
                    page,
                )

            return (
                source.lower(),
                9999,
            )


        sources.sort(key=sort_key)


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        for source, page in sources:

            st.markdown(
                f"• **{source}** — Page {page}"
            )


# ============================================================
# 21. FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by Gemini + Qdrant + LangChain | "
    "RAG-based Tamil Nadu Government Scheme Assistant"
)