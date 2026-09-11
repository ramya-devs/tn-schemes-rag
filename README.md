# 🌾 TN Government Schemes RAG Assistant

A Retrieval-Augmented Generation (RAG) application that enables users to ask natural-language questions about Tamil Nadu government agricultural schemes in both **English and Tamil**.

The system retrieves relevant information from official government documents using semantic search and generates **context-grounded answers with source document and page references**.

---

## 🚀 Demo

**Live Demo:** Coming soon

Built with **Streamlit** to provide an interactive conversational interface for querying government scheme documents.

---

## 📌 Problem Statement

Information about government agricultural schemes is often distributed across lengthy:

- Government Orders
- Press releases
- Scheme documents
- Official department publications

Finding specific information such as:

- Subsidy amounts
- Financial allocations
- Eligibility conditions
- Scheme components
- Agricultural benefits
- Implementation details

can be time-consuming when users have to manually search through multiple documents.

This project provides a conversational interface that makes it easier to retrieve relevant information from official Tamil Nadu government agricultural documents.

---

## 💡 Solution

The application uses a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of relying only on the language model's general knowledge, the system first searches the indexed government documents and retrieves relevant content. The retrieved content is then provided to the LLM as context for generating the answer.

### Workflow

1. Load official government PDF documents
2. Extract text from the documents
3. Split documents into smaller chunks
4. Generate embeddings using Gemini
5. Store embeddings in Qdrant Vector Database
6. Convert the user's question into an embedding
7. Perform semantic similarity search
8. Retrieve relevant document chunks
9. Pass the retrieved context to Gemini 3.6 Flash
10. Generate a context-grounded answer
11. Display the source document and page number

---

## 🧠 System Architecture

```text
                 Official Government PDFs
                           │
                           ▼
                    PDF Document Loader
                           │
                           ▼
                     Text Chunking
                           │
                           ▼
                  Gemini Embeddings
                           │
                           ▼
                    Qdrant Vector DB
                           │
                           │
                    User Question
                           │
                           ▼
              Semantic Similarity Search
                           │
                           ▼
                  Relevant Context
                           │
                           ▼
                   Gemini 3.6 Flash
                           │
                           ▼
                  Grounded Answer
                           │
                           ▼
              Source Document + Page
                           │
                           ▼
                    Streamlit UI


✨ Features
📄 Official government PDF document processing
🔎 Semantic similarity search
🧠 Retrieval-Augmented Generation
🌐 English and Tamil question support
📚 Source document and page number display
🛡️ Context-grounded responses
🚫 Refusal when requested information is unavailable in the indexed documents
☁️ Qdrant vector database integration
🎨 Streamlit web interface
⚡ Cached RAG resources for efficient application execution
🔐 Environment-based API key configuration

📚 Current Knowledge Base

The current prototype uses official government documents covering two agricultural scheme areas.

1. Kuruvai Special Package 2026

The indexed document contains information related to:

Financial allocation
Delta and non-delta districts
Paddy cultivation
Paddy cultivation incentives
Kuruvai, Kar and Sornavari seasons
Production and productivity-related information
2. Permanent Pandal Scheme

The indexed document contains information related to:

Subsidy details
Supporting structures for horticultural crops
Permanent pandal systems
GI wires
Planting material and labour
SC/ST additional assistance
Farmer eligibility
Implementation conditions

🛠️ Tech Stack
| Technology    | Purpose                   |
| ------------- | ------------------------- |
| Python        | Application development   |
| LangChain     | RAG orchestration         |
| Gemini        | Embeddings and LLM        |
| Qdrant        | Vector database           |
| PyPDF         | PDF text extraction       |
| Streamlit     | Web interface             |
| python-dotenv | Environment configuration |

🔄 RAG Pipeline
📥 Document Ingestion
Official Government PDFs
          │
          ▼
     PyPDFLoader
          │
          ▼
    Text Extraction
          │
          ▼
RecursiveCharacterTextSplitter
          │
          ▼
   Gemini Embeddings
          │
          ▼
   Qdrant Vector Database

🔎 Question Answering
User Question
      │
      ▼
Semantic Similarity Search
      │
      ▼
Top-K Relevant Chunks
      │
      ▼
Context Construction
      │
      ▼
Gemini 3.6 Flash
      │
      ▼
Grounded Answer
      │
      ▼
Source Document + Page Number

🔍 Example Questions
English
How much is allocated for the Kuruvai Special Package 2026?

What is the subsidy under the Permanent Pandal scheme?

What schemes are available in the documents?
Tamil
குறுவை சிறப்புத் தொகுப்புத் திட்டம் 2026 என்றால் என்ன?

🛡️ Grounded Answering

The application uses a strict prompt to keep generated answers grounded in the retrieved document context.

The LLM is instructed to:

Use only information available in the retrieved context
Avoid unsupported assumptions
Avoid inventing subsidy amounts, dates, eligibility conditions, or scheme details
Answer in Tamil when the user asks in Tamil
Answer in English when the user asks in English
Refuse to answer when the requested information is not available in the indexed documents
Example

If a user asks about a scholarship that is not covered by the indexed government documents, the application responds:

இந்த தகவல் வழங்கப்பட்ட அரசு ஆவணங்களில் இல்லை.

This demonstrates a basic grounded refusal mechanism rather than allowing the system to provide unsupported information.

📊 Evaluation
Retrieval Evaluation — Prototype

A retrieval evaluation was performed using 15 test questions covering the indexed government documents.

Metric	Result
Test questions	15
Expected source documents retrieved	15
Source retrieval rate	100%
Evaluation Note

The 100% result represents source retrieval performance, meaning the expected source document was successfully retrieved for the test questions.

It should not be interpreted as 100% end-to-end answer accuracy.

A separate automated answer-generation evaluation was limited by Gemini API quota availability and therefore was not used as an accuracy claim.

🖥️ Screenshots
🏠 Application Interface

The main Streamlit interface provides example questions in Tamil and English and allows users to ask questions about the indexed Tamil Nadu government agricultural documents.

🔎 RAG Answer with Sources

The application retrieves relevant document content and generates a grounded answer. The retrieved source documents and page numbers are displayed below the answer.

🛡️ Grounded Refusal

When the user asks for information that is not available in the indexed government documents, the application provides a grounded refusal instead of generating an unsupported answer.

📁 Project Structure
tn-schemes-rag/
│
├── data/
│   ├── kuruvai_special_package_2026.pdf
│   └── pandal_go_117.pdf
│
├── screenshots/
│   ├── 01-home.png
│   ├── 02-rag-answer.png
│   └── 03-grounded-refusal.png
│
├── src/
│   ├── ingest.py
│   ├── rag.py
│   ├── app.py
│   ├── evaluation.py
│   └── answer_evaluation.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md