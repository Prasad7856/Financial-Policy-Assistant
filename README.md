# Financial Policy Assistant 📚

Answer questions about financial policies and regulations using a hybrid Retrieval-Augmented Generation (RAG) pipeline that prioritises relevant evidence over LLM guesswork.

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

Financial policy documents are often long, fragmented, and difficult to search using traditional keyword matching. This project allows users to ask questions in natural language and retrieves the most relevant sections before generating an answer grounded in the source documents.

Instead of relying solely on semantic similarity, the retrieval pipeline combines dense vector search, lexical search (BM25), and reranking to improve relevance and reduce incorrect or unsupported responses.

---

# Key Features

- 🔍 Retrieves relevant policy sections using Hybrid Retrieval (Semantic Search + BM25)
- 📄 Answers natural language questions grounded in retrieved financial documents
- 🎯 Uses reranking to improve context quality before passing documents to the LLM
- ⚡ FastAPI backend exposes REST APIs for document ingestion and querying
- 🧠 LangChain orchestrates retrieval, prompt construction, and response generation
- 🗂️ Stores document embeddings in Pinecone for low-latency vector search
- 📑 Supports ingestion of financial and regulatory documents for semantic indexing
- 🔒 Reduces hallucinations by constraining responses to retrieved evidence

---

# Tech Stack

| Category | Technologies |
|----------|--------------|
| 🎨 Frontend | *(Optional UI / Streamlit or React if applicable)* |
| ⚙️ Backend | Python, FastAPI, LangChain |
| 🧠 LLM | LLaMA 3-70B (Groq API) |
| 🔎 Retrieval | Hybrid Retrieval, BM25, Semantic Search, Reranking |
| 🗄️ Vector Database | Pinecone |
| 📚 Embeddings | HuggingFace Embeddings |
| 📄 Document Processing | PDF Parsing, Text Chunking |
| ☁️ Infrastructure | REST APIs |

---

# Architecture

The application follows a Retrieval-Augmented Generation (RAG) architecture.

1. Financial policy documents are uploaded and parsed.
2. Documents are split into semantic chunks.
3. Chunks are converted into embeddings using HuggingFace models.
4. Embeddings are indexed in Pinecone.
5. User queries trigger both semantic retrieval and BM25 lexical search.
6. Retrieved documents are reranked to improve relevance.
7. The highest-ranked context is injected into the prompt.
8. LLaMA 3-70B generates an answer using only the retrieved evidence.

> Add an architecture diagram below showing the ingestion pipeline and query flow.

```markdown
<img width="1536" height="1024" alt="rag architecture" src="https://github.com/user-attachments/assets/d9977503-649d-46ca-bc8a-e1f975485bdb" />

```

---

# Demo / Screenshots

Include:

- Querying a financial policy in natural language
- Retrieved document snippets
- Final grounded response
- API response from FastAPI
- Optional GIF showing the end-to-end workflow

Example placeholders:

```text
<img width="1920" height="1080" alt="Financial Policy Assistant" src="https://github.com/user-attachments/assets/02d4eaa2-c3c7-482a-b690-a76cf4c7f297" />

```

---

# Installation & Setup

```bash
git clone https://github.com/yourusername/financial-policy-assistant.git

cd financial-policy-assistant

python -m venv venv

source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
PINECONE_API_KEY=your_api_key
PINECONE_INDEX=financial-policy-index
```

Run the application:

```bash
uvicorn app:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

---

# Usage

Example API request:

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{
  "question":"What is the maximum loan eligibility under this policy?"
}'
```

Example response:

```json
{
  "answer": "...",
  "sources": [
    "Loan_Policy.pdf - Page 12"
  ]
}
```

---

# Roadmap

- 🚧 Add metadata filtering to retrieve policies by organization, category, or publication date.
- 🚧 Support incremental document indexing without rebuilding the entire vector store.
- 🔮 Add evaluation metrics (Recall@K, MRR, Precision) to benchmark retrieval quality.

---

# What made this project technically interesting

The most challenging part of this project was improving retrieval quality rather than simply calling an LLM. Pure semantic search frequently returned documents that were conceptually similar but not the most useful for answering policy-specific questions. To address this, I combined dense vector retrieval with BM25 lexical matching and introduced a reranking stage before prompt construction. This additional retrieval layer significantly improved the relevance of the context supplied to the language model while keeping the generation grounded in source documents. Building the retrieval pipeline required balancing retrieval accuracy, latency, and implementation complexity instead of optimizing for only one metric.

---

# License

MIT License

---

# Contact

📧 Email: your.email@example.com

💼 LinkedIn: https://linkedin.com/in/your-profile

🔗 Portfolio: https://yourportfolio.com

🔗 GitHub: https://github.com/yourusername
