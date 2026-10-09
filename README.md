# RAG Documentation Assistant

A Retrieval-Augmented Generation (RAG) application built using **Python, LangChain, Pinecone, Tavily, and Streamlit** to deliver context-aware responses from technical documentation.

The project demonstrates end-to-end RAG implementation, including document ingestion, vector-based retrieval, LLM integration, and a conversational interface.

## Key Features

- **Document Ingestion:** Automated web crawling and content extraction using Tavily.
- **Text Processing:** Recursive text splitting with configurable chunk size and overlap.
- **Embeddings & Vector Search:** Embedding generation and semantic retrieval using Pinecone.
- **LLM Integration:** Context-aware response generation using retrieved documentation and configurable LLM providers.
- **Conversational Interface:** Streamlit-based chat with session management and source references.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, LangChain |
| Document Extraction | Tavily |
| Vector Database | Pinecone |
| Embeddings | OpenAI / NVIDIA |
| LLM Providers | OpenAI-compatible APIs / OpenRouter |
| Frontend | Streamlit |
| Dependency Management | Pipenv |

## Architecture

**Document Ingestion**

```text
Documentation → Tavily → Text Splitting → Embeddings → Pinecone
```

**Query Processing**

```text
User Query → Streamlit → Retriever → Pinecone
                                   ↓
                         Retrieved Context
                                   ↓
                              LLM → Answer
                                    + Sources
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/Hetal-3/langchain-rag-documentation-assistant.git
cd langchain-rag-documentation-assistant
```

**2. Install dependencies**

```bash
pipenv install
```

**3. Configure environment variables**

Create a `.env` file containing the required Tavily, Pinecone, and LLM provider API keys. Ensure credentials are excluded from version control.

**4. Run ingestion**

```bash
pipenv run python ingestion.py
```

**5. Start the application**

```bash
pipenv run streamlit run main.py
```

## Implementation Scope

- RAG pipeline development and document preprocessing
- Embedding generation, batching, and vector retrieval
- LLM and retrieval-tool integration
- Streamlit session-state management and source attribution
- API integration troubleshooting and execution-flow analysis

## Attribution

Adapted from [Eden Marco's Documentation Helper](https://github.com/emarco177/documentation-helper) as part of independent exploration of RAG architecture and LLM-based application development.

The original implementation served as the foundation for subsequent experimentation and modifications. Applicable copyright and license notices are retained in the repository.

## Project Status

**Proof of Concept (POC)** — Developed for hands-on exploration of RAG-based application architecture; not intended as a production-ready system.