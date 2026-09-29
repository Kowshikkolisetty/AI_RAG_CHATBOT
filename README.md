# Agentic AI RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot for the Agentic AI eBook reference document. It ingests a PDF, splits it into chunks, embeds the chunks with OpenAI embeddings, stores them in Pinecone, and answers queries using a LangGraph workflow.

## Features

- PDF ingestion and chunking with PyPDF and RecursiveCharacterTextSplitter
- Vector storage in Pinecone
- LangGraph stateful RAG workflow
- FastAPI backend exposing `/chat` and `/ingest`
- Optional Streamlit interface
- Sample benchmark queries for validation

## Project Structure

```text
AI-RAG/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
├── tests_sample_queries.py
└── data/
    └── Ebook-Agentic-AI.pdf
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and add your keys:

```bash
copy .env.example .env
```

Then update `.env` with:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

## Download the PDF

Use the reference PDF URL from the assignment or the helper in the code:

```python
from src.ingestion import download_reference_pdf

pdf_path = download_reference_pdf("https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing")
print(pdf_path)
```

## Ingest the document

```bash
python -c "from src.ingestion import download_reference_pdf, ingest_pdf; from src.config import DATA_DIR; download_reference_pdf('https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing'); ingest_pdf()"
```

Or start the API and call `/ingest`.

## Run the API

```bash
uvicorn app:app --reload
```

The app exposes:

- `GET /` - health info
- `POST /chat` - ask a question against the vector store
- `POST /ingest` - index a PDF by URL or default local file

Example request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Agentic AI according to the eBook?"}'
```

## Optional Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Validation Queries

Try these benchmark queries:

1. "What is Agentic AI according to the eBook?"
2. "How do AI agents differ from traditional automation systems?"
3. "What are the core components of an Agentic Architecture?"
4. "What role does memory play in Agentic AI workflows?"
5. "Who won the 2022 FIFA World Cup?"

The final query should fail gracefully when the evidence is outside the loaded document context.

## Notes

- The app is intentionally strict about grounding: if the document context does not support an answer, the model replies with a grounded refusal.
- Pinecone and OpenAI credentials are required for a live retrieval pipeline.
