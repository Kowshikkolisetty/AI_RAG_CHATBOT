from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import DEFAULT_PDF_PATH, get_env, ensure_data_dir
from src.graph import build_rag_graph
from src.ingestion import download_reference_pdf, ingest_pdf

app = FastAPI(title="Agentic AI RAG Chatbot", version="1.0.0")


graph = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


class IngestRequest(BaseModel):
    pdf_url: Optional[str] = None


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Agentic AI RAG Chatbot",
        "index_name": get_env("PINECONE_INDEX_NAME", "agentic-ai-index"),
    }


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    global graph
    if graph is None:
        try:
            graph = build_rag_graph(index_name=get_env("PINECONE_INDEX_NAME", "agentic-ai-index"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"RAG graph is unavailable: {exc}") from exc

    initial_state = {"question": request.query, "context": [], "answer": "", "score": 0.0}
    result = graph.invoke(initial_state)

    return QueryResponse(
        final_answer=result.get("answer", "I cannot answer based on the provided document."),
        retrieved_context=result.get("context", []),
        confidence_score=float(result.get("score", 0.0)),
    )


@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    ensure_data_dir()
    pdf_target = DEFAULT_PDF_PATH

    try:
        if request.pdf_url:
            pdf_target = download_reference_pdf(request.pdf_url)
        elif not pdf_target.exists():
            raise FileNotFoundError("No PDF was found in the data directory. Please provide a URL or add the ebook to data/Ebook-Agentic-AI.pdf.")

        ingest_pdf(pdf_path=pdf_target, index_name=get_env("PINECONE_INDEX_NAME", "agentic-ai-index"))
        return {
            "status": "ok",
            "message": "Document ingested and indexed successfully.",
            "pdf_path": str(pdf_target),
            "index_name": get_env("PINECONE_INDEX_NAME", "agentic-ai-index"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Document ingestion failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
