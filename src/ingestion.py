import re
from pathlib import Path
from typing import Optional, Sequence

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import DATA_DIR, DEFAULT_PDF_PATH, ensure_data_dir, get_env


def parse_google_drive_file_id(url: str) -> str:
    match = re.search(r"/d/([A-Za-z0-9_-]+)", url)
    if not match:
        raise ValueError("Invalid Google Drive URL. Please supply a direct file link.")
    return match.group(1)


def download_reference_pdf(url: str, destination: Optional[str | Path] = None) -> Path:
    ensure_data_dir()
    target_path = Path(destination) if destination else DEFAULT_PDF_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    file_id = parse_google_drive_file_id(url)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url, timeout=120, stream=True)
    response.raise_for_status()

    with open(target_path, "wb") as file_handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_handle.write(chunk)

    return target_path


def create_pinecone_index(index_name: Optional[str] = None, dimension: int = 1536, metric: str = "cosine") -> str:
    index_name = index_name or get_env("PINECONE_INDEX_NAME", "agentic-ai-index")
    api_key = get_env("PINECONE_API_KEY", "")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")

    pinecone_client = Pinecone(api_key=api_key)
    existing_indexes = [index.name for index in pinecone_client.list_indexes().indexes]

    if index_name not in existing_indexes:
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return index_name


def load_pdf_documents(pdf_path: str | Path) -> list:
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents: Sequence) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def ingest_pdf(
    pdf_path: str | Path | None = None,
    index_name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> PineconeVectorStore:
    target_pdf = Path(pdf_path) if pdf_path else DEFAULT_PDF_PATH
    if not target_pdf.exists():
        raise FileNotFoundError(f"PDF not found at {target_pdf}. Download or place the file in the data directory.")

    openai_key = get_env("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

    resolved_index_name = index_name or get_env("PINECONE_INDEX_NAME", "agentic-ai-index")
    create_pinecone_index(index_name=resolved_index_name)

    documents = load_pdf_documents(target_pdf)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=resolved_index_name,
    )
    return vector_store
