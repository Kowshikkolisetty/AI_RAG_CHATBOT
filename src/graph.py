from typing import Any, List, Optional, TypedDict

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency path
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from langchain_pinecone import PineconeVectorStore
except Exception:  # pragma: no cover - optional dependency path
    PineconeVectorStore = None

from langgraph.graph import END, START, StateGraph

from .config import get_env


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def get_vectorstore(index_name: Optional[str] = None) -> Any:
    if OpenAIEmbeddings is None or PineconeVectorStore is None:
        raise RuntimeError("Pinecone/OpenAI dependencies are unavailable.")

    resolved_index_name = index_name or get_env("PINECONE_INDEX_NAME", "agentic-ai-index")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index_name=resolved_index_name, embedding=embeddings)


def build_rag_graph(index_name: Optional[str] = None, vectorstore: Optional[Any] = None):
    active_index_name = index_name or get_env("PINECONE_INDEX_NAME", "agentic-ai-index")
    retriever = None
    llm = None

    if (
        PineconeVectorStore is not None
        and OpenAIEmbeddings is not None
        and get_env("PINECONE_API_KEY", "").strip()
        and get_env("OPENAI_API_KEY", "").strip()
    ):
        try:
            active_vectorstore = vectorstore or get_vectorstore(active_index_name)
            retriever = active_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        except Exception:
            retriever = None

    if ChatOpenAI is not None and get_env("OPENAI_API_KEY", "").strip():
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception:
            llm = None

    def retrieve_node(state: AgentState):
        question = state.get("question", "")
        if not retriever or not question:
            return {"context": []}

        documents = retriever.invoke(question)
        context_texts: List[str] = []
        for item in documents:
            if hasattr(item, "page_content"):
                context_texts.append(item.page_content)
            else:
                context_texts.append(str(item))

        return {"context": context_texts}

    def generate_node(state: AgentState):
        context = state.get("context", [])

        if not context:
            return {
                "answer": "I cannot answer based on the provided document.",
                "score": 0.0,
            }

        if llm is None:
            safe_answer = "AI generation is unavailable because OpenAI API credentials are not configured yet. Retrieved context is available below."
            return {"answer": safe_answer, "score": 0.0}

        context_text = "\n\n".join(context)
        prompt = f"""You are a strict assistant. Answer using only the provided context below.
If the context does not contain enough information, explicitly say that you cannot answer based on the provided document.

Context:
{context_text}

Question: {state['question']}"""

        response = llm.invoke(prompt)
        answer = response.content.strip() if hasattr(response, "content") else str(response)
        score = round(min(0.99, 0.35 + (len(context) * 0.2)), 2)

        return {"answer": answer, "score": score}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
