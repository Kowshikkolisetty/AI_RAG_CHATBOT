import streamlit as st

from src.config import get_env
from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI RAG", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1f2937 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stChatMessage"] {
            padding: 0.5rem 0.75rem;
            border-radius: 18px;
            margin-bottom: 0.45rem;
            background: rgba(17, 24, 39, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            color: #f8fafc;
            font-size: 1rem;
            line-height: 1.6;
        }
        .user-bubble {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            border-color: rgba(191, 219, 254, 0.4);
        }
        .assistant-bubble {
            background: rgba(15, 23, 42, 0.9);
            border-color: rgba(96, 165, 250, 0.25);
        }
        .context-box {
            background: rgba(15, 118, 110, 0.12);
            border: 1px solid rgba(45, 212, 191, 0.35);
            padding: 0.8rem;
            border-radius: 12px;
            margin-top: 0.65rem;
        }
        .score-pill {
            display: inline-block;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.45);
            color: #bbf7d0;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .sidebar .block-container {
            background: rgba(15, 23, 42, 0.75);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Agentic AI RAG Chatbot")
st.caption("Ask grounded questions from the Agentic AI eBook.")

index_name = get_env("PINECONE_INDEX_NAME", "agentic-ai-index")
api_key_present = bool(get_env("OPENAI_API_KEY", "").strip())
pinecone_key_present = bool(get_env("PINECONE_API_KEY", "").strip())

with st.sidebar:
    st.subheader("System status")
    st.info(f"Index: {index_name}")
    st.write(f"OpenAI key: {'Configured' if api_key_present else 'Missing'}")
    st.write(f"Pinecone key: {'Configured' if pinecone_key_present else 'Missing'}")
    st.caption("Add both keys in your .env file to enable live retrieval and generation.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can answer questions using the Agentic AI eBook. Ask me anything grounded in that document.",
            "context": [],
            "score": 0.0,
        }
    ]

for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(f"<div class={'user-bubble' if role == 'user' else 'assistant-bubble'}>" + message["content"] + "</div>", unsafe_allow_html=True)
        if message.get("context"):
            with st.expander("Retrieved context"):
                for chunk in message["context"]:
                    st.write(chunk)
        if message.get("score"):
            st.markdown(f'<div class="score-pill">Confidence: {message["score"]}</div>', unsafe_allow_html=True)

question = st.chat_input("Ask a question about the Agentic AI eBook")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "context": [], "score": 0.0})
    with st.chat_message("user"):
        st.markdown(f"<div class='user-bubble'>{question}</div>", unsafe_allow_html=True)

    try:
        graph = build_rag_graph(index_name=index_name)
        initial_state = {"question": question, "context": [], "answer": "", "score": 0.0}
        result = graph.invoke(initial_state)
        response = result.get("answer", "I cannot answer based on the provided document.")
        score = float(result.get("score", 0.0))
        context = result.get("context", [])

        st.session_state.messages.append({"role": "assistant", "content": response, "context": context, "score": score})

        with st.chat_message("assistant"):
            st.markdown(f"<div class='assistant-bubble'>{response}</div>", unsafe_allow_html=True)
            if context:
                with st.expander("Retrieved context"):
                    for chunk in context:
                        st.write(chunk)
            st.markdown(f'<div class="score-pill">Confidence: {score}</div>', unsafe_allow_html=True)
    except Exception as exc:
        error_msg = f"The project is not configured yet: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg, "context": [], "score": 0.0})
        with st.chat_message("assistant"):
            st.error(error_msg)
