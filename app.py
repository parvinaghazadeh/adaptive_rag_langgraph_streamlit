from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from adaptive_rag import AdaptiveRAGPipeline, PipelineConfig

load_dotenv()

ROOT_DIR = Path(__file__).parent
SAMPLE_KB_PATH = ROOT_DIR / "data" / "sample_knowledge_base.txt"
REFERENCE_IMAGE_PATH = ROOT_DIR / "assets" / "Adaptive_RAG_Pipeline_Architecture.png"


st.set_page_config(
    page_title="Adaptive RAG Pipeline",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def build_pipeline(
    knowledge_text: str,
    api_key: str,
    base_url: str,
    chat_model: str,
    embedding_model: str,
    embedding_backend: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    min_docs: int,
    threshold: float,
) -> AdaptiveRAGPipeline:
    config = PipelineConfig(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
        embedding_backend=embedding_backend,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        min_docs=min_docs,
        quality_threshold=threshold,
    )
    return AdaptiveRAGPipeline.from_text(knowledge_text, config)


def load_sample_text() -> str:
    return SAMPLE_KB_PATH.read_text(encoding="utf-8")


def render_doc_card(doc: dict, index: int) -> None:
    chunk_id = doc.get("metadata", {}).get("chunk_id", index)
    score = doc.get("score", 0)
    raw_score = doc.get("raw_score", 0)
    source_query = doc.get("query", "")
    with st.expander(f"Chunk {chunk_id} · similarity={score} · raw_distance={raw_score}"):
        st.caption(f"Retrieved by query: {source_query}")
        st.write(doc.get("content", ""))


st.title("Adaptive RAG Pipeline")
st.caption(
    "A Streamlit implementation of query analysis, query expansion, query decomposition, "
    "retrieval evaluation, HyDE fallback, and final RAG answer generation."
)

with st.sidebar:
    st.header("Runtime Settings")

    api_key = st.text_input(
        "API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Use a GapGPT/OpenAI-compatible API key for LLM routing and generation.",
    )
    base_url = st.text_input(
        "Base URL",
        value=os.getenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1"),
    )
    chat_model = st.text_input("Chat model", value=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    embedding_model = st.text_input(
        "Embedding model", value=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    embedding_backend_label = st.radio(
        "Embedding backend",
        options=[
            "Local hash embeddings (offline demo)",
            "OpenAI-compatible embeddings (GapGPT/OpenAI)",
        ],
        index=0,
        help="Use local hash embeddings to run without an API key. Use OpenAI-compatible embeddings for a closer match to the notebooks.",
    )
    embedding_backend = (
        "openai_compatible"
        if embedding_backend_label.startswith("OpenAI-compatible")
        else "local_hash"
    )

    st.divider()
    st.subheader("Retrieval Evaluation")
    top_k = st.slider("Top-k chunks", min_value=1, max_value=10, value=5)
    min_docs = st.slider("Minimum retrieved chunks", min_value=1, max_value=5, value=2)
    threshold = st.slider(
        "Average similarity threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    st.divider()
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", min_value=200, max_value=1200, value=550, step=50)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=250, value=80, step=10)

    st.divider()
    show_reference_architecture = st.checkbox("Show reference architecture", value=False)

if show_reference_architecture and REFERENCE_IMAGE_PATH.exists():
    st.image(str(REFERENCE_IMAGE_PATH), caption="Reference architecture provided in the assignment")

with st.expander("Knowledge base", expanded=False):
    uploaded_file = st.file_uploader("Upload a .txt knowledge base", type=["txt"])
    default_text = load_sample_text()
    if uploaded_file is not None:
        knowledge_text = uploaded_file.read().decode("utf-8")
        st.success("Uploaded knowledge base loaded.")
    else:
        knowledge_text = st.text_area(
            "Sample knowledge base",
            value=default_text,
            height=260,
            help="The demo ships with around 30 small sections. You can replace it with your own text.",
        )

question = st.text_input(
    "Ask a question",
    value="How does LangChain use memory and agents compared to CrewAI?",
    placeholder="Example: LangChain memory",
)

run_clicked = st.button("Run Adaptive RAG", type="primary")

if run_clicked:
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    if embedding_backend == "openai_compatible" and not api_key.strip():
        st.error("OpenAI-compatible embeddings require an API key. Use local hash embeddings or provide an API key.")
        st.stop()

    try:
        with st.spinner("Building LangGraph RAG pipeline and running the graph..."):
            pipeline = build_pipeline(
                knowledge_text=knowledge_text,
                api_key=api_key.strip(),
                base_url=base_url.strip(),
                chat_model=chat_model.strip(),
                embedding_model=embedding_model.strip(),
                embedding_backend=embedding_backend,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                min_docs=min_docs,
                threshold=threshold,
            )
            result = pipeline.run(question.strip())
    except Exception as exc:
        st.exception(exc)
        st.stop()

    route = result.get("route", "DIRECT")
    used_hyde = result.get("used_hyde", False)
    quality = result.get("retrieval_quality", {})

    st.subheader("Decision Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected method", route)
    col2.metric("HyDE fallback", "Yes" if used_hyde else "No")
    col3.metric("Retrieved chunks", quality.get("num_docs", 0))
    col4.metric("Average similarity", quality.get("avg_score", 0))

    st.subheader("Final Answer")
    st.write(result.get("final_answer", ""))

    st.subheader("Adaptive Pipeline Details")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**1. Initial Query Analysis**")
        st.json(result.get("analysis", {}))

        st.markdown("**2. Query Enhancement Path**")
        if route == "EXPANSION":
            st.info("The query was treated as simple/short, so Query Expansion was used.")
            st.code(result.get("expanded_query", ""), language="text")
        elif route == "DECOMPOSITION":
            st.info("The query was treated as complex/multi-part, so Query Decomposition was used.")
            for i, sub_q in enumerate(result.get("sub_questions", []), start=1):
                st.write(f"{i}. {sub_q}")
        else:
            st.info("The query was already well-formed, so it went directly to retrieval.")
            st.code(question.strip(), language="text")

        if used_hyde:
            st.markdown("**3. HyDE Fallback**")
            st.warning("Initial retrieval did not pass the quality threshold, so HyDE was used.")
            st.code(result.get("hyde_document", ""), language="text")

    with right:
        st.markdown("**Retrieval Quality Evaluation**")
        st.json(quality)

        st.markdown("**Execution Trace**")
        for step in result.get("trace", []):
            st.write(f"- {step}")

    st.subheader("Retrieved Chunks")
    for i, doc in enumerate(result.get("retrieved_docs", []), start=1):
        render_doc_card(doc, i)

    with st.expander("LangGraph Mermaid diagram", expanded=False):
        st.code(pipeline.mermaid(), language="mermaid")
else:
    st.info("Enter a question and click **Run Adaptive RAG**.")
    if not api_key.strip():
        st.caption(
            "No API key detected. The app can still run with local hash embeddings and heuristic fallbacks, "
            "but LLM-based expansion, decomposition, HyDE, and answer generation need an API key."
        )
