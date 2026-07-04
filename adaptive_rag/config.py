from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Runtime configuration for the Adaptive RAG graph."""

    api_key: str = ""
    base_url: str = "https://api.gapgpt.app/v1"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_backend: str = "local_hash"  # "local_hash" or "openai_compatible"
    temperature: float = 0.2

    chunk_size: int = 550
    chunk_overlap: int = 80

    top_k: int = 5
    min_docs: int = 2
    quality_threshold: float = 0.25

    # Query analysis heuristics used as a fallback when no LLM is configured.
    short_query_word_limit: int = 6
