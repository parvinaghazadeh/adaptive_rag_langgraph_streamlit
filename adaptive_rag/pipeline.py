from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from .config import PipelineConfig
from .embeddings import GapGPTEmbeddings, LocalHashEmbeddings
from .prompts import (
    ANALYSIS_PROMPT,
    ANSWER_PROMPT,
    DECOMPOSITION_PROMPT,
    HYDE_PROMPT,
    QUERY_EXPANSION_PROMPT,
)

Route = Literal["EXPANSION", "DECOMPOSITION", "DIRECT"]


class GraphState(TypedDict, total=False):
    question: str
    route: Route
    analysis: Dict[str, Any]
    expanded_query: str
    sub_questions: List[str]
    retrieval_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    retrieval_quality: Dict[str, Any]
    used_hyde: bool
    hyde_document: str
    final_answer: str
    trace: List[str]


def _append_trace(state: GraphState, message: str) -> List[str]:
    return [*state.get("trace", []), message]


def _clean_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
        if line and line not in lines:
            lines.append(line)
    return lines


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


class AdaptiveRAGPipeline:
    """
    Adaptive RAG implementation using LangChain components inside a LangGraph workflow.

    Graph flow:
    analyze_query -> expansion/decomposition/direct -> retrieve -> evaluate
    if retrieval fails -> hyde -> retrieve_with_hyde -> generate
    if retrieval passes -> generate
    """

    def __init__(self, documents: List[Document], config: PipelineConfig):
        self.config = config
        self.documents = documents
        self.llm = self._build_llm(config)
        self.vectorstore = self._build_vectorstore(documents, config)
        self.graph = self._build_graph()

        self.analysis_chain = None
        self.expansion_chain = None
        self.decomposition_chain = None
        self.hyde_chain = None
        self.answer_chain = None

        if self.llm is not None:
            parser = StrOutputParser()
            self.analysis_chain = ANALYSIS_PROMPT | self.llm | parser
            self.expansion_chain = QUERY_EXPANSION_PROMPT | self.llm | parser
            self.decomposition_chain = DECOMPOSITION_PROMPT | self.llm | parser
            self.hyde_chain = HYDE_PROMPT | self.llm | parser
            self.answer_chain = ANSWER_PROMPT | self.llm | parser

    @classmethod
    def from_text(cls, text: str, config: PipelineConfig) -> "AdaptiveRAGPipeline":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n---\n", "\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.create_documents([text])
        for idx, doc in enumerate(docs, start=1):
            doc.metadata["chunk_id"] = idx
            doc.metadata["source"] = "knowledge_base"
        return cls(docs, config)

    @staticmethod
    def _build_llm(config: PipelineConfig):
        if not config.api_key.strip():
            return None
        return ChatOpenAI(
            model=config.chat_model,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    @staticmethod
    def _build_vectorstore(documents: List[Document], config: PipelineConfig) -> FAISS:
        if config.embedding_backend == "openai_compatible":
            if not config.api_key.strip():
                raise ValueError("OpenAI-compatible embeddings require an API key.")
            embeddings = GapGPTEmbeddings(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.embedding_model,
            )
        else:
            embeddings = LocalHashEmbeddings()
        return FAISS.from_documents(documents, embeddings)

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("expand_query", self._expand_query)
        workflow.add_node("decompose_query", self._decompose_query)
        workflow.add_node("direct_query", self._direct_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("evaluate_retrieval", self._evaluate_retrieval)
        workflow.add_node("hyde", self._hyde)
        workflow.add_node("retrieve_with_hyde", self._retrieve_with_hyde)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.set_entry_point("analyze_query")
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_after_analysis,
            {
                "EXPANSION": "expand_query",
                "DECOMPOSITION": "decompose_query",
                "DIRECT": "direct_query",
            },
        )
        workflow.add_edge("expand_query", "retrieve")
        workflow.add_edge("decompose_query", "retrieve")
        workflow.add_edge("direct_query", "retrieve")
        workflow.add_edge("retrieve", "evaluate_retrieval")
        workflow.add_conditional_edges(
            "evaluate_retrieval",
            self._route_after_evaluation,
            {
                "PASS": "generate_answer",
                "FAIL": "hyde",
            },
        )
        workflow.add_edge("hyde", "retrieve_with_hyde")
        workflow.add_edge("retrieve_with_hyde", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def run(self, question: str) -> GraphState:
        initial_state: GraphState = {
            "question": question,
            "used_hyde": False,
            "trace": ["User query received."],
        }
        return self.graph.invoke(initial_state)

    def mermaid(self) -> str:
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return "graph TD; A[analyze_query] --> B{route}; B --> C[expand_query]; B --> D[decompose_query]; B --> E[direct_query]; C --> F[retrieve]; D --> F; E --> F; F --> G[evaluate_retrieval]; G -->|pass| H[generate_answer]; G -->|fail| I[hyde]; I --> J[retrieve_with_hyde]; J --> H;"

    # ----------------------------- graph nodes -----------------------------

    def _analyze_query(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"].strip()
        analysis = self._heuristic_analysis(question)

        if self.analysis_chain is not None:
            try:
                raw = self.analysis_chain.invoke({"question": question})
                parsed = _parse_json_object(raw)
                if parsed and parsed.get("route") in {"EXPANSION", "DECOMPOSITION", "DIRECT"}:
                    analysis = parsed
            except Exception as exc:
                analysis["llm_router_error"] = str(exc)

        route = analysis.get("route", "DIRECT")
        return {
            "route": route,
            "analysis": analysis,
            "trace": _append_trace(state, f"Initial Query Analysis selected route: {route}."),
        }

    def _expand_query(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        expanded = f"{question} related concepts definitions examples technical terms context"

        if self.expansion_chain is not None:
            try:
                expanded = self.expansion_chain.invoke({"query": question}).strip()
            except Exception as exc:
                expanded = f"{expanded} expansion_error:{exc}"

        return {
            "expanded_query": expanded,
            "retrieval_queries": [expanded],
            "trace": _append_trace(state, "Query Expansion node created an expanded retrieval query."),
        }

    def _decompose_query(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        sub_questions = self._fallback_decomposition(question)

        if self.decomposition_chain is not None:
            try:
                raw = self.decomposition_chain.invoke({"question": question})
                parsed = _clean_lines(raw)
                if parsed:
                    sub_questions = parsed[:4]
            except Exception:
                pass

        if not sub_questions:
            sub_questions = [question]

        return {
            "sub_questions": sub_questions,
            "retrieval_queries": sub_questions,
            "trace": _append_trace(state, f"Query Decomposition node produced {len(sub_questions)} sub-question(s)."),
        }

    def _direct_query(self, state: GraphState) -> Dict[str, Any]:
        return {
            "retrieval_queries": [state["question"]],
            "trace": _append_trace(state, "Direct path selected because the query was already well-formed."),
        }

    def _retrieve(self, state: GraphState) -> Dict[str, Any]:
        docs = self._search_many(state.get("retrieval_queries", [state["question"]]))
        return {
            "retrieved_docs": docs,
            "trace": _append_trace(state, f"Retriever returned {len(docs)} unique chunk(s)."),
        }

    def _evaluate_retrieval(self, state: GraphState) -> Dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        avg_score = sum(float(d["score"]) for d in docs) / len(docs) if docs else 0.0
        passed = len(docs) >= self.config.min_docs and avg_score >= self.config.quality_threshold

        quality = {
            "passed": passed,
            "num_docs": len(docs),
            "avg_score": round(avg_score, 4),
            "min_docs": self.config.min_docs,
            "threshold": self.config.quality_threshold,
            "criteria": "num_docs >= min_docs AND avg_score >= threshold",
        }
        status = "passed" if passed else "failed"
        return {
            "retrieval_quality": quality,
            "trace": _append_trace(state, f"Retrieval evaluation {status}: {quality}."),
        }

    def _hyde(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        hyde_document = f"A detailed answer to '{question}' would discuss the key definitions, mechanisms, comparisons, examples, and relevant implementation details."

        if self.hyde_chain is not None:
            try:
                hyde_document = self.hyde_chain.invoke({"query": question}).strip()
            except Exception:
                pass

        return {
            "used_hyde": True,
            "hyde_document": hyde_document,
            "trace": _append_trace(state, "HyDE node generated a hypothetical document because initial retrieval quality was below threshold."),
        }

    def _retrieve_with_hyde(self, state: GraphState) -> Dict[str, Any]:
        hyde_document = state.get("hyde_document", state["question"])
        docs = self._search_many([hyde_document])
        avg_score = sum(float(d["score"]) for d in docs) / len(docs) if docs else 0.0
        quality = {
            "passed": len(docs) >= 1,
            "num_docs": len(docs),
            "avg_score": round(avg_score, 4),
            "min_docs": self.config.min_docs,
            "threshold": self.config.quality_threshold,
            "criteria": "HyDE retry result; final generation uses best available context",
        }
        return {
            "retrieved_docs": docs,
            "retrieval_quality": quality,
            "trace": _append_trace(state, f"Re-retrieved with HyDE embeddings and returned {len(docs)} chunk(s)."),
        }

    def _generate_answer(self, state: GraphState) -> Dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        context = self._format_context(docs)
        retrieval_queries = "\n".join(f"- {q}" for q in state.get("retrieval_queries", []))

        if self.answer_chain is not None:
            try:
                final_answer = self.answer_chain.invoke(
                    {
                        "route": state.get("route", "DIRECT"),
                        "used_hyde": str(state.get("used_hyde", False)),
                        "question": state["question"],
                        "retrieval_queries": retrieval_queries,
                        "context": context,
                    }
                ).strip()
            except Exception as exc:
                final_answer = self._fallback_answer(state, exc)
        else:
            final_answer = self._fallback_answer(state)

        return {
            "final_answer": final_answer,
            "trace": _append_trace(state, "Final answer generated from retrieved context."),
        }

    # ----------------------------- helpers -----------------------------

    def _route_after_analysis(self, state: GraphState) -> str:
        return state.get("route", "DIRECT")

    def _route_after_evaluation(self, state: GraphState) -> str:
        quality = state.get("retrieval_quality", {})
        return "PASS" if quality.get("passed") else "FAIL"

    def _heuristic_analysis(self, question: str) -> Dict[str, Any]:
        lower_q = question.lower()
        words = re.findall(r"[\w\u0600-\u06FF]+", question)
        multi_part_markers = [
            " and ",
            " vs ",
            " versus ",
            " compare",
            " compared",
            "difference",
            "differences",
            "how does",
            "چطور",
            "مقایسه",
            "فرق",
            "تفاوت",
            " و ",
        ]
        has_multiple_marks = question.count("?") > 1
        has_marker = any(marker in lower_q for marker in multi_part_markers)

        if has_multiple_marks or (has_marker and len(words) > 8):
            return {
                "route": "DECOMPOSITION",
                "reason": "The query appears to contain multiple intents or a comparison.",
                "signals": ["multi-part marker", f"word_count={len(words)}"],
            }
        if len(words) <= self.config.short_query_word_limit:
            return {
                "route": "EXPANSION",
                "reason": "The query is short or underspecified, so expansion can add useful retrieval terms.",
                "signals": ["short query", f"word_count={len(words)}"],
            }
        return {
            "route": "DIRECT",
            "reason": "The query is clear enough for direct retrieval.",
            "signals": ["single intent", f"word_count={len(words)}"],
        }

    def _fallback_decomposition(self, question: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", question).strip()
        parts = re.split(r"\?|\band\b|\bvs\b|\bversus\b|\bcompared with\b|\bcompared to\b| و ", normalized, flags=re.IGNORECASE)
        cleaned = [p.strip(" ,.;:") for p in parts if p.strip(" ,.;:")]
        if len(cleaned) >= 2:
            return [p if p.endswith("?") else f"{p}?" for p in cleaned[:4]]
        return [question]

    def _search_many(self, queries: List[str]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        all_docs: List[Dict[str, Any]] = []

        for query in queries:
            results: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(query, k=self.config.top_k)
            for doc, raw_score in results:
                key = doc.page_content.strip()
                if key in seen:
                    continue
                seen.add(key)
                score = self._distance_to_similarity(float(raw_score))
                all_docs.append(
                    {
                        "content": doc.page_content,
                        "metadata": dict(doc.metadata),
                        "raw_score": round(float(raw_score), 4),
                        "score": round(score, 4),
                        "query": query,
                    }
                )

        all_docs.sort(key=lambda item: item["score"], reverse=True)
        return all_docs[: self.config.top_k]

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        # FAISS returns a distance where lower is better. This maps it to 0..1-ish.
        return 1.0 / (1.0 + max(distance, 0.0))

    @staticmethod
    def _format_context(docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "No context retrieved."
        blocks = []
        for idx, doc in enumerate(docs, start=1):
            chunk_id = doc.get("metadata", {}).get("chunk_id", idx)
            score = doc.get("score", 0)
            blocks.append(f"[Chunk {chunk_id} | score={score}]\n{doc['content']}")
        return "\n\n".join(blocks)

    def _fallback_answer(self, state: GraphState, error: Optional[Exception] = None) -> str:
        docs = state.get("retrieved_docs", [])
        route = state.get("route", "DIRECT")
        used_hyde = state.get("used_hyde", False)
        if not docs:
            return "The knowledge base does not contain enough information to answer this question."

        top_context = "\n\n".join(f"- {doc['content']}" for doc in docs[:3])
        error_note = f"\n\nLLM generation fallback was used because: {error}" if error else ""
        return (
            f"Method used: {route}. HyDE fallback used: {used_hyde}.\n\n"
            f"Based on the most relevant retrieved chunks:\n{top_context}"
            f"{error_note}"
        )
