from langchain_core.prompts import PromptTemplate


ANALYSIS_PROMPT = PromptTemplate.from_template(
    """
You are the query-router node of an Adaptive RAG system.
Classify the user's question into exactly one route:

- EXPANSION: short, simple, vague, or underspecified question that needs more retrieval terms.
- DECOMPOSITION: complex, multi-part, comparative, or multi-intent question that should be broken into sub-questions.
- DIRECT: clear, well-formed, single-intent question that can go directly to retrieval.

Return ONLY valid JSON with these keys:
{{
  "route": "EXPANSION" | "DECOMPOSITION" | "DIRECT",
  "reason": "brief explanation",
  "signals": ["signal 1", "signal 2"]
}}

Question: {question}
""".strip()
)


QUERY_EXPANSION_PROMPT = PromptTemplate.from_template(
    """
You are a helpful assistant. Expand the following query to improve document retrieval by adding relevant synonyms, technical terms, and useful context.
Keep the expanded query concise and retrieval-focused.

Original query: "{query}"

Expanded query:
""".strip()
)


DECOMPOSITION_PROMPT = PromptTemplate.from_template(
    """
You are an AI assistant. Decompose the following complex question into 2 to 4 smaller sub-questions for better document retrieval.
Return each sub-question on a separate line. Do not add explanations.

Question: "{question}"

Sub-questions:
""".strip()
)


HYDE_PROMPT = PromptTemplate.from_template(
    """
Imagine you are an expert writing a detailed explanation on the topic: "{query}".
Create a concise hypothetical answer that would likely contain the information needed to answer the question.
Do not mention that it is hypothetical.

Hypothetical document:
""".strip()
)


ANSWER_PROMPT = PromptTemplate.from_template(
    """
You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.
If the answer is not present in the context, say that the knowledge base does not contain enough information.

Routing method used: {route}
HyDE fallback used: {used_hyde}

Original question:
{question}

Enhanced retrieval queries:
{retrieval_queries}

Context:
{context}

Final answer:
""".strip()
)
