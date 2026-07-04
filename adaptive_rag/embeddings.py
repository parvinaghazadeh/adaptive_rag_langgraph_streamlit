from __future__ import annotations

import hashlib
import math
import re
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from openai import OpenAI


class GapGPTEmbeddings(Embeddings):
    """
    OpenAI-compatible embeddings wrapper.

    This follows the same idea used in the course notebooks: the OpenAI SDK is
    pointed at an OpenAI-compatible base_url such as GapGPT.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.gapgpt.app/v1",
        model: str = "text-embedding-3-small",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            res = self.client.embeddings.create(model=self.model, input=text)
            embeddings.append(res.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        res = self.client.embeddings.create(model=self.model, input=text)
        return res.data[0].embedding


class LocalHashEmbeddings(Embeddings):
    """
    Tiny local embedding fallback for demos without an API key.

    It hashes tokens into a fixed-size vector and L2-normalizes the vector.
    This is not meant to replace real semantic embeddings, but it keeps the
    Streamlit project runnable offline while still using LangChain + FAISS.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._token_pattern = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

    def _embed(self, text: str) -> List[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = self._token_pattern.findall(text.lower())

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm > 0:
            vector = vector / norm
        return vector.astype(float).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
