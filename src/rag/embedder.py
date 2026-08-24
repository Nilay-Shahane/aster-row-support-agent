"""
embedder.py

Thin wrapper around OpenAI's embedding endpoint (text-embedding-3-small).
This is the ONLY place in the codebase that talks to the embedding API —
everything else (chunking, vector store, retrieval) works with plain
lists of floats and doesn't know or care which provider produced them.
Swapping embedding models/providers later means changing this file only.
"""
from dotenv import load_dotenv

load_dotenv()

import os
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # native dimensionality of text-embedding-3-small
BATCH_SIZE = 100  # OpenAI accepts many inputs per call; batch to cut round trips


class Embedder:
    def __init__(self, api_key: str | None = None, model: str = EMBEDDING_MODEL):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of texts in batches, preserving input order. Used at
        ingest time to embed every chunk's `embedding_text` in one pass.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            # response.data preserves input order, so a plain extend is safe
            all_vectors.extend(item.embedding for item in response.data)

        return all_vectors

    def embed_query(self, query_text: str) -> list[float]:
        """
        Embeds a single query string at retrieval time. Kept separate from
        embed_texts so callers don't have to unwrap a single-element list
        every time they embed a user question.
        """
        return self.embed_texts([query_text])[0]