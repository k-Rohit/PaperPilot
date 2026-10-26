"""Pinecone vector store: hybrid (dense + sparse) storage and search for paper chunks.

Dense vectors come from the caller (our OpenAI embedder) so chunks and queries share one model.
Sparse (keyword) vectors are generated here by Pinecone's hosted sparse model.
"""

import math
from functools import cached_property

from pinecone import Pinecone, ServerlessSpec

from src.config import PineconeSettings


def normalize(values: list[float]) -> list[float]:
    """Scale a vector to length 1."""
    norm = math.sqrt(sum(v * v for v in values)) or 1.0  # `or 1.0` avoids dividing by zero
    return [v / norm for v in values]


class PineconeClient:
    def __init__(self, settings: PineconeSettings):
        if not settings.api_key:
            raise ValueError("PINECONE_API_KEY is not set — add it to .env")
        self.pc = Pinecone(api_key=settings.api_key)
        self.index_name = settings.index_name
        self.sparse_model = settings.sparse_model
        self.embed_batch_size = settings.embed_batch_size
        self.dimension = settings.dimension
        self.cloud = settings.cloud
        self.region = settings.region

    def setup_index(self) -> None:
        """Create the index once. Hybrid (sparse + dense) indexes must use dotproduct."""
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="dotproduct",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

    @cached_property
    def index(self):
        return self.pc.Index(self.index_name)

    def create_sparse_embeddings(self, texts: list[str], input_type: str = "passage") -> list:
        """Keyword vectors. input_type is "passage" for chunks, "query" for questions."""
        embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            embeddings.extend(
                self.pc.inference.embed(
                    model=self.sparse_model,
                    inputs=texts[i : i + self.embed_batch_size],
                    parameters={"input_type": input_type, "truncate": "END"},
                )
            )
        return embeddings

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Store chunks with their dense vectors (from the caller) and sparse vectors (made here)."""
        sparse_embeddings = self.create_sparse_embeddings([c["chunk_text"] for c in chunks])

        records = []
        for chunk, dense, sparse in zip(chunks, embeddings, sparse_embeddings, strict=True):
            records.append(
                {
                    "id": chunk["chunk_id"],
                    "values": dense,
                    "sparse_values": {"indices": sparse["sparse_indices"], "values": normalize(sparse["sparse_values"])},
                    # Pinecone rejects None in metadata, hence `or ""`
                    "metadata": {
                        "arxiv_id": chunk["arxiv_id"],
                        "chunk_text": chunk["chunk_text"],
                        "title": chunk.get("title") or "",
                        "pdf_url": chunk.get("pdf_url") or "",
                        "authors": chunk.get("authors") or "",
                        "section_title": chunk.get("section_title") or "",
                    },
                }
            )

        self.index.upsert(vectors=records)

    def delete_paper_chunks(self, arxiv_id: str) -> None:
        """Delete every chunk of one paper (used before re-indexing it)."""
        self.index.delete(filter={"arxiv_id": {"$eq": arxiv_id}})

    def get_chunks_by_paper(self, arxiv_id: str) -> list[dict]:
        """All chunks of one paper in reading order (relies on chunk ids being "{arxiv_id}#{chunk_index}")."""
        ids = [v.id for page in self.index.list(prefix=f"{arxiv_id}#") for v in page.vectors]
        if not ids:
            return []
        vectors = self.index.fetch(ids=ids).vectors
        ids.sort(key=lambda i: int(i.rsplit("#", 1)[1]))
        return [{**vectors[i].metadata, "chunk_id": i} for i in ids if i in vectors]

    def search(self, query: str, query_embedding: list[float], top_k: int, alpha: float = 0.6) -> list[dict]:
        """Hybrid search. alpha=1.0 → pure meaning (dense), alpha=0.0 → pure keywords (sparse).
        combined = alpha * dense + (1 - alpha) * sparse"""
        sparse = self.create_sparse_embeddings([query], input_type="query")[0]

        result = self.index.query(
            vector=[v * alpha for v in query_embedding],
            sparse_vector={
                "indices": sparse["sparse_indices"],
                "values": [v * (1 - alpha) for v in normalize(sparse["sparse_values"])],
            },
            top_k=top_k,
            include_metadata=True,
        )
        return [{**match.metadata, "score": match.score} for match in result.matches]
