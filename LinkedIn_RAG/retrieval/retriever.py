"""Hybrid retriever — orchestrates dense, sparse, and reranking."""

from indexing.embed import DenseIndex
from indexing.bm25_index import SparseIndex
from indexing.hybrid import reciprocal_rank_fusion
from retrieval.reranker import Reranker


class HybridRetriever:
    def __init__(self, dense_index: DenseIndex, sparse_index: SparseIndex,
                 reranker: Reranker, chunk_store: dict,
                 dense_weight: float = 0.6, sparse_weight: float = 0.4,
                 top_k: int = 20, rerank_top_k: int = 5):
        self.dense = dense_index
        self.sparse = sparse_index
        self.reranker = reranker
        self.chunk_store = chunk_store  # chunk_id -> Chunk object
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query: str, filters: dict | None = None) -> list[dict]:
        """Full retrieval pipeline: dense + sparse → RRF → rerank → filter."""

        # Step 1: Dense search
        dense_results = self.dense.search(query, top_k=self.top_k)

        # Step 2: Sparse search
        sparse_results = self.sparse.search(query, top_k=self.top_k)

        # Step 3: RRF fusion
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=self.top_k,
        )

        # Step 4: Retrieve chunk texts for reranking
        candidates = []
        for chunk_id, rrf_score in fused:
            if chunk_id in self.chunk_store:
                chunk = self.chunk_store[chunk_id]
                candidates.append({
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                    "chunk_type": chunk.chunk_type,
                    "metadata": chunk.metadata,
                    "rrf_score": rrf_score,
                })

        # Step 5: Rerank
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=self.rerank_top_k)

        # Step 6: Apply metadata filters
        if filters:
            candidates = self._apply_filters(candidates, filters)

        return candidates

    def _apply_filters(self, candidates: list[dict], filters: dict) -> list[dict]:
        """Apply metadata filters to candidates."""
        filtered = []
        for c in candidates:
            meta = c.get("metadata", {})
            keep = True

            if "location" in filters:
                loc = meta.get("location", "").lower()
                if filters["location"].lower() not in loc:
                    keep = False

            if "experience_level" in filters:
                exp = meta.get("experience_level", "").lower()
                if filters["experience_level"].lower() not in exp:
                    keep = False

            if keep:
                filtered.append(c)

        return filtered if filtered else candidates  # fallback to unfiltered if all removed
