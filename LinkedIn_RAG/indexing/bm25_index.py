"""BM25 sparse index for keyword-based retrieval."""

import os
import pickle
from rank_bm25 import BM25Okapi


class SparseIndex:
    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []
        self.tokenized_corpus = []

    def build_index(self, chunks: list) -> None:
        """Build BM25 index from chunks."""
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 index built: {len(self.chunk_ids):,} documents")

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) for a query."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        results = [
            (self.chunk_ids[i], float(scores[i]))
            for i in top_indices if scores[i] > 0
        ]
        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "chunk_ids": self.chunk_ids,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved BM25 index to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"Loaded BM25 index: {len(self.chunk_ids):,} documents")
