"""Retrieval evaluation — Recall@k, MRR@k, NDCG@k."""

import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """Fraction of relevant items found in top-k retrieved."""
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(retrieved_k & relevant) / len(relevant)


def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """Mean Reciprocal Rank — rank of the first relevant result."""
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain."""
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], 1)
        if rid in relevant
    )
    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_retrieval(queries: list[dict], retriever, k_values: list[int] = [5, 10, 20]) -> dict:
    """Evaluate retriever on a set of (query, relevant_ids) pairs.

    Args:
        queries: list of {"query": str, "relevant_ids": list[str]}
        retriever: object with .retrieve(query) returning list of dicts with "chunk_id"
        k_values: list of k values to evaluate

    Returns:
        dict of metric_name -> value
    """
    results = {}

    for k in k_values:
        recalls, mrrs, ndcgs = [], [], []

        for q in queries:
            retrieved = retriever.retrieve(q["query"])
            retrieved_ids = [r["chunk_id"] for r in retrieved]
            relevant_ids = q["relevant_ids"]

            recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
            mrrs.append(mrr_at_k(retrieved_ids, relevant_ids, k))
            ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k))

        results[f"Recall@{k}"] = round(np.mean(recalls), 4)
        results[f"MRR@{k}"] = round(np.mean(mrrs), 4)
        results[f"NDCG@{k}"] = round(np.mean(ndcgs), 4)

    return results
