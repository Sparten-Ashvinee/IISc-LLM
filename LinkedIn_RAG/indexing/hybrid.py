"""Reciprocal Rank Fusion (RRF) for merging dense + sparse results."""


def reciprocal_rank_fusion(
    results_list: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF.

    Args:
        results_list: list of ranked results, each is [(chunk_id, score), ...]
        k: RRF constant (default 60, standard value from the paper)
        top_k: number of results to return

    Returns:
        Merged ranked list of (chunk_id, rrf_score)
    """
    rrf_scores: dict[str, float] = {}

    for results in results_list:
        for rank, (chunk_id, _score) in enumerate(results):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
