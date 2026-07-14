"""Generation evaluation — faithfulness, answer relevancy, end-to-end metrics."""

import re
import time


def citation_coverage(answer: str, num_sources: int) -> float:
    """What fraction of provided sources were actually cited in the answer."""
    cited = set(int(m) for m in re.findall(r'\[Source (\d+)\]', answer))
    if num_sources == 0:
        return 0.0
    return len(cited) / num_sources


def answer_has_refusal(answer: str) -> bool:
    """Check if the model correctly refused when it lacked information."""
    refusal_patterns = [
        r"don.?t have enough information",
        r"cannot answer",
        r"no relevant .* found",
        r"not .* in the provided context",
    ]
    return any(re.search(p, answer, re.IGNORECASE) for p in refusal_patterns)


def measure_latency(func, *args, **kwargs):
    """Measure wall-clock latency of a function call."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def evaluate_generation(test_cases: list[dict], pipeline) -> dict:
    """Run end-to-end evaluation.

    Args:
        test_cases: list of {"query": str, "expected_keywords": list[str], "has_answer": bool}
        pipeline: callable(query) -> {"answer": str, "chunks": list, "guard": dict}

    Returns:
        Aggregated metrics dict
    """
    faithfulness_scores = []
    citation_scores = []
    latencies = []
    hallucination_flags = 0

    for tc in test_cases:
        result, latency = measure_latency(pipeline, tc["query"])
        latencies.append(latency)

        answer = result["answer"]
        guard = result.get("guard", {})
        num_chunks = len(result.get("chunks", []))

        # Faithfulness from hallucination guard
        faithfulness_scores.append(guard.get("score", 0.0))
        if not guard.get("is_faithful", True):
            hallucination_flags += 1

        # Citation coverage
        citation_scores.append(citation_coverage(answer, num_chunks))

    n = len(test_cases)
    latencies_sorted = sorted(latencies)

    return {
        "avg_faithfulness": round(sum(faithfulness_scores) / n, 4) if n else 0,
        "avg_citation_coverage": round(sum(citation_scores) / n, 4) if n else 0,
        "hallucination_rate": round(hallucination_flags / n, 4) if n else 0,
        "latency_p50": round(latencies_sorted[n // 2], 3) if n else 0,
        "latency_p95": round(latencies_sorted[int(n * 0.95)], 3) if n else 0,
        "num_evaluated": n,
    }
