"""Query planner — decomposes complex queries and extracts metadata filters."""

import json
import re


# Rule-based filter extraction (no LLM needed for common patterns)
EXPERIENCE_PATTERNS = {
    r"(\d+)\+?\s*(?:years?|yrs?)": "min_experience",
    r"(?:entry|junior)": "entry",
    r"(?:senior|sr\.?)": "senior",
    r"(?:lead|principal|staff)": "lead",
    r"(?:director|vp|head)": "director",
}

LOCATION_PATTERNS = [
    r"(?:in|at|from|based in|located in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z]{2})?)",
    r"\b(remote|hybrid|on-?site)\b",
]


def extract_filters(query: str) -> dict:
    """Extract structured filters from a natural language query."""
    filters = {}

    # Experience level
    for pattern, label in EXPERIENCE_PATTERNS.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            if label == "min_experience":
                filters["min_experience"] = int(match.group(1))
            else:
                filters["experience_level"] = label

    # Location
    for pattern in LOCATION_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            filters["location"] = match.group(1) if match.lastindex else match.group(0)

    return filters


def decompose_query(query: str) -> list[str]:
    """Split a complex query into simpler sub-queries.

    Uses rule-based decomposition on 'and'/'who'/'with' boundaries.
    """
    # Split on conjunctions that introduce new constraints
    parts = re.split(r"\s+(?:and|who|that|with experience in|having)\s+", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if len(p.strip()) > 10]

    if len(parts) <= 1:
        return [query]

    return parts


class QueryPlanner:
    def __init__(self):
        pass

    def plan(self, query: str) -> dict:
        """Analyze a query and return a plan with sub-queries and filters."""
        filters = extract_filters(query)
        sub_queries = decompose_query(query)

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "filters": filters,
        }


if __name__ == "__main__":
    planner = QueryPlanner()

    test_queries = [
        "Find senior ML engineers in San Francisco with 5+ years experience",
        "Remote Python developer roles at startups",
        "Data scientist who published at NeurIPS and knows PyTorch",
    ]

    for q in test_queries:
        plan = planner.plan(q)
        print(f"\nQuery: {q}")
        print(f"  Sub-queries: {plan['sub_queries']}")
        print(f"  Filters: {plan['filters']}")
