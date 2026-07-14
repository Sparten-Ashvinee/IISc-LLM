"""FastAPI serving layer for TalentRAG."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="TalentRAG API", version="1.0.0")

# Global pipeline reference — initialized on startup
_pipeline = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    enable_guard: bool = True


class SourceInfo(BaseModel):
    chunk_id: str
    title: str
    company: str
    location: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    is_faithful: bool
    faithfulness_score: float
    latency_ms: float


@app.on_event("startup")
async def startup():
    """Load all models and indexes on startup."""
    global _pipeline
    # Import here to avoid circular imports
    from pipeline import TalentRAGPipeline
    _pipeline = TalentRAGPipeline.from_config("configs/config.yaml")
    print("Pipeline loaded and ready.")


@app.get("/health")
async def health():
    cache_status = "disabled"
    if _pipeline and _pipeline.cache:
        try:
            stats = _pipeline.cache.stats()
            cache_status = f"connected ({stats['cached_queries']} cached)"
        except Exception:
            cache_status = "error"
    return {"status": "ok", "pipeline_loaded": _pipeline is not None, "redis": cache_status}


@app.post("/cache/invalidate")
async def invalidate_cache():
    if _pipeline and _pipeline.cache:
        count = _pipeline.cache.invalidate_all()
        return {"cleared": count}
    return {"cleared": 0, "message": "Redis not enabled"}


@app.get("/cache/stats")
async def cache_stats():
    if _pipeline and _pipeline.cache:
        return _pipeline.cache.stats()
    return {"cached_queries": 0, "hits": 0, "misses": 0, "message": "Redis not enabled"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    import time
    start = time.perf_counter()

    result = _pipeline.run(req.query, top_k=req.top_k, enable_guard=req.enable_guard)

    latency_ms = (time.perf_counter() - start) * 1000

    sources = []
    for chunk in result.get("chunks", []):
        meta = chunk.get("metadata", {})
        sources.append(SourceInfo(
            chunk_id=chunk["chunk_id"],
            title=meta.get("title", ""),
            company=meta.get("company_name", ""),
            location=meta.get("location", ""),
            score=chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
        ))

    guard = result.get("guard", {})
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        is_faithful=guard.get("is_faithful", True),
        faithfulness_score=guard.get("score", 0.0),
        latency_ms=round(latency_ms, 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=True)
