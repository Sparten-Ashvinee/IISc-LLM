"""TalentRAG — Main pipeline that wires all components together."""

import os
import yaml
from data.load_data import load_linkedin_data
from data.chunker import chunk_dataframe
from indexing.embed import DenseIndex
from indexing.bm25_index import SparseIndex
from retrieval.query_planner import QueryPlanner
from retrieval.reranker import Reranker
from retrieval.retriever import HybridRetriever
from generation.generator import Generator
from generation.hallucination_guard import HallucinationGuard


class TalentRAGPipeline:
    def __init__(self, retriever, generator, guard, query_planner, cache=None):
        self.retriever = retriever
        self.generator = generator
        self.guard = guard
        self.query_planner = query_planner
        self.cache = cache

    @classmethod
    def from_config(cls, config_path: str = "configs/config.yaml"):
        """Build entire pipeline from config file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Check if ChromaDB collection already exists
        chroma_dir = cfg["indexing"]["chroma_persist_dir"]
        collection_name = cfg["indexing"]["chroma_collection"]
        bm25_path = cfg["indexing"]["bm25_path"]

        dense_index = DenseIndex(
            model_name=cfg["embedding"]["model_name"],
            persist_dir=chroma_dir,
            collection_name=collection_name,
        )

        if os.path.exists(chroma_dir) and os.path.exists(bm25_path):
            print("Loading pre-built indexes...")
            dense_index.load()

            sparse_index = SparseIndex()
            sparse_index.load(bm25_path)

            # Load chunk store
            import pickle
            store_path = os.path.join(chroma_dir, "chunk_store.pkl")
            with open(store_path, "rb") as f:
                chunk_store = pickle.load(f)
        else:
            print("Building indexes from scratch...")
            dense_index, sparse_index, chunk_store = cls._build_indexes(cfg, dense_index)

        # Reranker
        reranker = Reranker(model_name=cfg["reranker"]["model_name"])

        # Retriever
        retriever = HybridRetriever(
            dense_index=dense_index,
            sparse_index=sparse_index,
            reranker=reranker,
            chunk_store=chunk_store,
            top_k=cfg["retrieval"]["top_k"],
            rerank_top_k=cfg["reranker"]["rerank_top_k"],
        )

        # Generator
        generator = Generator(
            model_name=cfg["generation"]["model_name"],
            quantization=cfg["generation"]["quantization"],
        )

        # Hallucination guard
        guard = None
        if cfg["hallucination_guard"]["enabled"]:
            guard = HallucinationGuard(
                model_name=cfg["hallucination_guard"]["nli_model"],
                entailment_threshold=cfg["hallucination_guard"]["entailment_threshold"],
            )

        query_planner = QueryPlanner()

        # Redis cache
        cache = None
        redis_cfg = cfg.get("redis", {})
        if redis_cfg.get("enabled", False):
            try:
                from serving.cache import RedisCache
                cache = RedisCache(
                    host=redis_cfg.get("host", "localhost"),
                    port=redis_cfg.get("port", 6379),
                    db=redis_cfg.get("db", 0),
                    ttl=redis_cfg.get("ttl", 3600),
                )
            except ConnectionError as e:
                print(f"Redis not available, running without cache: {e}")

        return cls(retriever, generator, guard, query_planner, cache=cache)

    @staticmethod
    def _build_indexes(cfg: dict, dense_index: DenseIndex):
        """Load data, chunk, and build all indexes."""
        # Load data
        df = load_linkedin_data(
            dataset_name=cfg["data"]["dataset_name"],
            max_records=cfg["data"]["max_records"],
            cache_dir=cfg["data"]["cache_dir"],
        )

        # Chunk
        chunks = chunk_dataframe(df)

        # Build chunk store (chunk_id -> Chunk)
        chunk_store = {c.chunk_id: c for c in chunks}

        # Dense index (ChromaDB)
        dense_index.build_index(chunks, batch_size=cfg["embedding"]["batch_size"])

        # Sparse index
        sparse_index = SparseIndex()
        sparse_index.build_index(chunks)
        sparse_index.save(cfg["indexing"]["bm25_path"])

        # Save chunk store
        import pickle
        store_path = os.path.join(cfg["indexing"]["chroma_persist_dir"], "chunk_store.pkl")
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        with open(store_path, "wb") as f:
            pickle.dump(chunk_store, f)

        return dense_index, sparse_index, chunk_store

    def run(self, query: str, top_k: int = 5, enable_guard: bool = True) -> dict:
        """Execute the full RAG pipeline (checks Redis cache first)."""

        # Check Redis cache
        if self.cache:
            cached = self.cache.get(query, top_k, enable_guard)
            if cached:
                return cached

        # Step 1: Query planning
        plan = self.query_planner.plan(query)

        # Step 2: Retrieve (use first sub-query for main retrieval, merge if multiple)
        all_chunks = []
        seen_ids = set()
        for sub_q in plan["sub_queries"]:
            chunks = self.retriever.retrieve(sub_q, filters=plan["filters"])
            for c in chunks:
                if c["chunk_id"] not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c["chunk_id"])

        # Limit to top_k
        all_chunks = all_chunks[:top_k]

        # Step 3: Generate
        answer = self.generator.generate(query, all_chunks)

        # Step 4: Hallucination guard
        guard_result = {"is_faithful": True, "score": 1.0, "flagged_sentences": []}
        if enable_guard and self.guard and all_chunks:
            guard_result = self.guard.check(answer, all_chunks)

        result = {
            "query": query,
            "plan": plan,
            "answer": answer,
            "chunks": all_chunks,
            "guard": guard_result,
        }

        # Store in Redis cache
        if self.cache:
            self.cache.set(query, top_k, enable_guard, result)

        return result


if __name__ == "__main__":
    pipeline = TalentRAGPipeline.from_config()

    test_query = "Find senior machine learning engineer roles that require PyTorch"
    result = pipeline.run(test_query)

    print(f"\nQuery: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {len(result['chunks'])}")
    print(f"Faithful: {result['guard']['is_faithful']} (score: {result['guard']['score']})")
