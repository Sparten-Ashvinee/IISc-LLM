# TalentRAG — From-Scratch RAG over LinkedIn Job Postings

A production-grade Retrieval-Augmented Generation system built **entirely from scratch** (no LangChain, no LlamaIndex) for semantic search over 50K+ LinkedIn job postings.

## Architecture

```
Query → Query Planner → Hybrid Retrieval (Dense + BM25 → RRF) → Cross-Encoder Reranker → Grounded Generation → Hallucination Guard → Answer
```

## Project Structure

```
LinkedIn_RAG/
├── configs/config.yaml           # All hyperparameters
├── data/
│   ├── load_data.py              # HuggingFace dataset loader
│   └── chunker.py                # Profile-aware chunking
├── indexing/
│   ├── embed.py                  # Dense embeddings + FAISS
│   ├── bm25_index.py             # BM25 sparse index
│   └── hybrid.py                 # Reciprocal Rank Fusion
├── retrieval/
│   ├── query_planner.py          # Query decomposition + filter extraction
│   ├── retriever.py              # Hybrid retrieval orchestrator
│   └── reranker.py               # Cross-encoder reranking
├── generation/
│   ├── generator.py              # Grounded generation with citations
│   └── hallucination_guard.py    # NLI-based faithfulness check
├── eval/
│   ├── retrieval_eval.py         # Recall@k, MRR@k, NDCG@k
│   └── generation_eval.py        # Faithfulness, citation coverage, latency
├── serving/
│   ├── api.py                    # FastAPI endpoints
│   └── ui.py                     # Streamlit demo UI
├── notebooks/
│   └── eda.py                    # Quick data exploration
├── pipeline.py                   # Main end-to-end pipeline
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
cd LinkedIn_RAG
pip install -r requirements.txt
```

### 2. Run EDA (optional)
```bash
cd notebooks
python eda.py
```

### 3. Build indexes and run pipeline
```bash
python pipeline.py
```
This will:
- Download the LinkedIn dataset from HuggingFace
- Chunk all job postings
- Build FAISS + BM25 indexes (saved to `artifacts/`)
- Run a sample query end-to-end

### 4. Start API server
```bash
python -m serving.api
```

### 5. Start Streamlit UI (in another terminal)
```bash
streamlit run serving/ui.py
```

## Models Used

| Component | Model |
|---|---|
| Dense Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `mistralai/Mistral-7B-Instruct-v0.3` (4-bit quantized) |
| Hallucination Guard | `microsoft/deberta-v3-base` (NLI) |

## Dataset

**`arshkon/linkedin-job-postings`** — 124K LinkedIn job postings with title, description, company, location, skills, experience level.

Imported directly via HuggingFace `datasets` — no synthetic data needed.

## Evaluation Metrics

- **Retrieval**: Recall@10, MRR@10, NDCG@5
- **Generation**: Faithfulness score, citation coverage, hallucination rate
- **System**: Latency p50/p95

## Key Design Decisions

1. **No framework dependency** — every component is hand-built and explainable
2. **Hybrid retrieval + RRF** — combines semantic understanding with keyword precision
3. **Profile-aware chunking** — splits job descriptions on semantic section boundaries, not arbitrary character limits
4. **Mandatory citations** — every generated claim maps to a source
5. **NLI-based guard** — flags hallucinated content before it reaches the user
