"""Dense embedding and ChromaDB vector store."""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DenseIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_dir: str = "./artifacts/chromadb",
                 collection_name: str = "linkedin_jobs"):
        self.encoder = SentenceTransformer(model_name)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def _init_client(self) -> None:
        """Initialize ChromaDB persistent client."""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

    def build_index(self, chunks: list, batch_size: int = 256) -> None:
        """Encode all chunks and store in ChromaDB."""
        self._init_client()

        # Delete existing collection if rebuilding
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "chunk_type": c.chunk_type,
                "title": c.metadata.get("title", ""),
                "company_name": c.metadata.get("company_name", ""),
                "location": c.metadata.get("location", ""),
                "experience_level": c.metadata.get("experience_level", ""),
            }
            for c in chunks
        ]

        print(f"Encoding and indexing {len(texts):,} chunks into ChromaDB...")

        # ChromaDB has a batch limit, insert in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            end = min(i + batch_size, len(texts))
            batch_texts = texts[i:end]
            batch_ids = chunk_ids[i:end]
            batch_meta = metadatas[i:end]

            embeddings = self.encoder.encode(
                batch_texts, normalize_embeddings=True,
            ).tolist()

            self.collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )

        print(f"ChromaDB index built: {self.collection.count()} vectors in '{self.collection_name}'")

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) for a query."""
        q_emb = self.encoder.encode([query], normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
        )

        output = []
        if results["ids"] and results["distances"]:
            for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
                # ChromaDB cosine distance: lower = more similar; convert to similarity
                score = 1.0 - distance
                output.append((chunk_id, float(score)))
        return output

    def load(self, persist_dir: str | None = None, collection_name: str | None = None) -> None:
        """Load an existing ChromaDB collection from disk."""
        self.persist_dir = persist_dir or self.persist_dir
        self.collection_name = collection_name or self.collection_name
        self._init_client()
        self.collection = self.client.get_collection(self.collection_name)
        print(f"Loaded ChromaDB collection '{self.collection_name}': {self.collection.count()} vectors")
