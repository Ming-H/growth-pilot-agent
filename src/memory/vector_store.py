"""VectorMemoryStore – ChromaDB-backed persistent vector memory.

Provides ``store()`` and ``search()`` methods that operate on per-organisation
ChromaDB collections.  Uses ``OpenAIEmbeddings`` (text-embedding-3-small) for
embedding generation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import chromadb  # type: ignore[import-untyped]
    from chromadb.config import Settings as ChromaSettings  # type: ignore[import-untyped]

    _CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    _CHROMADB_AVAILABLE = False


class VectorMemoryStore:
    """Persistent vector store backed by ChromaDB with cosine similarity.

    Each organisation gets its own collection (``org_{org_id}``).  Documents
    are embedded with ``text-embedding-3-small`` and upserted with metadata
    so they can be recalled via semantic search.
    """

    def __init__(self, persist_dir: str = "./data/chroma") -> None:
        if not _CHROMADB_AVAILABLE:
            raise RuntimeError(
                "chromadb is not installed. Install it or use the file-based fallback."
            )
        self.client = chromadb.PersistentClient(path=persist_dir)

        from src.memory.embedding import get_embeddings

        self.embeddings = get_embeddings()
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings are not available. Cannot use VectorMemoryStore."
            )

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def _get_collection(self, org_id: str):  # noqa: ANN202
        """Get or create a ChromaDB collection for the given organisation."""
        return self.client.get_or_create_collection(
            name=f"org_{org_id}",
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _make_doc_id(query: str, org_id: str) -> str:
        """Deterministic document ID from query + org_id."""
        return f"mem_{hashlib.sha256((query + org_id).encode()).hexdigest()[:16]}"

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store(
        self,
        org_id: str,
        query: str,
        summary: str,
        results: dict[str, Any] | str,
        tags: list[str] | None = None,
    ) -> str:
        """Store an analysis result as a vector document.

        Returns the generated document ID.
        """
        tags = tags or []
        collection = self._get_collection(org_id)
        doc_id = self._make_doc_id(query, org_id)

        # Build the text to embed: query + summary gives richer semantic signal
        embed_text = f"{query}\n{summary}"

        # Synchronous embed – ChromaDB client is synchronous
        embedding = self.embeddings.embed_query(embed_text)

        # Ensure results is a JSON-serialisable string
        if isinstance(results, (dict, list)):
            results_json = json.dumps(results, ensure_ascii=False)
        else:
            results_json = str(results)

        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[embed_text],
            metadatas=[
                {
                    "org_id": org_id,
                    "tags": json.dumps(tags, ensure_ascii=False),
                    "results_summary": results_json[:2000],
                    "timestamp": time.time(),
                }
            ],
        )

        logger.info(
            "VectorMemoryStore.store: id=%s org=%s tags=%s",
            doc_id,
            org_id,
            tags,
        )
        return doc_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        org_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for relevant memories by semantic similarity.

        Returns a list of dicts with keys: ``content``, ``metadata``,
        ``distance``.
        """
        collection = self._get_collection(org_id)

        # Check if collection has any documents
        if collection.count() == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )

        memories: list[dict[str, Any]] = []
        if not results["documents"] or not results["documents"][0]:
            return memories

        for i, doc in enumerate(results["documents"][0]):
            entry: dict[str, Any] = {
                "content": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            }
            memories.append(entry)

        return memories

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_recent(self, org_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve the most recent entries for an organisation.

        Since ChromaDB does not natively sort by metadata, we peek at the
        collection and sort by the ``timestamp`` metadata field.
        """
        collection = self._get_collection(org_id)
        count = collection.count()
        if count == 0:
            return []

        # Peek at all entries (ChromaDB .get with no filter returns all)
        all_data = collection.get(include=["documents", "metadatas"])

        entries: list[dict[str, Any]] = []
        for i, doc in enumerate(all_data["documents"]):
            meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
            entries.append(
                {
                    "id": all_data["ids"][i],
                    "content": doc,
                    "metadata": meta,
                    "timestamp": meta.get("timestamp", 0),
                }
            )

        # Sort by timestamp descending
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        return entries[:limit]

    def count(self, org_id: str) -> int:
        """Return the number of stored documents for an organisation."""
        collection = self._get_collection(org_id)
        return collection.count()

    def clear(self, org_id: str) -> int:
        """Delete all documents for an organisation. Returns count removed."""
        collection = self._get_collection(org_id)
        count = collection.count()
        if count > 0:
            # Delete the collection entirely
            self.client.delete_collection(name=f"org_{org_id}")
        return count
