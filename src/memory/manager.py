"""MemoryManager - Persistent memory with vector semantic search.

Architecture follows a 4-layer pattern:
  1. Extract  - Identify key information from analysis results
  2. Store    - Persist structured memory entries (ChromaDB vector store)
  3. Retrieve - Embedding-based semantic search with recency boosting
  4. Inject   - Build context string for prompt injection

Falls back to a file-based TF-IDF store when ChromaDB or embeddings are
not available, ensuring the application always starts.

Reference: claude-cookbooks MemoryToolHandler (filesystem persistence)
           + ai-app-lab longterm_memory (extract-store-retrieve-inject)
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chinese + English tokenization helper (used by TF-IDF fallback)
# ---------------------------------------------------------------------------

_ZH_CHAR = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize text for TF-IDF. Uses jieba for Chinese if available."""
    tokens: list[str] = []
    try:
        import jieba
        zh_chars = _ZH_CHAR.findall(text)
        if zh_chars:
            tokens.extend(jieba.cut("".join(zh_chars)))
    except ImportError:
        tokens.extend(_ZH_CHAR.findall(text))
    tokens.extend(w.lower() for w in _WORD_RE.findall(text))
    return tokens


# ---------------------------------------------------------------------------
# Stop words (minimal set for filtering very common tokens)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    # English
    {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
     "in", "on", "at", "to", "for", "of", "with", "by", "from", "and",
     "or", "but", "not", "no", "this", "that", "it", "its", "has", "have",
     "had", "do", "does", "did", "will", "would", "can", "could",
     # Chinese common particles
     "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
     "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
     "会", "着", "没有", "看", "好", "自己", "这"}
)


def _filter_tokens(tokens: list[str]) -> list[str]:
    """Remove stop words and single-char non-Chinese tokens."""
    return [t for t in tokens if t not in _STOP_WORDS and (_ZH_CHAR.match(t) or len(t) > 1)]


# ---------------------------------------------------------------------------
# TF-IDF computation (fallback mode)
# ---------------------------------------------------------------------------

def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency for a token list."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _compute_idf(doc_tokens_list: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across all documents."""
    n_docs = len(doc_tokens_list)
    if n_docs == 0:
        return {}

    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens_list:
        unique = set(tokens)
        for term in unique:
            doc_freq[term] += 1

    # Smooth IDF: log((1 + N) / (1 + df)) + 1
    return {
        term: math.log((1 + n_docs) / (1 + df)) + 1
        for term, df in doc_freq.items()
    }


def _tfidf_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
) -> float:
    """Compute cosine similarity between query and document TF-IDF vectors."""
    if not query_tokens or not doc_tokens:
        return 0.0

    query_tf = _compute_tf(query_tokens)
    doc_tf = _compute_tf(doc_tokens)

    # Compute dot product over shared terms
    dot_product = 0.0
    for term, q_tf in query_tf.items():
        if term in doc_tf:
            idf_val = idf.get(term, 1.0)
            dot_product += q_tf * idf_val * doc_tf[term] * idf_val

    # Compute magnitudes
    def _magnitude(tf_dict: dict[str, float]) -> float:
        return math.sqrt(sum((v * idf.get(t, 1.0)) ** 2 for t, v in tf_dict.items()))

    mag_q = _magnitude(query_tf)
    mag_d = _magnitude(doc_tf)

    if mag_q == 0.0 or mag_d == 0.0:
        return 0.0

    return dot_product / (mag_q * mag_d)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Persistent memory manager with vector semantic search.

    Uses ChromaDB + OpenAI embeddings when available.  Falls back to a
    file-based TF-IDF store when those dependencies are missing.

    Memory entries are stored as a single JSON file (``memory_store.json``)
    under the configured base path when in fallback mode.  Each entry
    contains:
      - id:        unique UUID
      - run_id:    the analysis run identifier
      - query:     original user query
      - scope:     analysis scope (prospect / conversion / etc.)
      - results_summary:  summary dict from the analysis
      - timestamp: epoch seconds when stored

    The public API is the same regardless of which backend is active.
    """

    _STORE_FILENAME = "memory_store.json"

    def __init__(self, base_path: str = "./data/memory", *, org_id: str) -> None:
        self._org_id = org_id
        self.base_path = Path(base_path).resolve()
        self._store_path = self.base_path / self._org_id / self._STORE_FILENAME

        # Attempt to initialise the vector store backend
        self._vector_store = None
        self._use_vector = False
        self._init_vector_store()

        # TF-IDF fallback entries (always loaded so count/clear work)
        self._entries: list[dict[str, Any]] = self._load()

    def _init_vector_store(self) -> None:
        """Try to set up the ChromaDB vector store.  On failure, silently
        fall back to TF-IDF file mode."""
        try:
            from src.memory.embedding import get_embeddings, is_fallback

            embeddings = get_embeddings()
            if embeddings is None or is_fallback():
                logger.info("Embeddings unavailable – using TF-IDF fallback for memory.")
                return

            from src.memory.vector_store import VectorMemoryStore

            chroma_dir = str(self.base_path.parent / "chroma")
            self._vector_store = VectorMemoryStore(persist_dir=chroma_dir)
            self._use_vector = True
            logger.info("Vector memory backend active (ChromaDB at %s)", chroma_dir)
        except Exception as exc:
            logger.info(
                "Vector store unavailable (%s) – using TF-IDF fallback for memory.",
                exc,
            )
            self._vector_store = None
            self._use_vector = False

    # ------------------------------------------------------------------
    # Persistence (TF-IDF fallback – Extract + Store layers)
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        """Load memory entries from JSON file."""
        if not self._store_path.exists():
            return []
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            logger.warning("Memory store has unexpected format, resetting.")
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load memory store: %s", exc)
            return []

    def _save(self) -> None:
        """Persist memory entries to JSON file."""
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store_path.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to save memory store: %s", exc)

    # ------------------------------------------------------------------
    # Public API - Store
    # ------------------------------------------------------------------

    def store(
        self,
        run_id: str,
        query: str,
        scope: str,
        results_summary: dict[str, Any] | str,
    ) -> str:
        """Store an analysis result as a memory entry.

        Args:
            run_id:         Unique identifier for this analysis run.
            query:          The user's original query.
            scope:          The analysis scope (e.g. ``prospect``, ``full``).
            results_summary: Summary dict or string from the analysis.

        Returns:
            The memory entry ID (UUID string).
        """
        entry_id = uuid.uuid4().hex

        # Ensure results_summary is serializable string or dict
        if isinstance(results_summary, (dict, list)):
            summary_data = results_summary
        else:
            summary_data = str(results_summary)

        entry: dict[str, Any] = {
            "id": entry_id,
            "run_id": run_id,
            "query": query,
            "scope": scope,
            "results_summary": summary_data,
            "timestamp": time.time(),
        }

        # Always keep in-memory list for get_recent/count/clear compatibility
        self._entries.append(entry)
        self._save()

        # Also persist to vector store if available
        if self._use_vector and self._vector_store is not None:
            try:
                import asyncio

                summary_str = (
                    json.dumps(summary_data, ensure_ascii=False)
                    if isinstance(summary_data, (dict, list))
                    else summary_data
                )
                # Run the async store in a synchronous context
                try:
                    asyncio.get_running_loop()  # raises RuntimeError if no loop
                    # We're inside an existing event loop – schedule it
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run,
                            self._vector_store.store(
                                org_id=self._org_id,
                                query=query,
                                summary=summary_str,
                                results=summary_data,
                                tags=[scope],
                            ),
                        )
                        future.result(timeout=10)
                except RuntimeError:
                    # No running loop – safe to use asyncio.run
                    asyncio.run(
                        self._vector_store.store(
                            org_id=self._org_id,
                            query=query,
                            summary=summary_str,
                            results=summary_data,
                            tags=[scope],
                        )
                    )
            except Exception as exc:
                logger.warning("Vector store write failed: %s", exc)

        logger.info(
            "Memory stored: id=%s run_id=%s scope=%s vector=%s",
            entry_id, run_id, scope, self._use_vector,
        )
        return entry_id

    # ------------------------------------------------------------------
    # Public API - Retrieve
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for relevant memory entries.

        Uses the vector store (ChromaDB) when available, otherwise falls
        back to TF-IDF cosine similarity.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.

        Returns:
            List of memory entries sorted by relevance (most relevant first).
        """
        if not query:
            return []

        # --- Vector search path ---
        if self._use_vector and self._vector_store is not None:
            try:
                import asyncio

                try:
                    asyncio.get_running_loop()  # raises RuntimeError if no loop
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run,
                            self._vector_store.search(
                                org_id=self._org_id,
                                query=query,
                                top_k=limit,
                            ),
                        )
                        vector_results = future.result(timeout=10)
                except RuntimeError:
                    vector_results = asyncio.run(
                        self._vector_store.search(
                            org_id=self._org_id,
                            query=query,
                            top_k=limit,
                        )
                    )

                if vector_results:
                    return self._enrich_vector_results(vector_results, limit)
            except Exception as exc:
                logger.warning("Vector search failed, falling back to TF-IDF: %s", exc)

        # --- TF-IDF fallback path ---
        return self._tfidf_search(query, limit)

    def _enrich_vector_results(
        self,
        vector_results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Convert vector store results to the public format with recency boost."""
        enriched: list[dict[str, Any]] = []

        for item in vector_results:
            content = item.get("content", "")
            meta = item.get("metadata", {})
            distance = item.get("distance", 1.0)

            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score [0, 1]
            similarity = max(0.0, 1.0 - distance / 2.0)

            # Recency boost (same logic as TF-IDF path)
            ts = meta.get("timestamp", 0)
            age_seconds = time.time() - ts if ts else float("inf")
            recency_boost = 1.0 / (1.0 + age_seconds / 86400)

            final_score = similarity * (1 + recency_boost)

            # Try to find the matching local entry for richer data
            local_entry = self._find_local_entry(meta, content)

            enriched.append({
                **local_entry,
                "content": content,
                "_relevance_score": round(final_score, 4),
                "_source": "vector",
            })

        enriched.sort(key=lambda x: x["_relevance_score"], reverse=True)
        return enriched[:limit]

    def _find_local_entry(
        self,
        meta: dict[str, Any],
        content: str,
    ) -> dict[str, Any]:
        """Try to match a vector result to a local file entry for richer data."""
        for entry in self._entries:
            if content.startswith(entry.get("query", "")):
                return dict(entry)
        return {
            "id": meta.get("id", ""),
            "query": content.split("\n")[0] if content else "",
            "scope": "",
            "results_summary": meta.get("results_summary", ""),
            "timestamp": meta.get("timestamp", 0),
        }

    def _tfidf_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """TF-IDF cosine similarity search (fallback)."""
        if not self._entries:
            return []

        # Tokenize the query
        query_tokens = _filter_tokens(_tokenize(query))
        if not query_tokens:
            # Fallback to recent entries if query has no useful tokens
            return self.get_recent(limit=limit)

        # Build token lists for all entries (query + scope + summary text)
        doc_token_map: list[tuple[dict[str, Any], list[str]]] = []
        all_doc_tokens: list[list[str]] = []

        for entry in self._entries:
            # Combine searchable fields into one text blob
            searchable = self._entry_searchable_text(entry)
            doc_tokens = _filter_tokens(_tokenize(searchable))
            doc_token_map.append((entry, doc_tokens))
            all_doc_tokens.append(doc_tokens)

        # Add query tokens to compute IDF across query+docs
        all_doc_tokens.append(query_tokens)

        # Compute IDF
        idf = _compute_idf(all_doc_tokens)

        # Score each entry
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry, doc_tokens in doc_token_map:
            score = _tfidf_score(query_tokens, doc_tokens, idf)
            # Boost by recency (linear decay: newer entries get a slight boost)
            age_seconds = time.time() - entry.get("timestamp", 0)
            recency_boost = 1.0 / (1.0 + age_seconds / 86400)  # half-life ~1 day
            final_score = score * (1 + recency_boost)
            scored.append((final_score, entry))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Filter zero-score entries
        results = [
            {**entry, "_relevance_score": round(score, 4)}
            for score, entry in scored
            if score > 0
        ]

        return results[:limit]

    def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most recent memory entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of memory entries sorted by timestamp (newest first).
        """
        sorted_entries = sorted(
            self._entries,
            key=lambda e: e.get("timestamp", 0),
            reverse=True,
        )
        return sorted_entries[:limit]

    # ------------------------------------------------------------------
    # Public API - Inject (build context for prompt)
    # ------------------------------------------------------------------

    def build_context(self, query: str) -> str:
        """Build a memory context string for injection into agent prompts.

        Retrieves relevant historical memories and formats them as a
        structured context block.

        Args:
            query: The current user query to search against.

        Returns:
            Formatted context string ready for prompt injection.
            Returns empty string if no relevant memories found.
        """
        if not query:
            return ""

        # Search for relevant memories
        relevant = self.search(query, limit=5)
        if not relevant:
            # Fallback: include 2 most recent entries for general context
            relevant = self.get_recent(limit=2)

        if not relevant:
            return ""

        parts = ["## 历史分析记忆（参考）"]
        parts.append("以下是之前的分析结果，供综合当前分析时参考：")
        parts.append("")

        for i, entry in enumerate(relevant, 1):
            query_text = entry.get("query", "")
            scope = entry.get("scope", "")
            summary = entry.get("results_summary", "")
            run_id = entry.get("run_id", "")
            ts = entry.get("timestamp", 0)

            # Format timestamp
            if ts:
                from datetime import datetime
                time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            else:
                time_str = "未知时间"

            # Format summary
            if isinstance(summary, dict):
                summary_str = json.dumps(summary, ensure_ascii=False, indent=2)
            else:
                summary_str = str(summary)

            # Truncate very long summaries
            if len(summary_str) > 500:
                summary_str = summary_str[:500] + "..."

            parts.append(f"### 记忆 {i} [{time_str}] (scope: {scope}, run: {run_id})")
            parts.append(f"- 原始查询: {query_text}")
            parts.append(f"- 分析摘要: {summary_str}")
            parts.append("")

        parts.append("请在分析时参考以上历史记忆，但以当前数据和分析结果为主。")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """Clear all memory entries. Returns the count of entries removed."""
        count = len(self._entries)
        self._entries = []
        self._save()

        # Also clear vector store
        if self._use_vector and self._vector_store is not None:
            try:
                vector_count = self._vector_store.clear(self._org_id)
                logger.info("Vector store cleared: %d entries removed", vector_count)
            except Exception as exc:
                logger.warning("Vector store clear failed: %s", exc)

        logger.info("Memory cleared: %d entries removed", count)
        return count

    def count(self) -> int:
        """Return the total number of stored memory entries."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Async public API
    # ------------------------------------------------------------------

    async def astore(
        self,
        run_id: str,
        query: str,
        scope: str,
        results_summary: str = "",
        **kwargs: Any,
    ) -> str:
        """Async version of store()."""
        return self.store(
            run_id=run_id,
            query=query,
            scope=scope,
            results_summary=results_summary,
        )

    async def asearch(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Async version of search()."""
        return self.search(query=query, limit=limit)

    async def aclear(self) -> int:
        """Async version of clear()."""
        return self.clear()

    @property
    def uses_vector_store(self) -> bool:
        """Return True if the ChromaDB vector backend is active."""
        return self._use_vector

    @staticmethod
    def _entry_searchable_text(entry: dict[str, Any]) -> str:
        """Combine entry fields into a single searchable text string."""
        parts = [
            entry.get("query", ""),
            entry.get("scope", ""),
        ]
        summary = entry.get("results_summary", "")
        if isinstance(summary, dict):
            # Flatten dict values into text
            for v in summary.values():
                if isinstance(v, str):
                    parts.append(v)
                else:
                    parts.append(str(v))
        elif isinstance(summary, str):
            parts.append(summary)
        return " ".join(parts)
