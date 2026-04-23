"""MemoryManager - Persistent memory with TF-IDF based semantic search.

Architecture follows a 4-layer pattern:
  1. Extract  - Identify key information from analysis results
  2. Store    - Persist structured memory entries as JSON
  3. Retrieve - TF-IDF keyword matching for semantic search
  4. Inject   - Build context string for prompt injection

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
# Chinese + English tokenization helper
# ---------------------------------------------------------------------------

# Match individual Chinese characters or runs of alphanumeric words
_ZH_CHAR = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_ZH_SEGMENT_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: split Chinese into unigrams + English/numbers into words.

    Returns lowercase tokens suitable for TF-IDF matching.
    """
    if not text:
        return []

    tokens: list[str] = []

    # Extract Chinese unigrams
    for ch in text:
        if _ZH_CHAR.match(ch):
            tokens.append(ch)

    # Extract English/number words
    for word in _WORD_RE.findall(text):
        tokens.append(word.lower())

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
# TF-IDF computation
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
    """File-based persistent memory manager with TF-IDF semantic search.

    Memory entries are stored as a single JSON file (``memory_store.json``)
    under the configured base path.  Each entry contains:
      - id:        unique UUID
      - run_id:    the analysis run identifier
      - query:     original user query
      - scope:     analysis scope (prospect / conversion / etc.)
      - results_summary:  summary dict from the analysis
      - timestamp: epoch seconds when stored

    The search method uses TF-IDF cosine similarity for retrieval.
    """

    _STORE_FILENAME = "memory_store.json"

    def __init__(self, base_path: str = "./data/memory") -> None:
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._store_path = self.base_path / self._STORE_FILENAME
        self._entries: list[dict[str, Any]] = self._load()

    # ------------------------------------------------------------------
    # Persistence (Extract + Store layers)
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

        self._entries.append(entry)
        self._save()

        logger.info(
            "Memory stored: id=%s run_id=%s scope=%s",
            entry_id, run_id, scope,
        )
        return entry_id

    # ------------------------------------------------------------------
    # Public API - Retrieve (TF-IDF search)
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for relevant memory entries using TF-IDF cosine similarity.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.

        Returns:
            List of memory entries sorted by relevance (most relevant first).
        """
        if not self._entries or not query:
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
        logger.info("Memory cleared: %d entries removed", count)
        return count

    def count(self) -> int:
        """Return the total number of stored memory entries."""
        return len(self._entries)

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
