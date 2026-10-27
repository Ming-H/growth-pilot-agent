"""Three-layer SQLite memory (v9) — replaces the ChromaDB/TF-IDF hybrid.

Layers (tally-inspired):

- **L0 会话记忆** (``sessions`` / ``session_messages``): the chat REPL and
  multi-turn API calls inherit context — last *N* turns verbatim, earlier
  turns compressed to structured extracts (query + report head + KPIs).
  Never inherit free text as truth.
- **L1 事实台账** (``verified_facts``): structured KPI facts keyed by
  ``(org, domain, metric_key, period, run_fingerprint)``. **Write gate**:
  only successful, validated expert results with non-empty ``kpis`` enter
  the ledger — numbers only, no prose. **Read never bypasses**: ledger
  hits are context for planning/synthesis, and the KPI reconciliation
  gate treats them as ground truth alongside ``kpi_snapshot``.
- **L2 语义资产** (``aliases``): org-specific vocabulary (e.g. "货主" →
  prospect) that steers keyword routing without code changes.

Episodic search (``analyses`` + ``analyses_fts``): FTS5 over jieba
pre-tokenized text, fused with a LIKE substring fallback via RRF, with a
recency boost — the same recipe tally proved sufficient at this scale,
without a vector DB.

Everything is async (aiosqlite) with a single serialized writer — no sync
file I/O in the event loop, no asyncio.run gymnastics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_ZH_CHAR = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_STOP_WORDS = frozenset(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 "
    "the a an is are was were be been being in on at to for of with by from"
    " and or but not no this that it its".split()
)

RRF_K = 60  # reciprocal-rank fusion constant


def _tokenize(text: str) -> list[str]:
    """Tokenize for FTS indexing: jieba for Chinese, word chars for English."""
    tokens: list[str] = []
    zh_chars = _ZH_CHAR.findall(text)
    if zh_chars:
        try:
            import jieba

            tokens.extend(jieba.cut("".join(zh_chars)))
        except ImportError:
            tokens.extend(zh_chars)
    tokens.extend(w.lower() for w in _WORD_RE.findall(text))
    return [t for t in tokens if t not in _STOP_WORDS and (_ZH_CHAR.match(t) or len(t) > 1)]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    user_id TEXT DEFAULT 'anonymous',
    created_at REAL,
    updated_at REAL,
    turn_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS session_messages(
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    structured_json TEXT,
    ts REAL,
    PRIMARY KEY(session_id, seq)
);
CREATE TABLE IF NOT EXISTS verified_facts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    metric_key TEXT NOT NULL,
    period TEXT,
    value REAL,
    kpis_json TEXT,
    run_fingerprint TEXT,
    created_at REAL,
    UNIQUE(org_id, domain, metric_key, period, run_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_facts_org ON verified_facts(org_id, domain);
CREATE TABLE IF NOT EXISTS aliases(
    org_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    canonical TEXT NOT NULL,
    kind TEXT DEFAULT 'expert',
    source TEXT DEFAULT 'manual',
    hits INTEGER DEFAULT 0,
    PRIMARY KEY(org_id, alias, kind)
);
CREATE TABLE IF NOT EXISTS analyses(
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    session_id TEXT,
    query TEXT,
    scope TEXT,
    report TEXT,
    kpis_json TEXT,
    tokens TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_analyses_org ON analyses(org_id, created_at);
"""


class MemoryStore:
    """Async three-layer memory backed by one SQLite file."""

    def __init__(self, db_path: str = "./data/memory/memory.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._fts_available: bool | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.executescript(_SCHEMA)
            await self._init_fts()
            await self._migrate_legacy_json()
            await self._db.commit()
        return self._db

    async def _init_fts(self) -> None:
        """Create the FTS5 table if the SQLite build supports it."""
        assert self._db is not None
        try:
            await self._db.executescript(
                "CREATE VIRTUAL TABLE IF NOT EXISTS analyses_fts USING fts5(tokens, content='');"
            )
            await self._db.commit()
            self._fts_available = True
        except Exception as exc:
            self._fts_available = False
            logger.info("[memory] FTS5 unavailable (%s) — LIKE-only search", exc)

    @property
    def fts_available(self) -> bool:
        return bool(self._fts_available)

    async def _migrate_legacy_json(self) -> None:
        """One-time import of the v8 TF-IDF memory_store.json files."""
        assert self._db is not None
        legacy_dir = Path(self._db_path).parent
        for org_dir in legacy_dir.glob("*"):
            legacy = org_dir / "memory_store.json"
            if not org_dir.is_dir() or not legacy.is_file():
                continue
            try:
                entries = json.loads(legacy.read_text(encoding="utf-8"))
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    # _conn() is mid-initialisation here (self._db is set);
                    # awaiting it again would bind .execute on the coroutine.
                    await self._db.execute(
                        "INSERT OR IGNORE INTO analyses(id, org_id, query, scope, report,"
                        " tokens, created_at) VALUES(?,?,?,?,?,?,?)",
                        (
                            str(e.get("id", f"legacy-{time.time_ns()}")),
                            org_dir.name,
                            str(e.get("query", "")),
                            str(e.get("scope", "")),
                            str(e.get("results_summary", ""))[:2000],
                            " ".join(_tokenize(str(e.get("query", "")))),
                            float(e.get("timestamp", 0)) or time.time(),
                        ),
                    )
                await self._db.commit()
                legacy.rename(legacy.with_suffix(".json.migrated"))
                logger.info(
                    "[memory] migrated %d legacy entries for org=%s", len(entries), org_dir.name
                )
            except Exception as exc:
                logger.warning("[memory] legacy import failed for %s: %s", legacy, exc)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Episodic: save / search analyses
    # ------------------------------------------------------------------

    async def save_analysis(
        self,
        *,
        analysis_id: str,
        org_id: str,
        query: str,
        scope: str,
        report: str,
        kpis: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        tokens = " ".join(_tokenize(f"{query} {scope} {report[:2000]}"))
        async with self._lock:
            db = await self._conn()
            await db.execute(
                "INSERT OR REPLACE INTO analyses"
                "(id, org_id, session_id, query, scope, report, kpis_json, tokens, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    analysis_id,
                    org_id,
                    session_id,
                    query,
                    scope,
                    report,
                    json.dumps(kpis or {}, ensure_ascii=False),
                    tokens,
                    time.time(),
                ),
            )
            if self._fts_available:
                # Contentless FTS5: index directly by the analyses rowid.
                # (Re-saved ids get a fresh rowid; stale FTS rows join to
                # nothing and are simply unreachable.)
                await db.execute(
                    "INSERT INTO analyses_fts(rowid, tokens) "
                    "SELECT rowid, tokens FROM analyses WHERE id = ?",
                    (analysis_id,),
                )
            await db.commit()

    async def search_analyses(
        self, org_id: str, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """RRF-fuse FTS5 MATCH and LIKE substring search, recency-boosted."""
        if not query.strip():
            return []
        db = await self._conn()

        fts_ranks: dict[str, int] = {}
        if self._fts_available:
            try:
                fts_query = " ".join(_tokenize(query)) or query
                async with db.execute(
                    "SELECT a.id FROM analyses_fts f JOIN analyses a ON a.rowid = f.rowid"
                    " WHERE analyses_fts MATCH ? AND a.org_id = ? ORDER BY rank LIMIT 50",
                    (fts_query, org_id),
                ) as cur:
                    rows = await cur.fetchall()
                fts_ranks = {r[0]: i for i, r in enumerate(rows)}
            except Exception as exc:
                logger.debug("[memory] FTS search failed: %s", exc)

        like_ranks: dict[str, int] = {}
        try:
            terms = [t for t in query.split() if len(t.strip()) >= 2][:5] or [query[:40]]
            clauses = " OR ".join("(query LIKE ? OR report LIKE ?)" for _ in terms)
            params: list[Any] = [org_id]
            for t in terms:
                pattern = f"%{t}%"
                params.extend([pattern, pattern])
            async with db.execute(
                f"SELECT id FROM analyses WHERE org_id = ? AND ({clauses})"
                " ORDER BY created_at DESC LIMIT 50",
                params,
            ) as cur:
                rows = await cur.fetchall()
            like_ranks = {r[0]: i for i, r in enumerate(rows)}
        except Exception as exc:
            logger.debug("[memory] LIKE search failed: %s", exc)

        ids = set(fts_ranks) | set(like_ranks)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        async with db.execute(
            f"SELECT id, query, scope, report, kpis_json, created_at FROM analyses"
            f" WHERE org_id = ? AND id IN ({placeholders})",
            (org_id, *ids),
        ) as cur:
            rows = await cur.fetchall()

        def _score(row: tuple) -> float:
            aid = row[0]
            s = 0.0
            if aid in fts_ranks:
                s += 1.0 / (RRF_K + fts_ranks[aid] + 1)
            if aid in like_ranks:
                s += 1.0 / (RRF_K + like_ranks[aid] + 1)
            age = time.time() - (row[5] or 0)
            return s * (1.0 + 1.0 / (1.0 + age / 86400))

        scored = sorted(rows, key=_score, reverse=True)[:limit]
        return [
            {
                "id": r[0],
                "query": r[1],
                "scope": r[2],
                "results_summary": (r[3] or "")[:400],
                "kpis": json.loads(r[4]) if r[4] else {},
                "timestamp": r[5],
                "_relevance_score": round(_score(r), 4),
            }
            for r in scored
        ]

    async def get_recent_analyses(self, org_id: str, limit: int = 10) -> list[dict[str, Any]]:
        db = await self._conn()
        async with db.execute(
            "SELECT id, query, scope, created_at FROM analyses WHERE org_id = ?"
            " ORDER BY created_at DESC LIMIT ?",
            (org_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [{"id": r[0], "query": r[1], "scope": r[2], "timestamp": r[3]} for r in rows]

    # ------------------------------------------------------------------
    # L0: sessions
    # ------------------------------------------------------------------

    async def save_turn(
        self,
        session_id: str,
        *,
        org_id: str,
        user_id: str = "anonymous",
        role: str,
        content: str,
        structured: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            db = await self._conn()
            now = time.time()
            await db.execute(
                "INSERT INTO sessions(id, org_id, user_id, created_at, updated_at, turn_count)"
                " VALUES(?,?,?,?,?,0)"
                " ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at,"
                " turn_count = turn_count + 1",
                (session_id, org_id, user_id, now, now),
            )
            async with db.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM session_messages WHERE session_id = ?",
                (session_id,),
            ) as cur:
                (seq,) = await cur.fetchone()
            await db.execute(
                "INSERT INTO session_messages(session_id, seq, role, content, structured_json, ts)"
                " VALUES(?,?,?,?,?,?)",
                (
                    session_id,
                    seq,
                    role,
                    content[:8000],
                    json.dumps(structured, ensure_ascii=False, default=str) if structured else None,
                    now,
                ),
            )
            await db.commit()

    async def load_session(self, session_id: str, *, last_n_full: int = 3) -> dict[str, Any]:
        """Return {'recent': [...verbatim last N turns...], 'compressed': str}.

        Earlier turns are compressed to a structured extract — the query,
        the report head, and the KPIs — never free text masquerading as truth.
        """
        db = await self._conn()
        async with db.execute(
            "SELECT role, content, structured_json FROM session_messages"
            " WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return {"recent": [], "compressed": ""}

        recent = [
            {"role": r[0], "content": r[1]}
            for r in rows[-(last_n_full * 2) :]  # user+assistant per turn
        ]
        older = rows[: -(last_n_full * 2)] if len(rows) > last_n_full * 2 else []
        compressed_parts: list[str] = []
        for r in older:
            kpis = ""
            if r[2]:
                try:
                    parsed = json.loads(r[2])
                    if parsed.get("kpis"):
                        kpis = " kpis=" + json.dumps(parsed["kpis"], ensure_ascii=False)[:200]
                except json.JSONDecodeError:
                    pass
            head = (r[1] or "")[:200].replace("\n", " ")
            compressed_parts.append(f"[{r[0]}] {head}{kpis}")
        return {"recent": recent, "compressed": "\n".join(compressed_parts)}

    # ------------------------------------------------------------------
    # L1: verified-fact ledger
    # ------------------------------------------------------------------

    async def record_facts(
        self,
        org_id: str,
        expert_result: dict[str, Any],
        run_fingerprint: str,
    ) -> int:
        """Write an expert's KPIs into the ledger — gated.

        Gate: ``success is True`` ∧ ``kpis`` non-empty ∧ ``_validated``.
        Returns the number of facts written.
        """
        if expert_result.get("success") is not True:
            return 0
        if not expert_result.get("_validated"):
            return 0
        kpis = expert_result.get("kpis") or {}
        if not isinstance(kpis, dict) or not kpis:
            return 0

        domain = str(expert_result.get("expert", "unknown"))
        written = 0
        async with self._lock:
            db = await self._conn()
            for key, value in kpis.items():
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    continue
                await db.execute(
                    "INSERT OR IGNORE INTO verified_facts"
                    "(org_id, domain, metric_key, period, value, kpis_json, run_fingerprint,"
                    " created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        org_id,
                        domain,
                        str(key),
                        "",  # '' not NULL: SQLite UNIQUE treats NULLs as distinct
                        num,
                        json.dumps(kpis, ensure_ascii=False),
                        run_fingerprint,
                        time.time(),
                    ),
                )
                written += 1
            await db.commit()
        return written

    async def search_facts(
        self,
        org_id: str,
        *,
        domain: str | None = None,
        metric_key: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Read facts (never bypasses the gate — read is context, not truth)."""
        db = await self._conn()
        sql = (
            "SELECT domain, metric_key, period, value, run_fingerprint, created_at"
            " FROM verified_facts WHERE org_id = ?"
        )
        params: list[Any] = [org_id]
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if metric_key:
            sql += " AND metric_key LIKE ?"
            params.append(f"%{metric_key}%")
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            {
                "domain": r[0],
                "metric_key": r[1],
                "period": r[2],
                "value": r[3],
                "run_fingerprint": r[4],
                "timestamp": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # L2: aliases
    # ------------------------------------------------------------------

    async def upsert_alias(
        self, org_id: str, alias: str, canonical: str, kind: str = "expert"
    ) -> None:
        async with self._lock:
            db = await self._conn()
            await db.execute(
                "INSERT INTO aliases(org_id, alias, canonical, kind, source)"
                " VALUES(?,?,?,?, 'manual')"
                " ON CONFLICT(org_id, alias, kind)"
                " DO UPDATE SET canonical = excluded.canonical",
                (org_id, alias, canonical, kind),
            )
            await db.commit()

    async def resolve_aliases(self, org_id: str, text: str) -> dict[str, set[str]]:
        """Scan text for known aliases → {kind: {canonical,...}}; bumps hit counts."""
        db = await self._conn()
        async with db.execute(
            "SELECT alias, canonical, kind FROM aliases WHERE org_id = ?", (org_id,)
        ) as cur:
            rows = await cur.fetchall()
        hits: dict[str, set[str]] = {}
        matched: list[tuple[str, str]] = []
        for alias, canonical, kind in rows:
            if alias and alias in text:
                hits.setdefault(kind, set()).add(canonical)
                matched.append((alias, kind))
        if matched:
            async with self._lock:
                for alias, kind in matched:
                    await db.execute(
                        "UPDATE aliases SET hits = hits + 1 WHERE org_id = ? AND alias = ?"
                        " AND kind = ?",
                        (org_id, alias, kind),
                    )
                await db.commit()
        return hits

    async def list_aliases(self, org_id: str) -> list[dict[str, Any]]:
        db = await self._conn()
        async with db.execute(
            "SELECT alias, canonical, kind, source, hits FROM aliases WHERE org_id = ?"
            " ORDER BY kind, alias",
            (org_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {"alias": r[0], "canonical": r[1], "kind": r[2], "source": r[3], "hits": r[4]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    async def clear_org(self, org_id: str) -> int:
        """Delete every trace of an org (analyses, facts, aliases, sessions)."""
        async with self._lock:
            db = await self._conn()
            cur = await db.execute("DELETE FROM analyses WHERE org_id = ?", (org_id,))
            removed = cur.rowcount or 0
            for table in ("verified_facts", "aliases", "sessions", "session_messages"):
                if table == "session_messages":
                    await db.execute(
                        "DELETE FROM session_messages WHERE session_id IN"
                        " (SELECT id FROM sessions WHERE org_id = ?)",
                        (org_id,),
                    )
                else:
                    await db.execute(f"DELETE FROM {table} WHERE org_id = ?", (org_id,))
            await db.commit()
            return removed

    async def count(self, org_id: str) -> dict[str, int]:
        db = await self._conn()
        out: dict[str, int] = {}
        for table in ("analyses", "verified_facts", "aliases", "sessions"):
            async with db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE org_id = ?", (org_id,)
            ) as cur:
                (out[table],) = await cur.fetchone()
        return out
