"""Memory endpoints: recent analyses, clear, L2 aliases."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from src.api_models import MemoryClearResponse, MemoryEntryResponse, MemoryResponse
from src.auth.dependencies import get_current_user
from src.db.models import User
from src.web.deps import get_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


@router.get("", response_model=MemoryResponse)
async def list_memory(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
) -> MemoryResponse:
    """Retrieve recent analyses, scoped to the user's org."""
    store = get_memory()
    entries = await store.get_recent_analyses(current_user.org_id, limit=limit)
    memories = [
        MemoryEntryResponse(
            id=e.get("id", ""),
            run_id=e.get("id", ""),
            query=e.get("query", ""),
            scope=e.get("scope", ""),
            results_summary=e.get("results_summary", ""),
            timestamp=e.get("timestamp", 0.0),
        )
        for e in entries
    ]
    return MemoryResponse(memories=memories, total=len(memories))


@router.delete("", response_model=MemoryClearResponse)
async def clear_memory(
    current_user: User = Depends(get_current_user),
) -> MemoryClearResponse:
    """Clear every memory trace for the user's org."""
    removed = await get_memory().clear_org(current_user.org_id)
    return MemoryClearResponse(
        removed_count=removed,
        message=f"Cleared {removed} analyses (plus facts/aliases/sessions) for org"
        f" {current_user.org_id}",
    )


@router.get("/aliases")
async def list_aliases(
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List org vocabulary aliases (L2 semantic assets)."""
    return await get_memory().list_aliases(current_user.org_id)


@router.put("/aliases")
async def upsert_alias(
    alias: str = Query(..., min_length=1, max_length=64),
    canonical: str = Query(..., min_length=1, max_length=64),
    kind: str = Query("expert"),
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Upsert an org alias (e.g. 货主 → prospect) that steers keyword routing."""
    await get_memory().upsert_alias(current_user.org_id, alias, canonical, kind)
    return {"status": "ok", "alias": alias, "canonical": canonical}


@router.get("/facts")
async def list_facts(
    domain: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Read the L1 verified-fact ledger for the org (write-gated upstream)."""
    return await get_memory().search_facts(current_user.org_id, domain=domain, limit=limit)
