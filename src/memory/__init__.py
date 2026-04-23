"""Memory system for GrowthPilot - persistent cross-session memory.

Provides:
  - MemoryManager: High-level API with automatic backend selection
    (ChromaDB + embeddings when available, TF-IDF file fallback otherwise).
  - VectorMemoryStore: Low-level ChromaDB vector store (optional dependency).
  - get_embeddings / is_fallback: Embedding provider utilities.
"""

from src.memory.manager import MemoryManager

__all__ = ["MemoryManager"]
