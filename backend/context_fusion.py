"""
Context Fusion — merge, deduplicate, and cap retrieved chunks within token budget.
Owned by Dev A.
"""

import logging
from backend.config import TOKEN_BUDGET
from backend.models import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 chars per token for English, ~2-3 for Arabic.
    Uses a conservative average of 3 chars/token.
    """
    return max(1, len(text) // 3)


def fuse_context(retrieval: RetrievalResult, token_budget: int = TOKEN_BUDGET) -> RetrievalResult:
    """
    Merge + deduplicate + cap retrieved chunks to fit within the token budget.

    Steps:
    1. Deduplicate by article_id (prefer higher-scored version)
    2. Sort by score (descending)
    3. Cap to token budget

    Args:
        retrieval: Raw retrieval result from combined vector + graph search.
        token_budget: Maximum tokens for the context window.

    Returns:
        A new RetrievalResult with deduplicated, capped chunks.
    """
    # Step 1: Deduplicate by article_id, keeping highest-scored version
    seen: dict[str, RetrievedChunk] = {}
    for chunk in retrieval.chunks:
        key = f"{chunk.law_name}:{chunk.article_id}"
        if key not in seen or chunk.score > seen[key].score:
            seen[key] = chunk

    unique_chunks = list(seen.values())
    logger.info(f"[context_fusion] deduped: {len(retrieval.chunks)} → {len(unique_chunks)} chunks")

    # Step 2: Sort by score descending
    unique_chunks.sort(key=lambda c: c.score, reverse=True)

    # Step 3: Cap to token budget
    capped_chunks = []
    total_tokens = 0

    for chunk in unique_chunks:
        chunk_tokens = _estimate_tokens(chunk.text)
        if total_tokens + chunk_tokens > token_budget:
            logger.info(f"[context_fusion] token budget reached at {len(capped_chunks)} chunks ({total_tokens} tokens)")
            break
        capped_chunks.append(chunk)
        total_tokens += chunk_tokens

    logger.info(f"[context_fusion] final: {len(capped_chunks)} chunks, ~{total_tokens} tokens")

    return RetrievalResult(
        chunks=capped_chunks,
        graph_path=retrieval.graph_path,
        vector_top_score=retrieval.vector_top_score,
        graph_corroboration_count=retrieval.graph_corroboration_count,
    )
