"""
Confidence score — derived from retrieval signals, NOT self-reported LLM confidence.
Outputs a 3-tier label: High / Medium / Low.
"""

import logging
from models import (
    RetrievalResult,
    CitationCheckOutcome,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────────────────────
VECTOR_SCORE_HIGH = 0.75
VECTOR_SCORE_LOW = 0.45
GRAPH_CORROBORATION_HIGH = 2  # at least 2 graph hits overlap with vector


def compute_confidence(
    retrieval: RetrievalResult,
    citation_outcome: CitationCheckOutcome,
) -> ConfidenceLevel:
    """
    Derive confidence from retrieval signals:
    - Qdrant similarity score of top hit
    - Number of corroborating graph hits (vector ∩ graph overlap)
    - Citation check outcome (passed / patched / retried / fallback)

    Returns:
        ConfidenceLevel.HIGH / MEDIUM / LOW
    """
    score = 0.0

    # ── Factor 1: Vector similarity (0–40 points) ────────────────────
    if retrieval.vector_top_score >= VECTOR_SCORE_HIGH:
        score += 40
    elif retrieval.vector_top_score >= VECTOR_SCORE_LOW:
        score += 20
    else:
        score += 5

    # ── Factor 2: Graph corroboration (0–30 points) ──────────────────
    if retrieval.graph_corroboration_count >= GRAPH_CORROBORATION_HIGH:
        score += 30
    elif retrieval.graph_corroboration_count >= 1:
        score += 15
    else:
        score += 0

    # ── Factor 3: Citation check outcome (0–30 points) ───────────────
    match citation_outcome:
        case CitationCheckOutcome.PASSED:
            score += 30
        case CitationCheckOutcome.PATCHED:
            score += 15
        case CitationCheckOutcome.RETRIED:
            score += 10
        case CitationCheckOutcome.FALLBACK:
            score += 0

    # ── Map to tier ──────────────────────────────────────────────────
    if score >= 70:
        level = ConfidenceLevel.HIGH
    elif score >= 40:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    logger.info(f"[confidence] score={score} level={level.value} "
                f"(vector={retrieval.vector_top_score:.2f}, "
                f"graph_corr={retrieval.graph_corroboration_count}, "
                f"citation={citation_outcome.value})")
    return level


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Confidence Check:")
    mock_retrieval = RetrievalResult(chunks=[], graph_path=[], vector_top_score=0.8, graph_corroboration_count=2)
    conf = compute_confidence(mock_retrieval, CitationCheckOutcome.PASSED)
    print(f"Calculated Confidence: {conf.value} (Expected: high)")
