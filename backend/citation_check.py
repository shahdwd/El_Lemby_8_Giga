"""
Citation Check — deterministic grounding verification.
NOT an LLM call. Checks whether cited article IDs in the response
actually appear in the retrieved context.
"""

import re
import logging
from .models import (
    RetrievalResult,
    RetrievedChunk,
    Citation,
    CitationCheckOutcome,
)

logger = logging.getLogger(__name__)

# ── Regex patterns to extract citations from LLM response ────────────────

# Matches patterns like: "المادة 318", "مادة 15", "Article 318"
CITATION_PATTERNS = [
    re.compile(r"(?:ال)?مادة\s+(\d+(?:\s*[\u0660-\u0669]+)?(?:\s*(?:مكرر|بند|فقرة)\s*[\u0621-\u064A]*)?)", re.UNICODE),
    re.compile(r"[Aa]rticle\s+(\d+(?:\s*(?:bis|paragraph|clause)\s*\w*)?)", re.IGNORECASE),
]

# Matches law names like "قانون العقوبات", "القانون المدني"
LAW_NAME_PATTERNS = [
    re.compile(r"(قانون\s+[\u0621-\u064A]+(?:\s+[\u0621-\u064A]+)*)", re.UNICODE),
    re.compile(r"(القانون\s+[\u0621-\u064A]+(?:\s+[\u0621-\u064A]+)*)", re.UNICODE),
]


def extract_citations_from_response(response_text: str) -> list[Citation]:
    """Extract all article citations from the LLM's response text."""
    citations = []
    seen = set()

    for pattern in CITATION_PATTERNS:
        for match in pattern.finditer(response_text):
            article_id = match.group(1).strip()

            # Try to find the associated law name nearby (within ~100 chars before the match)
            start = max(0, match.start() - 150)
            context_window = response_text[start:match.end()]
            law_name = ""
            for lp in LAW_NAME_PATTERNS:
                law_match = lp.search(context_window)
                if law_match:
                    law_name = law_match.group(1).strip()
                    break

            key = (law_name, article_id)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(
                    law_name=law_name,
                    article_id=article_id,
                    text_snippet="",
                ))

    logger.info(f"[citation_check] extracted {len(citations)} citations from response")
    return citations


def check_citations(
    response_text: str,
    retrieval: RetrievalResult,
) -> tuple[list[Citation], list[Citation], list[Citation]]:
    """
    Check if citations in the response are grounded in the retrieved context.

    Returns:
        (all_citations, grounded_citations, ungrounded_citations)
    """
    extracted = extract_citations_from_response(response_text)

    # Build set of article IDs from retrieved chunks
    retrieved_article_ids = set()
    for chunk in retrieval.chunks:
        # Normalize: strip whitespace, store both raw and cleaned
        aid = chunk.article_id.strip()
        retrieved_article_ids.add(aid)
        # Also add just the numeric part for fuzzy matching
        nums = re.findall(r"\d+", aid)
        for n in nums:
            retrieved_article_ids.add(n)

    grounded = []
    ungrounded = []

    for citation in extracted:
        aid = citation.article_id.strip()
        # Check if the article ID (or its numeric part) exists in retrieved context
        nums = re.findall(r"\d+", aid)
        is_grounded = (
            aid in retrieved_article_ids
            or any(n in retrieved_article_ids for n in nums)
        )

        if is_grounded:
            # Find the matching chunk to get the text snippet
            for chunk in retrieval.chunks:
                chunk_nums = re.findall(r"\d+", chunk.article_id)
                if aid == chunk.article_id.strip() or any(n in chunk_nums for n in nums):
                    citation.text_snippet = chunk.text[:200]
                    citation.law_name = citation.law_name or chunk.law_name
                    break
            grounded.append(citation)
        else:
            ungrounded.append(citation)

    logger.info(
        f"[citation_check] grounded={len(grounded)} ungrounded={len(ungrounded)}"
    )
    return extracted, grounded, ungrounded


def strip_ungrounded_claims(
    response_text: str,
    ungrounded: list[Citation],
) -> str:
    """
    Attempt to strip sentences containing ungrounded citations.
    Returns the cleaned text, or empty string if too much was removed.
    """
    lines = response_text.split("\n")
    cleaned_lines = []

    for line in lines:
        contains_bad_citation = False
        for citation in ungrounded:
            if citation.article_id in line:
                contains_bad_citation = True
                break
        if not contains_bad_citation:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    # Check if the remaining text is coherent (at least 30% of original)
    if len(cleaned) < len(response_text) * 0.3:
        logger.warning("[citation_check] too much removed during stripping — text incoherent")
        return ""

    return cleaned


def determine_outcome(
    all_citations: list[Citation],
    grounded: list[Citation],
    ungrounded: list[Citation],
) -> CitationCheckOutcome:
    """Determine the citation check outcome."""
    if not all_citations:
        # No citations found — could be a general response, treat as passed
        return CitationCheckOutcome.PASSED
    if not ungrounded:
        return CitationCheckOutcome.PASSED
    return CitationCheckOutcome.FALLBACK  # will be updated by pipeline logic


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Citation Check:")
    response_text = "وفقاً للمادة 318 من قانون العقوبات، يعاقب السارق بالحبس."
    chunk = RetrievedChunk(article_id="318", law_name="قانون العقوبات", text="يعاقب بالحبس مع الشغل...", score=0.9)
    mock_retrieval = RetrievalResult(chunks=[chunk], graph_path=[])
    
    extracted, grounded, ungrounded = check_citations(response_text, mock_retrieval)
    print(f"Extracted: {len(extracted)}, Grounded: {len(grounded)}, Ungrounded: {len(ungrounded)}")
    outcome = determine_outcome(extracted, grounded, ungrounded)
    print(f"Outcome: {outcome.value}")
