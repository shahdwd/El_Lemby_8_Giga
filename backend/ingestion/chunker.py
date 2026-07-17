"""
Text Chunker — splits long articles into ~512-token chunks with overlap.
Preserves metadata across chunks.

Dev A owns this file.
"""

import logging
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_document(record: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a normalized record into chunks.

    Each chunk inherits the original record's metadata (law_name, article_id, categories)
    and gets a unique chunk_id.

    Args:
        record: Normalized record with {law_name, article_id, categories, text}.
        chunk_size: Max characters per chunk (~512 tokens ≈ ~1500 chars for Arabic).
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk dicts, each with the original metadata + chunk-specific fields.
    """
    text = record.get("text", "")
    char_limit = chunk_size * 3  # rough: 1 token ≈ 3 chars for Arabic text

    if len(text) <= char_limit:
        # Text fits in one chunk — no splitting needed
        return [{
            **record,
            "chunk_id": f"{record['article_id']}_0",
            "chunk_index": 0,
            "total_chunks": 1,
        }]

    # Split into chunks with overlap
    chunks = []
    start = 0
    chunk_index = 0
    overlap_chars = overlap * 3

    while start < len(text):
        end = start + char_limit

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation near the boundary
            search_window = text[max(0, end - 200):end]
            for sep in [".", "。", "،", "\n", "؛", ";"]:
                last_sep = search_window.rfind(sep)
                if last_sep != -1:
                    end = max(0, end - 200) + last_sep + 1
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                **record,
                "text": chunk_text,
                "chunk_id": f"{record['article_id']}_{chunk_index}",
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        start = end - overlap_chars if end < len(text) else len(text)

    # Add total_chunks count to each chunk
    for c in chunks:
        c["total_chunks"] = len(chunks)

    logger.info(
        f"[chunker] article {record['article_id']}: "
        f"{len(text)} chars → {len(chunks)} chunks"
    )
    return chunks


def chunk_all(records: list[dict]) -> list[dict]:
    """Chunk all normalized records."""
    all_chunks = []
    for record in records:
        all_chunks.extend(chunk_document(record))
    logger.info(f"[chunker] total: {len(records)} records → {len(all_chunks)} chunks")
    return all_chunks
