"""
Embedding Generator — generates vector embeddings for text chunks.

Dev A owns this file.
"""

import logging
from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level model (loaded once)
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        logger.info(f"[embed] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("[embed] Model loaded successfully")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding.
        show_progress: Show tqdm progress bar.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = get_model()
    logger.info(f"[embed] Embedding {len(texts)} texts (batch_size={batch_size})")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )

    logger.info(f"[embed] Done. Embedding dim: {len(embeddings[0])}")
    return [e.tolist() for e in embeddings]


def embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
