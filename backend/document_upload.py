"""
Document upload — extract plain text from a user's uploaded file so it can be
pinned to their session and explained against the retrieved law.
"""

import io
import logging
import unicodedata

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
# Kept in the session and sent to the LLM; long contracts are truncated.
MAX_DOCUMENT_CHARS = 12000
ALLOWED_EXTENSIONS = (".pdf", ".txt", ".md")


class DocumentError(ValueError):
    """Upload can't be used; the message is safe to show the user."""


def extract_text(filename: str, data: bytes) -> str:
    """Return the document's text, normalized and capped at MAX_DOCUMENT_CHARS."""
    name = filename.lower()
    if not name.endswith(ALLOWED_EXTENSIONS):
        raise DocumentError("Only PDF, TXT and MD files are supported.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentError("File is larger than 5 MB.")

    if name.endswith(".pdf"):
        text = _pdf_text(data)
    else:
        text = data.decode("utf-8", errors="replace")

    # NFKC folds Arabic presentation forms (common in PDF output) back to normal letters.
    text = unicodedata.normalize("NFKC", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        raise DocumentError(
            "No text could be read from this file. Scanned PDFs (images) aren't supported."
        )

    if len(text) > MAX_DOCUMENT_CHARS:
        logger.info(f"[upload] truncating {filename}: {len(text)} → {MAX_DOCUMENT_CHARS} chars")
        text = text[:MAX_DOCUMENT_CHARS]
    return text


def _pdf_text(data: bytes) -> str:
    import pypdf

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.warning(f"[upload] PDF parse failed: {e}")
        raise DocumentError("This PDF could not be read.") from e
