"""
Translation — detects query language and translates to Arabic for retrieval if needed.
Also handles response language alignment.
"""

import logging
from backend.llm_client import call_llm_cheap

logger = logging.getLogger(__name__)


def is_arabic(text: str) -> bool:
    """Quick heuristic: check if the majority of characters are Arabic."""
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F")
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return True  # default to Arabic for non-alpha input
    return (arabic_chars / total_alpha) > 0.5


async def translate_to_arabic(text: str) -> str:
    """Translate a non-Arabic query to Arabic for retrieval."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a translator. Translate the following text to Arabic. "
                "This is a legal query about Egyptian law. Preserve legal terminology accurately. "
                "Return ONLY the Arabic translation, nothing else."
            ),
        },
        {"role": "user", "content": text},
    ]
    result = await call_llm_cheap(messages, temperature=0.1, max_tokens=256)
    logger.info(f"[translation] en→ar: '{text[:50]}...' → '{result[:50]}...'")
    return result.strip()


async def ensure_arabic_query(query: str) -> tuple[str, bool]:
    """
    Ensure the query is in Arabic for retrieval.

    Returns:
        (arabic_query, was_translated)
    """
    if is_arabic(query):
        return query, False
    translated = await translate_to_arabic(query)
    return translated, True
