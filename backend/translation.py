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


async def rewrite_for_retrieval(text: str) -> str | None:
    """
    Rewrite a question (any language) in the wording Egyptian statutes use, as an
    extra retrieval query. People say "موظف عام يقبل رشوة"; the Penal Code says
    "موظف عمومي ... أخذ عطية ... يعد مرتشياً" — neither search method bridges that.
    Returns None on failure (retrieval then uses the original query only).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's legal question about Egyptian law as one short Arabic sentence "
                "using the exact wording Egyptian statutes use (e.g. 'موظف عمومي' not 'موظف عام', "
                "'عطية' and 'مرتشياً' for bribery, 'إنهاء عقد العمل' for dismissal). "
                "Do not answer the question. Return ONLY the rewritten sentence."
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        result = (await call_llm_cheap(messages, temperature=0.1, max_tokens=120)).strip()
        logger.info(f"[translation] retrieval rewrite: '{result[:80]}'")
        return result or None
    except Exception as e:
        logger.warning(f"[translation] retrieval rewrite failed: {e}")
        return None


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


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    async def test():
        print("Testing Translator:")
        ar_res, was_translated_1 = await ensure_arabic_query("ما عقوبة السرقة")
        print(f"Arabic input -> Output: '{ar_res}', Translated: {was_translated_1}")
        
        en_res, was_translated_2 = await ensure_arabic_query("What is the penalty for theft in Egypt?")
        print(f"English input -> Output: '{en_res}', Translated: {was_translated_2}")
    asyncio.run(test())
