"""
Citation Check — deterministic grounding verification.
NOT an LLM call. Checks whether each (law, article) cited in the response
actually appears in the retrieved context.
"""

import re
import logging
from .models import (
    RetrievalResult,
    RetrievedChunk,
    Citation,
    CitationCheckOutcome,
)
from .ingestion.normalize import canonical_article_id

logger = logging.getLogger(__name__)

# ── Regex patterns to extract citations from LLM response ────────────────

# One article id: digits (any script), optionally "مكرر" and/or a letter suffix.
# A bare letter only counts as a suffix right after "مكرر" ("86 مكرر ب");
# otherwise it must be bracketed ("77 (و)"), so the "و" in "311 و 318" stays "and".
_ID = (
    r"[\d٠-٩۰-۹]+"
    r"(?:\s*مكرر(?:اً|ا|ًا)?(?:\s+[أ-ي]ـ?(?=[\s.,،؛:)]|$))?)?"
    r"(?:\s*\([أ-ي]ـ?\))?"
)

# "المادة 318", "مادة (122)", "المادتين 311 و 318", "المواد 311، 312 و313"
_AR_CITATION = re.compile(
    rf"(?:ال)?(?:مادة|مادتين|مادتان|مواد)\s*(?:من\s+)?\(?\s*((?:{_ID})(?:\s*\)?\s*(?:،|,|و|-|إلى)\s*\(?\s*(?:{_ID}))*)",
    re.UNICODE,
)
# "Article 318", "Articles 311 and 318"
_EN_CITATION = re.compile(
    r"[Aa]rticles?\s+(\d+(?:\s*bis)?(?:\s*(?:,|and|-|to)\s*\d+(?:\s*bis)?)*)",
)
_ID_RE = re.compile(_ID)
# "المادة 3 من العقد" / "Article 3 of the contract" points into an uploaded document.
_DOCUMENT_REF_RE = re.compile(
    r"\s*\)?\s*(?:من|في)\s+(?:هذا\s+|هذه\s+)?(?:العقد|الوثيقة|المستند|الاتفاق|الاتفاقية)"
    r"|\s*(?:of|in)\s+the\s+(?:contract|document|agreement)",
    re.IGNORECASE,
)
_EN_ID_RE = re.compile(r"\d+(?:\s*bis)?")

# Words that name a specific Egyptian code. Used to tell which law a citation
# points at, and to catch citations of a law that was never retrieved.
LAW_KEYWORDS = [
    "العقوبات", "المدني", "العمل", "التجارة", "الإجراءات الجنائية", "الجنائية",
    "الضريبة", "المرافعات", "الأحوال الشخصية", "الدستور", "الإثبات",
    "Penal", "Civil", "Labor", "Labour", "Commercial", "Criminal Procedure", "Tax",
]
_LAW_KEYWORD_RE = re.compile("|".join(re.escape(k) for k in LAW_KEYWORDS))
_EN_TO_AR = {
    "Penal": "العقوبات", "Civil": "المدني", "Labor": "العمل", "Labour": "العمل",
    "Commercial": "التجارة", "Criminal Procedure": "الجنائية", "Tax": "الضريبة",
}


def _law_mentioned_near(text: str, start: int, end: int) -> str | None:
    """The law keyword closest to a citation: first look just after it ("المادة 318 من قانون العقوبات"), then just before."""
    after = _LAW_KEYWORD_RE.search(text[end:end + 80])
    if after:
        return after.group(0)
    before = list(_LAW_KEYWORD_RE.finditer(text[max(0, start - 80):start]))
    return before[-1].group(0) if before else None


def _law_matches(keyword: str, law_name: str) -> bool:
    return _EN_TO_AR.get(keyword, keyword) in law_name or keyword in law_name


def extract_citations_from_response(response_text: str) -> list[Citation]:
    """
    Extract all article citations from the LLM's response text.
    law_name is set to the law keyword found next to the citation ("" if none).
    """
    citations = []
    seen = set()

    matches = [(m, _ID_RE) for m in _AR_CITATION.finditer(response_text)]
    matches += [(m, _EN_ID_RE) for m in _EN_CITATION.finditer(response_text)]
    for match, id_re in matches:
        if _DOCUMENT_REF_RE.match(response_text, match.end()):
            continue  # a clause of the user's uploaded document, not a law

        law_keyword = _law_mentioned_near(response_text, match.start(), match.end()) or ""
        for id_match in id_re.finditer(match.group(1)):
            article_id = canonical_article_id(id_match.group(0).replace("bis", "مكرر"))
            key = (law_keyword, article_id)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(law_name=law_keyword, article_id=article_id))

    logger.info(f"[citation_check] extracted {len(citations)} citations from response")
    return citations


def _find_chunk(citation: Citation, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    """
    The retrieved chunk this citation refers to, or None if it isn't grounded.
    - Citation names a law -> that law's article must have been retrieved.
    - Citation names no law -> grounded only if exactly one retrieved law has that article
      (otherwise we can't tell which one the model meant).
    """
    same_id = [c for c in chunks if c.article_id == citation.article_id]
    if citation.law_name:
        return next((c for c in same_id if _law_matches(citation.law_name, c.law_name)), None)
    return same_id[0] if len({c.law_name for c in same_id}) == 1 else None


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

    grounded = []
    ungrounded = []
    grounded_keys = set()
    for citation in extracted:
        chunk = _find_chunk(citation, retrieval.chunks)
        if chunk:
            # The same article may be cited with and without its law name — keep one.
            if (chunk.law_name, chunk.article_id) in grounded_keys:
                continue
            grounded_keys.add((chunk.law_name, chunk.article_id))
            citation.law_name = chunk.law_name
            citation.text_snippet = chunk.text[:200]
            grounded.append(citation)
        else:
            ungrounded.append(citation)

    logger.info(
        f"[citation_check] grounded={len(grounded)} ungrounded={len(ungrounded)}"
        + (f" ungrounded_ids={[(c.law_name, c.article_id) for c in ungrounded]}" if ungrounded else "")
    )
    return extracted, grounded, ungrounded


def strip_ungrounded_claims(
    response_text: str,
    ungrounded: list[Citation],
) -> str:
    """
    Drop the lines that contain an ungrounded citation.
    Returns the cleaned text, or empty string if too much was removed.
    """
    bad = {(c.law_name, c.article_id) for c in ungrounded}
    cleaned_lines = []

    for line in response_text.split("\n"):
        line_citations = extract_citations_from_response(line)
        if not any((c.law_name, c.article_id) in bad for c in line_citations):
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    # Check if the remaining text is coherent (at least 30% of original)
    if len(cleaned) < len(response_text) * 0.3:
        logger.warning("[citation_check] too much removed during stripping — text incoherent")
        return ""

    return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Citation Check:")
    response_text = (
        "وفقاً للمادة 318 من قانون العقوبات، يعاقب السارق بالحبس.\n"
        "كما تنص المادة (122) من قانون العمل على الإجازة.\n"
        "والمادة 5 من القانون المدني تتحدث عن أمر آخر."
    )
    chunks = [
        RetrievedChunk(article_id="318", law_name="كود قانون العقوبات المصري", text="يعاقب بالحبس مع الشغل...", score=0.9),
        RetrievedChunk(article_id="122", law_name="قانون رقم ١٤ لسنة ٢٠٢٥ بإصدار قانون العمل", text="للعامل الحق في إجازة...", score=0.8),
        RetrievedChunk(article_id="5", law_name="قانون رقم ١٤ لسنة ٢٠٢٥ بإصدار قانون العمل", text="...", score=0.7),
    ]
    extracted, grounded, ungrounded = check_citations(response_text, RetrievalResult(chunks=chunks))
    print(f"Extracted: {[(c.law_name, c.article_id) for c in extracted]}")
    print(f"Grounded: {[c.article_id for c in grounded]}  (expected 318, 122)")
    print(f"Ungrounded: {[(c.law_name, c.article_id) for c in ungrounded]}  (expected المدني 5)")
    print("Stripped:\n" + strip_ungrounded_claims(response_text, ungrounded))
