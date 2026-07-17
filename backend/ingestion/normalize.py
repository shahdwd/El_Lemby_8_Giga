"""
Schema Normalization — normalize HF datasets AND the canonical
Egyptian-law .md format (YAML frontmatter + "# ARTICLE <n>" blocks)
to the shared schema: law_name, article_id, categories, text.

Dev A owns this file.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_record(record: dict, dataset_name: str) -> dict | None:
    """
    Normalize a single record from any HF dataset to the shared schema.

    Expected output:
    {
        "law_name": str,
        "article_id": str,
        "categories": list[str],
        "text": str,
    }

    Dev A: Update the field mappings below based on the actual HF dataset schema.
    """

    try:
        if dataset_name == "egypt-legal-corpus":
            # TODO (Dev A): Inspect actual column names and update these mappings
            return {
                "law_name": record.get("law_name", record.get("title", "")),
                "article_id": str(record.get("article_id", record.get("article_number", ""))),
                "categories": _extract_categories(record),
                "text": record.get("text", record.get("content", "")),
            }

        elif dataset_name == "QA_LAW_Egyptian_dataset":
            # This is the eval set — normalize Q&A pairs
            return {
                "law_name": record.get("law_name", ""),
                "article_id": str(record.get("article_id", "")),
                "categories": _extract_categories(record),
                "text": record.get("question", "") + "\n" + record.get("answer", ""),
            }

        else:
            logger.warning(f"Unknown dataset: {dataset_name}")
            return None

    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return None


def _extract_categories(record: dict) -> list[str]:
    """Extract categories from various possible field names."""
    cats = record.get("categories", record.get("category", record.get("topic", [])))
    if isinstance(cats, str):
        return [c.strip() for c in cats.split(",")]
    if isinstance(cats, list):
        return cats
    return []


# ── Digit normalization ──────────────────────────────────────────────
# Article numbers in the canonical .md files are Arabic-Indic numerals
# (١٢٣...). We convert these to plain ASCII so article_id is stable and
# matches cleanly across the md path and the HF-dataset path (which
# already uses ASCII ids). This also absorbs a couple of digit-encoding
# corruptions observed in real files (e.g. "٨" mis-rendered as the CJK
# character "八", "٢" mis-rendered as the fullwidth "２") rather than
# silently dropping or mis-splitting those articles.

_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"        # U+0660–U+0669
_EXTENDED_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"  # U+06F0–U+06F9 (Persian/Urdu variant)
_FULLWIDTH = "０１２３４５６７８９"    # U+FF10–U+FF19

_DIGIT_MAP: dict[str, str] = {}
for _western, _arabic, _ext, _full in zip("0123456789", _ARABIC_INDIC, _EXTENDED_ARABIC_INDIC, _FULLWIDTH):
    _DIGIT_MAP[_arabic] = _western
    _DIGIT_MAP[_ext] = _western
    _DIGIT_MAP[_full] = _western

# Known corruption fallback: a single Arabic-Indic digit occasionally
# gets mangled into a CJK numeral somewhere upstream (encoding mixup).
# We fix it but log loudly, since this is a data-quality bug worth
# tracking down at the source, not a format we should treat as normal.
_CJK_DIGIT_FALLBACK = {
    "〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}


def normalize_digits(s: str) -> str:
    """Convert Arabic-Indic / extended-Arabic-Indic / fullwidth digits to ASCII."""
    out = []
    for ch in s:
        if ch in _DIGIT_MAP:
            out.append(_DIGIT_MAP[ch])
        elif ch in _CJK_DIGIT_FALLBACK:
            logger.warning(
                f"[normalize] digit-corruption fallback used: {ch!r} -> "
                f"{_CJK_DIGIT_FALLBACK[ch]!r} (check source file encoding)"
            )
            out.append(_CJK_DIGIT_FALLBACK[ch])
        else:
            out.append(ch)
    return "".join(out)


# ── Canonical .md parsing ────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ARTICLE_HEADER_RE = re.compile(r"^#\s+ARTICLE\s+(.+?)\s*$", re.MULTILINE)
_CHAPTER_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_TEXT_SECTION_RE = re.compile(r"##\s+TEXT\s*\n(.*?)(?=##\s+REFERENCES|\Z)", re.DOTALL)
_REFERENCES_SECTION_RE = re.compile(r"##\s+REFERENCES\s*\n(.*?)\Z", re.DOTALL)


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split off the YAML-style frontmatter block, return (metadata, body)."""
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    meta = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, content[match.end():]


def normalize_md_document(filepath: str, content: str) -> list[dict]:
    """
    Parse a canonical Egyptian-law markdown file into article-level records.

    Expected shape:
        ---
        law_name: ...
        law_number: ...
        law_year: ...
        ---
        # ARTICLE ١
        ## TEXT
        <article text>
        ## REFERENCES
        - <ref>
        (or "None")

    Known quirk this handles: a chapter heading ("### الفصل ...") is
    sometimes emitted *inside* the article block that precedes the new
    chapter — sitting between that article's TEXT and REFERENCES —
    even though it semantically opens the chapter starting at the
    *next* article. We detect it and attach it forward instead of to
    the block it's textually embedded in, and keep applying it to
    subsequent articles until a new chapter heading is found.

    Returns a list of dicts with the shared schema plus:
        - law_number, law_year (from frontmatter, if present)
        - references: list[str] (cross-references parsed from the
          REFERENCES section — useful for Neo4j REFERENCES edges)
    """
    meta, body = _parse_frontmatter(content)
    law_name = meta.get("law_name") or Path(filepath).stem.replace("_", " ")
    law_number = normalize_digits(meta["law_number"]) if meta.get("law_number") else None
    law_year = normalize_digits(meta["law_year"]) if meta.get("law_year") else None

    headers = list(_ARTICLE_HEADER_RE.finditer(body))
    if not headers:
        logger.warning(f"[normalize] no '# ARTICLE' headers found in {filepath}; treating whole file as one record")
        return [{
            "law_name": law_name,
            "law_number": law_number,
            "law_year": law_year,
            "article_id": "0",
            "categories": [],
            "text": body.strip(),
            "references": [],
        }]

    records = []
    current_chapter: str | None = None

    for i, header_match in enumerate(headers):
        article_id = normalize_digits(header_match.group(1))

        block_start = header_match.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        block = body[block_start:block_end]

        # Pull out any chapter heading embedded in this block. It belongs
        # to the *next* article's chapter, not this one — see docstring.
        chapter_match = _CHAPTER_RE.search(block)
        next_chapter = chapter_match.group(1).strip() if chapter_match else None
        if chapter_match:
            block = block[:chapter_match.start()] + block[chapter_match.end():]

        text_match = _TEXT_SECTION_RE.search(block)
        refs_match = _REFERENCES_SECTION_RE.search(block)

        article_text = text_match.group(1).strip() if text_match else block.strip()
        # Clean up any stray "---" section-break left behind after removing
        # an embedded chapter heading.
        article_text = re.sub(r"\n?-{3,}\n?", "\n", article_text).strip()

        references: list[str] = []
        if refs_match:
            raw_refs = refs_match.group(1).strip()
            if raw_refs and raw_refs != "None":
                references = [
                    line.lstrip("- ").strip()
                    for line in raw_refs.splitlines()
                    if line.strip().startswith("-")
                ]

        if not article_text:
            logger.warning(f"[normalize] empty TEXT for article {article_id} in {filepath}")

        records.append({
            "law_name": law_name,
            "law_number": law_number,
            "law_year": law_year,
            "article_id": article_id,
            "categories": [current_chapter] if current_chapter else [],
            "text": article_text,
            "references": references,
        })

        if next_chapter:
            current_chapter = next_chapter

    logger.info(f"[normalize] parsed {filepath}: {len(records)} articles, law='{law_name}'")
    return records