"""
Keyword search (BM25) over the law articles — the lexical half of hybrid retrieval.

The small multilingual embedding model often misses articles that share the
query's exact legal terms (e.g. "إجازة سنوية" → Labor Law art. 125). BM25 over
lightly normalized Arabic catches those; the retriever merges both result lists.

The index is built in memory from backend/data/laws/*.md (the same files and
parser as ingestion), so article ids match Qdrant/Neo4j exactly.
"""

import logging
import math
import re
from collections import Counter
from pathlib import Path

from .models import RetrievedChunk

logger = logging.getLogger(__name__)

LAWS_DIR = Path(__file__).resolve().parent / "data" / "laws"

_DIACRITICS = re.compile(r"[ً-ْـ]")  # harakat + tatweel
_TOKEN = re.compile(r"[ء-ي]+|[a-zA-Z]+|\d+")
# Attached prefixes stripped before matching (longest first): و, ف, ب, ل, ك + ال
_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال")
_STOPWORDS = {
    "في", "من", "على", "الى", "عن", "ان", "او", "ما", "لا", "هذا", "هذه", "ذلك", "التي",
    "الذي", "كل", "اي", "مع", "بين", "قد", "لم", "هل", "هو", "هي", "كان", "يكون", "به",
    "له", "لها", "بها", "عند", "اذا", "ولا", "او", "وفي", "ومن", "حتي", "غير", "بعد", "قبل",
    "المادة", "ماده", "مادة", "قانون", "القانون",
}

# BM25 parameters (standard defaults)
_K1 = 1.5
_B = 0.75


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    return text.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(normalize_arabic(text).lower()):
        for p in _PREFIXES:
            if tok.startswith(p) and len(tok) - len(p) >= 3:
                tok = tok[len(p):]
                break
        # Accusative tanween alef: "ضررا" / "تعويضا" -> "ضرر" / "تعويض"
        if len(tok) >= 4 and tok.endswith("ا"):
            tok = tok[:-1]
        if tok not in _STOPWORDS and len(tok) > 1:
            tokens.append(tok)
    return tokens


class KeywordIndex:
    def __init__(self, records: list[dict]):
        self.records = records
        self.doc_tokens = [Counter(tokenize(r["text"])) for r in records]
        self.doc_len = [sum(c.values()) for c in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        df = Counter(t for c in self.doc_tokens for t in c)
        n = len(records)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        q_tokens = set(tokenize(query))
        scores = []
        for i, counts in enumerate(self.doc_tokens):
            s = 0.0
            for t in q_tokens:
                tf = counts.get(t)
                if tf:
                    norm = _K1 * (1 - _B + _B * self.doc_len[i] / self.avg_len)
                    s += self.idf[t] * tf * (_K1 + 1) / (tf + norm)
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.records[i], s) for s, i in scores[:top_k]]


_index: KeywordIndex | None = None


def get_index() -> KeywordIndex:
    """Lazy-build the index from the canonical law files (~1-2s)."""
    global _index
    if _index is None:
        from .ingestion.normalize import normalize_md_document

        records = []
        for md in sorted(LAWS_DIR.glob("*.md")):
            if md.name == "index.md":
                continue
            records.extend(r for r in normalize_md_document(str(md), md.read_text(encoding="utf-8")) if r.get("text"))
        _index = KeywordIndex(records)
        logger.info(f"[keyword_search] indexed {len(records)} articles")
    return _index


def keyword_search(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """BM25 search; RetrievedChunk.score is the BM25 score (not comparable to cosine scores)."""
    return [
        RetrievedChunk(
            text=r["text"],
            law_name=r["law_name"],
            article_id=r["article_id"],
            categories=r.get("categories", []),
            source="keyword",
            score=s,
        )
        for r, s in get_index().search(query, top_k)
    ]
