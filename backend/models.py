"""
Shared data models used across all pipeline components.
This is the agreed-upon Retriever interface contract (Task 0.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    """Planner output — what kind of request is this?"""
    QA = "qa"
    DOCUMENT_EXPLANATION = "document_explanation"
    CASE_GUIDANCE = "case_guidance"
    OFF_TOPIC = "off_topic"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CitationCheckOutcome(str, Enum):
    PASSED = "passed"             # all citations grounded on first try
    PATCHED = "patched"           # some ungrounded claims stripped
    RETRIED = "retried"           # regenerated once with stricter prompt
    FALLBACK = "fallback"         # could not ground — raw sources shown


# ── Retriever Interface (agreed in Phase 0) ──────────────────────────────

@dataclass
class RetrievedChunk:
    """A single chunk returned by Qdrant or Neo4j."""
    text: str
    law_name: str
    article_id: str
    categories: list[str] = field(default_factory=list)
    source: str = "qdrant"        # "qdrant" | "neo4j"
    score: float = 0.0            # similarity score 0-1


@dataclass
class GraphPathNode:
    """One node in the Neo4j traversal path."""
    node_id: str
    label: str                    # e.g. "Law", "Article"
    name: str                     # e.g. "قانون العقوبات", "المادة 318"
    relationship: str = ""        # edge label to next node, empty for last


@dataclass
class RetrievalResult:
    """The full output of the retriever — vector + graph combined."""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    graph_path: list[GraphPathNode] = field(default_factory=list)
    vector_top_score: float = 0.0
    graph_corroboration_count: int = 0  # overlap between vector & graph hits


# ── Pipeline I/O ─────────────────────────────────────────────────────────

@dataclass
class ChatRequest:
    """Incoming request from the frontend."""
    message: str
    session_id: str = ""
    language: str = "ar"          # "ar" | "en"


@dataclass
class Citation:
    """A single citation attached to the response."""
    law_name: str
    article_id: str
    text_snippet: str = ""


@dataclass
class ChatResponse:
    """Outgoing response to the frontend."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    graph_path: list[GraphPathNode] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    citation_check_outcome: CitationCheckOutcome = CitationCheckOutcome.PASSED
    session_id: str = ""
    language: str = "ar"
    disclaimer: str = (
        "هذا المساعد يقدم معلومات قانونية عامة ولا يُغني عن استشارة محامٍ مختص. "
        "This assistant provides general legal information and is not a substitute "
        "for professional legal advice."
    )


# ── Session ──────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """A single conversation turn."""
    role: str       # "user" | "assistant"
    content: str


@dataclass
class Session:
    """Session state for one user conversation."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    pinned_document: Optional[str] = None   # uploaded doc content, if any