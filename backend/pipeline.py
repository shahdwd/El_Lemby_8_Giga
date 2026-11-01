"""
Pipeline Orchestrator — the core of the system.
Plain Python async function, no framework:

  Planner → Research agent (LLM + tools) → Citation Check → Confidence → Response

The agent decides what to look up; the checks after it are deterministic and it
can't skip them. If the agent fails, the fixed pipeline answers instead:

  Translation → Retriever → Context Fusion → LLM
"""

import asyncio
import logging
from backend.agent import run_agent
from backend.answer_verifier import verify_answer
from backend.config import AGENT_ENABLED
from backend.models import (
    AgentStep,
    ChatResponse,
    Citation,
    ConfidenceLevel,
    CitationCheckOutcome,
    GraphPathNode,
    Intent,
    RetrievalResult,
)
from backend.planner import classify_intent
from backend.translation import ensure_arabic_query, rewrite_for_retrieval
from backend.retriever import retrieve, article_node_name
from backend.context_fusion import fuse_context
from backend.response_generator import generate_response
from backend.citation_check import (
    check_citations,
    strip_ungrounded_claims,
    extract_citations_from_response,
)
from backend.confidence import compute_confidence
from backend.session import session_store

logger = logging.getLogger(__name__)

OFF_TOPIC_MESSAGE = {
    "ar": "أنا مساعد متخصص في القانون المصري فقط. يرجى طرح سؤال قانوني وسأساعدك بالاستناد إلى نصوص القانون.",
    "en": "I can only help with questions about Egyptian law. Please ask a legal question and I'll answer from the legal texts.",
}

UNVERIFIED_WARNING = {
    "ar": "⚠️ تنبيه: قد لا ينطبق أحد المراجع التالية تماماً على هذه الحالة، فيرجى مراجعة النص.",
    "en": "⚠️ Warning: one of the references below may not fully apply here, so please check its text.",
}


def _document_paragraphs(document: str, limit: int = 15) -> list[str]:
    """Substantive lines of an uploaded document (clauses), skipping headers and signatures."""
    return [line for line in document.splitlines() if len(line.strip()) >= 40][:limit]


def _pick_graph_path(retrieval: RetrievalResult, citations: list[Citation]) -> list[GraphPathNode]:
    """
    Show the graph path that explains the answer: prefer one passing through an
    article the answer cites, then one inside a cited law, else the strongest.
    """
    cited_articles = {article_node_name(c.law_name, c.article_id) for c in citations}
    cited_laws = {c.law_name for c in citations}
    for path in retrieval.graph_paths:
        if any(n.label == "Article" and n.name in cited_articles for n in path):
            return path
    for path in retrieval.graph_paths:
        if path and path[0].label == "Law" and path[0].name in cited_laws:
            return path
    return retrieval.graph_path


def _fallback_text(retrieval: RetrievalResult, language: str) -> str:
    """Transparent fallback: no synthesized answer, just the articles found."""
    source_list = "\n".join(
        f"• {c.law_name} — المادة {c.article_id}:\n  {c.text[:200]}"
        for c in retrieval.chunks
    )
    fallback_msg = (
        "لم أتمكن من تقديم إجابة موثقة بالكامل. "
        "فيما يلي المواد القانونية المسترجعة ذات الصلة:\n\n"
        if language == "ar"
        else "I could not generate a fully grounded answer. "
        "Here are the relevant retrieved legal articles:\n\n"
    )
    return fallback_msg + source_list


async def _fixed_answer(
    message: str,
    intent: Intent,
    language: str,
    history: list[dict] | None,
    document: str | None,
) -> tuple[str, RetrievalResult]:
    """The non-agent path: retrieve once with fixed steps, then answer."""
    # Translation and the statutory-wording rewrite are independent cheap LLM
    # calls — run them together to keep latency at one round trip.
    (arabic_query, was_translated), statutory_query = await asyncio.gather(
        ensure_arabic_query(message),
        rewrite_for_retrieval(message),
    )
    if was_translated:
        logger.info(f"[pipeline] translated query to Arabic")

    extra_queries = [statutory_query] if statutory_query else []
    if document:
        extra_queries += _document_paragraphs(document)
    raw_retrieval = await retrieve(arabic_query, intent=intent, extra_queries=extra_queries)
    logger.info(f"[pipeline] retrieved {len(raw_retrieval.chunks)} raw chunks")

    fused = fuse_context(raw_retrieval)
    logger.info(f"[pipeline] fused to {len(fused.chunks)} chunks")

    response_text = await generate_response(
        query=message,
        retrieval=fused,
        language=language,
        history=history,
        pinned_document=document,
    )
    return response_text, fused


async def handle_chat(message: str, session_id: str, language: str = "ar") -> ChatResponse:
    """
    Main pipeline orchestrator.

    Flow:
    1. Planner — classify intent; off-topic questions stop here
    2. Research agent — tool-calling loop that reads the law and answers
       (fixed retrieval pipeline if the agent fails or AGENT_ENABLED is off)
    3. Citation Check — every citation must be an article the answer's context holds
    4. Answer Verification — each cited article must say what it's cited for
       (failing 3 or 4 leads to a patch, a rewrite, then a transparent fallback)
    5. Confidence score derivation
    6. Session update
    """

    logger.info(f"[pipeline] START session={session_id} lang={language}")

    # ── 1. Planner ───────────────────────────────────────────────────
    session = session_store.get_or_create(session_id)
    has_document = session.pinned_document is not None
    intent = await classify_intent(message, has_document=has_document)
    logger.info(f"[pipeline] intent={intent.value}")

    if intent == Intent.OFF_TOPIC:
        # Nothing to retrieve or ground — answer without an LLM call.
        answer = OFF_TOPIC_MESSAGE["ar" if language == "ar" else "en"]
        session_store.add_turn(session_id, "user", message)
        session_store.add_turn(session_id, "assistant", answer)
        return ChatResponse(
            answer=answer,
            confidence=ConfidenceLevel.LOW,
            session_id=session_id,
            language=language,
        )

    # ── 2. Research + Answer ─────────────────────────────────────────
    history = session_store.get_history_for_prompt(session_id) or None
    document = session.pinned_document
    response_text = None
    agent_steps: list[AgentStep] = []
    if AGENT_ENABLED:
        try:
            agent_result = await run_agent(
                query=message,
                language=language,
                history=history,
                pinned_document=document,
                # A document's clauses say which laws apply better than the
                # question does; the agent starts with those searches done.
                document_queries=[message, *_document_paragraphs(document)] if document else None,
            )
            response_text, fused, agent_steps = agent_result.answer, agent_result.retrieval, agent_result.steps
        except Exception as e:
            logger.error(f"[pipeline] agent failed, using fixed pipeline: {e}", exc_info=True)
    if response_text is None:
        response_text, fused = await _fixed_answer(message, intent, language, history, document)

    # ── 3. Citation Check ────────────────────────────────────────────
    all_citations, grounded, ungrounded = check_citations(response_text, fused)
    citation_outcome = CitationCheckOutcome.PASSED

    if ungrounded:
        logger.warning(f"[pipeline] {len(ungrounded)} ungrounded citations detected")

        # ── Step 1: Try cheap patch (strip ungrounded claims) ────────
        patched_text = strip_ungrounded_claims(response_text, ungrounded)

        if patched_text:
            logger.info("[pipeline] citation check: PATCHED (stripped ungrounded claims)")
            response_text = patched_text
            citation_outcome = CitationCheckOutcome.PATCHED
            # Re-extract citations from patched text
            all_citations, grounded, ungrounded = check_citations(response_text, fused)
        else:
            # ── Step 2: Regenerate once with stricter prompt ─────────
            logger.info("[pipeline] citation check: regenerating with stricter prompt")
            response_text = await generate_response(
                query=message,
                retrieval=fused,
                language=language,
                history=history,
                pinned_document=document,
                is_retry=True,
            )
            all_citations, grounded, ungrounded = check_citations(response_text, fused)

            if not ungrounded:
                logger.info("[pipeline] citation check: RETRIED successfully")
                citation_outcome = CitationCheckOutcome.RETRIED
            else:
                # ── Step 3: Transparent fallback ─────────────────────
                logger.warning("[pipeline] citation check: FALLBACK — showing raw sources")
                citation_outcome = CitationCheckOutcome.FALLBACK

                response_text = _fallback_text(fused, language)
                grounded = []

    # ── 3b. Answer Verification ──────────────────────────────────────
    # The citation check proved the cited articles exist; this checks they say
    # what the answer uses them for. One rewrite, then the honest fallback.
    corrected = False
    unverified = False
    if grounded:
        verdict = await verify_answer(message, response_text, grounded, fused)
        if verdict and not verdict.ok:
            logger.warning(f"[pipeline] verifier: {verdict.problem}")
            corrected = True
            citation_outcome = CitationCheckOutcome.RETRIED
            response_text = await generate_response(
                query=message,
                retrieval=fused,
                language=language,
                history=history,
                pinned_document=document,
                correction=verdict.problem,
            )
            _, grounded, ungrounded = check_citations(response_text, fused)
            patched_text = strip_ungrounded_claims(response_text, ungrounded) if ungrounded else response_text

            second = None
            if patched_text and grounded:
                response_text = patched_text
                _, grounded, _ = check_citations(response_text, fused)
                second = await verify_answer(message, response_text, grounded, fused) if grounded else None
            elif patched_text:
                response_text = patched_text

            still_misused = second is not None and not second.ok
            if document and patched_text and still_misused:
                # A clause-by-clause review stays useful with one weak point, and a
                # list of articles wouldn't replace it: keep it, flagged.
                logger.warning("[pipeline] verifier: still misused after rewrite — keeping document review with a warning")
                unverified = True
                response_text = f"{UNVERIFIED_WARNING['ar' if language == 'ar' else 'en']} {second.problem}\n\n{response_text}"
            elif not patched_text or still_misused:
                logger.warning("[pipeline] verifier: still misused after rewrite — showing raw sources")
                citation_outcome = CitationCheckOutcome.FALLBACK
                response_text = _fallback_text(fused, language)
                grounded = []

    # ── 4. Confidence Score ──────────────────────────────────────────
    confidence = compute_confidence(fused, citation_outcome, grounded_citation_count=len(grounded))
    if corrected and confidence == ConfidenceLevel.HIGH:
        confidence = ConfidenceLevel.MEDIUM  # its first attempt misused an article
    if unverified:
        confidence = ConfidenceLevel.LOW

    # ── 5. Update Session ────────────────────────────────────────────
    session_store.add_turn(session_id, "user", message)
    session_store.add_turn(session_id, "assistant", response_text)

    # ── 6. Build Response ────────────────────────────────────────────
    final_citations = [
        Citation(
            law_name=c.law_name,
            article_id=c.article_id,
            text_snippet=c.text_snippet,
        )
        for c in grounded
    ]

    result = ChatResponse(
        answer=response_text,
        citations=final_citations,
        graph_path=_pick_graph_path(fused, final_citations),
        agent_steps=agent_steps,
        confidence=confidence,
        citation_check_outcome=citation_outcome,
        session_id=session_id,
        language=language,
    )

    logger.info(
        f"[pipeline] DONE confidence={confidence.value} "
        f"citation_outcome={citation_outcome.value} "
        f"citations={len(final_citations)}"
    )
    return result
