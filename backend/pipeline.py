"""
Pipeline Orchestrator — the core of the system.
Plain Python async function, no framework. Calls components in sequence:

  Planner → Translation → Retriever → Context Fusion → LLM → Citation Check → Response

This replaces LangGraph with a simple, linear, debuggable pipeline.
"""

import logging
from models import (
    ChatResponse,
    Citation,
    ConfidenceLevel,
    CitationCheckOutcome,
)
from planner import classify_intent
from translation import ensure_arabic_query
from retriever import retrieve
from context_fusion import fuse_context
from response_generator import generate_response
from citation_check import (
    check_citations,
    strip_ungrounded_claims,
    extract_citations_from_response,
)
from backend.confidence import compute_confidence
from backend.session import session_store
from models import Intent


logger = logging.getLogger(__name__)


async def handle_chat(message: str, session_id: str, language: str = "ar") -> ChatResponse:
    """
    Main pipeline orchestrator.
    
    Flow:
    1. Planner — classify intent
    2. Translation — ensure Arabic query for retrieval  
    3. Retriever — Qdrant + Neo4j
    4. Context Fusion — merge, dedupe, cap
    5. Legal Reasoning + Response Generation — single LLM call
    6. Citation Check — deterministic grounding verification
    7. Fallback flow (if citation check fails)
    8. Confidence score derivation
    9. Session update
    """

    logger.info(f"[pipeline] START session={session_id} lang={language}")

    # ── 1. Planner + Translation (Parallel) ──────────────────────────
    import asyncio
    session = session_store.get_or_create(session_id)
    has_document = session.pinned_document is not None
    
    intent_task = classify_intent(message, has_document=has_document)
    translation_task = ensure_arabic_query(message)
    
    intent, (arabic_query, was_translated) = await asyncio.gather(intent_task, translation_task)
    
    logger.info(f"[pipeline] intent={intent.value}")
    if was_translated:
        logger.info(f"[pipeline] translated query to Arabic")

    # ── 2. Intent-Specific Branching ─────────────────────────────────
    if intent == Intent.OFF_TOPIC:
        logger.info("[pipeline] intent is OFF_TOPIC; returning polite refusal")
        response_text = (
            "عذراً، هذا السؤال خارج نطاق القانون المصري. يرجى طرح سؤال قانوني." 
            if language == "ar" 
            else "Sorry, this question is outside the scope of Egyptian law. Please ask a legal question."
        )
        session_store.add_turn(session_id, "user", message)
        session_store.add_turn(session_id, "assistant", response_text)
        return ChatResponse(
            answer=response_text,
            citations=[],
            graph_path=[],
            confidence=ConfidenceLevel.HIGH,
            citation_check_outcome=CitationCheckOutcome.PASSED,
            session_id=session_id,
            language=language,
        )

    if intent == Intent.DOCUMENT_EXPLANATION and has_document:
        logger.info("[pipeline] intent is DOCUMENT_EXPLANATION with pinned document; skipping retrieval")
        from backend.models import RetrievalResult
        raw_retrieval = RetrievalResult()
    else:
        # ── 3. Retrieval ─────────────────────────────────────────────────
        raw_retrieval = await retrieve(arabic_query)
        logger.info(f"[pipeline] retrieved {len(raw_retrieval.chunks)} raw chunks")

    # ── 4. Context Fusion ────────────────────────────────────────────
    fused = fuse_context(raw_retrieval)
    logger.info(f"[pipeline] fused to {len(fused.chunks)} chunks")

    # ── 5. Response Generation ───────────────────────────────────────
    history = session_store.get_history_for_prompt(session_id)
    response_text = await generate_response(
        query=arabic_query,
        retrieval=fused,
        language=language,
        history=history if history else None,
        pinned_document=session.pinned_document,
        is_retry=False,
        intent=intent,
    )

    # ── 6. Citation Check ────────────────────────────────────────────
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
                query=arabic_query,
                retrieval=fused,
                language=language,
                history=history if history else None,
                pinned_document=session.pinned_document,
                is_retry=True,
                intent=intent,
            )
            all_citations, grounded, ungrounded = check_citations(response_text, fused)

            if not ungrounded:
                logger.info("[pipeline] citation check: RETRIED successfully")
                citation_outcome = CitationCheckOutcome.RETRIED
            else:
                # ── Step 3: Transparent fallback ─────────────────────
                logger.warning("[pipeline] citation check: FALLBACK — showing raw sources")
                citation_outcome = CitationCheckOutcome.FALLBACK

                # Build transparent fallback response
                source_list = "\n".join(
                    f"• {c.law_name} — المادة {c.article_id}:\n  {c.text[:200]}"
                    for c in fused.chunks
                )
                fallback_msg = (
                    "لم أتمكن من تقديم إجابة موثقة بالكامل. "
                    "فيما يلي المواد القانونية المسترجعة ذات الصلة:\n\n"
                    if language == "ar"
                    else "I could not generate a fully grounded answer. "
                    "Here are the relevant retrieved legal articles:\n\n"
                )
                response_text = fallback_msg + source_list

    # ── 7. Confidence Score ──────────────────────────────────────────
    confidence = compute_confidence(fused, citation_outcome)

    # ── 8. Update Session ────────────────────────────────────────────
    session_store.add_turn(session_id, "user", message)
    session_store.add_turn(session_id, "assistant", response_text)

    # ── 9. Build Response ────────────────────────────────────────────
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
        graph_path=fused.graph_path,
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
