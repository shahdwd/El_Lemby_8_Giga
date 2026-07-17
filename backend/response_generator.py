"""
Response Generator — merged Legal Reasoning + Response Generation.
Single LLM call: retrieved context in → cited, grounded answer out.
"""

import logging
from backend.llm_client import call_llm
from backend.models import RetrievalResult

logger = logging.getLogger(__name__)


def build_response_prompt(
    query: str,
    retrieval: RetrievalResult,
    language: str = "ar",
    history: list[dict] | None = None,
    pinned_document: str | None = None,
    is_retry: bool = False,
) -> list[dict]:
    """
    Build the full prompt for the merged reasoning + response LLM call.

    Args:
        query: The user's question (in original language).
        retrieval: The retrieval result with chunks and graph path.
        language: Response language ("ar" or "en").
        history: Conversation history as list of {"role", "content"} dicts.
        pinned_document: Content of uploaded document, if any.
        is_retry: If True, uses a stricter prompt (for citation check retry).
    """
    # Format retrieved chunks as context
    context_lines = []
    for i, chunk in enumerate(retrieval.chunks, 1):
        context_lines.append(
            f"[{i}] القانون: {chunk.law_name} | المادة: {chunk.article_id}\n{chunk.text}"
        )
    context_block = "\n\n".join(context_lines) if context_lines else "لا توجد نتائج مرتبطة."

    # Format graph path
    graph_path_text = ""
    if retrieval.graph_path:
        path_parts = []
        for node in retrieval.graph_path:
            path_parts.append(f"{node.label}: {node.name}")
            if node.relationship:
                path_parts.append(f"  --[{node.relationship}]-->")
        graph_path_text = "\n".join(path_parts)

    # System prompt
    strictness_note = ""
    if is_retry:
        strictness_note = (
            "\n\n⚠️ CRITICAL: A previous response cited articles NOT present in the context. "
            "You MUST ONLY cite articles that appear in the context below. "
            "If you are not sure about a citation, say so explicitly rather than guessing."
        )

    lang_instruction = (
        "أجب باللغة العربية." if language == "ar"
        else "Respond in English."
    )

    system_prompt = f"""أنت مساعد قانوني مصري متخصص. مهمتك هي الإجابة على الأسئلة القانونية بناءً على السياق المسترجع فقط.

You are a specialized Egyptian legal assistant. Your task is to answer legal questions based ONLY on the retrieved context.

## Rules:
1. ONLY cite articles that appear in the context below — do NOT invent or hallucinate citations.
2. For each claim, reference the specific law name and article number (e.g., "وفقاً للمادة 318 من قانون العقوبات").
3. If the context doesn't contain enough information to answer, say so explicitly.
4. {lang_instruction}
5. Structure your answer clearly with the legal reasoning first, then the conclusion.
6. End with a brief disclaimer that this is informational, not legal advice.{strictness_note}

## Retrieved Legal Context:
{context_block}

{f"## Graph Traversal Path:{chr(10)}{graph_path_text}" if graph_path_text else ""}

{f"## Uploaded Document:{chr(10)}{pinned_document[:2000]}" if pinned_document else ""}"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (if any)
    if history:
        messages.extend(history)

    # Add current user query
    messages.append({"role": "user", "content": query})

    return messages


async def generate_response(
    query: str,
    retrieval: RetrievalResult,
    language: str = "ar",
    history: list[dict] | None = None,
    pinned_document: str | None = None,
    is_retry: bool = False,
) -> str:
    """
    Generate a legal response using the merged reasoning + response call.

    Returns:
        The LLM's response text.
    """
    messages = build_response_prompt(
        query=query,
        retrieval=retrieval,
        language=language,
        history=history,
        pinned_document=pinned_document,
        is_retry=is_retry,
    )

    response = await call_llm(messages, temperature=0.3, max_tokens=2048)
    logger.info(f"[response_generator] generated {len(response)} chars (retry={is_retry})")
    return response
