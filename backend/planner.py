"""
Planner — classifies user intent and decides the pipeline path.
Single LLM call (cheap model) that returns one of: qa, document_explanation, case_guidance, off_topic.
"""

import json
import logging
from backend.llm_client import call_llm_cheap
from backend.models import Intent

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an intent classifier for an Egyptian legal AI assistant.
Given a user message, classify it into exactly ONE of these intents:

- "qa": The user is asking a legal question about Egyptian law (e.g., "ما عقوبة السرقة؟", "What is the penalty for theft?")
- "document_explanation": The user has uploaded a document and wants it explained or summarized
- "case_guidance": The user is describing a specific legal situation and wants guidance
- "off_topic": The message is unrelated to Egyptian law

Respond with ONLY a JSON object: {"intent": "<intent_value>"}
Do not include any other text."""


async def classify_intent(message: str, has_document: bool = False) -> Intent:
    """
    Classify the user's intent.

    Args:
        message: The user's message.
        has_document: Whether the user has uploaded a document in this session.

    Returns:
        Intent enum value.
    """
    # Quick heuristic: if a document is pinned and the message is short, it's likely explanation
    if has_document and len(message.split()) < 10:
        logger.info("[planner] heuristic → document_explanation (short msg + pinned doc)")
        return Intent.DOCUMENT_EXPLANATION

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    try:
        raw = await call_llm_cheap(messages, temperature=0.1, max_tokens=64)
        # Parse the JSON response
        cleaned = raw.strip()
        # Handle markdown code blocks
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        intent_str = parsed.get("intent", "qa")

        # Map to enum
        try:
            intent = Intent(intent_str)
        except ValueError:
            logger.warning(f"[planner] unknown intent '{intent_str}', defaulting to qa")
            intent = Intent.QA

        logger.info(f"[planner] classified as: {intent.value}")
        return intent

    except Exception as e:
        logger.error(f"[planner] classification failed: {e}, defaulting to qa")
        return Intent.QA
