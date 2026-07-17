"""
OpenRouter LLM client — async wrapper for chat completions.
Supports both the main model (reasoning/response) and cheap model (planner/translation).
"""

import httpx
import logging
from backend.config import OPENROUTER_API_KEY, LLM_MODEL, LLM_MODEL_CHEAP

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


async def call_llm(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    Call OpenRouter chat completions API.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Override model name. Defaults to LLM_MODEL.
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.

    Returns:
        The assistant's response text.
    """
    model = model or LLM_MODEL

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://el-lemby.vercel.app",
        "X-Title": "Qanony - Egyptian Law AI Assistant",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info(f"[llm] calling model={model} msgs={len(messages)} temp={temperature}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OPENROUTER_BASE_URL, json=payload, headers=headers)
        response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    logger.info(f"[llm] response length={len(content)} chars")
    return content


async def call_llm_cheap(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Call the cheaper/faster model — used for Planner and Translation."""
    return await call_llm(
        messages=messages,
        model=LLM_MODEL_CHEAP,
        temperature=temperature,
        max_tokens=max_tokens,
    )
