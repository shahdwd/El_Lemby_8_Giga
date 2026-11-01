"""
Answer verifier — the citation check proves a cited article exists in the context;
this checks it actually says what the answer uses it for. Article 152 of the Penal
Code (theft of government documents) is real, but it doesn't set the penalty for
theft in general, and only reading its text shows that.

A single cheap LLM call that returns a verdict. If that call fails the answer goes
through unverified: this adds a safety check, and shouldn't take the assistant down.
"""

import json
import logging
from dataclasses import dataclass

from .llm_client import call_llm_cheap
from .models import Citation, RetrievalResult

logger = logging.getLogger(__name__)

MAX_ARTICLES = 8
MAX_ARTICLE_CHARS = 1500

VERIFIER_PROMPT = """You check a legal assistant's answer against the articles it cites.

Decide whether the answer MISUSES an article, meaning either:
- it presents an article as settling the question when the article only covers a narrower or different case (e.g. using a rule about theft of government documents as "the penalty for theft" in general), or
- it states something as coming from an article whose text does not say it.

These are NOT misuse:
- Saying the available law doesn't cover the question, and mentioning an article as related only.
- Reasonable paraphrase, or applying an article to facts it clearly covers.
- Comparing a clause of an uploaded document with an article and concluding that it conflicts.

Respond with ONLY a JSON object:
{"misused": true or false, "problem": "<one sentence, in the language of the question, naming the article and what is wrong; empty if none>"}"""


@dataclass
class Verdict:
    ok: bool
    problem: str = ""


async def verify_answer(
    question: str,
    answer: str,
    citations: list[Citation],
    retrieval: RetrievalResult,
) -> Verdict | None:
    """Returns a Verdict, or None if the check couldn't run."""
    texts = {(c.law_name, c.article_id): c.text for c in retrieval.chunks}
    articles = [
        f"[{c.law_name} | المادة {c.article_id}]\n{texts.get((c.law_name, c.article_id), '')[:MAX_ARTICLE_CHARS]}"
        for c in citations[:MAX_ARTICLES]
    ]
    messages = [
        {"role": "system", "content": VERIFIER_PROMPT},
        {
            "role": "user",
            "content": (
                f"## Question\n{question}\n\n## Answer\n{answer}\n\n"
                "## Cited articles\n" + "\n\n".join(articles)
            ),
        },
    ]

    try:
        raw = (await call_llm_cheap(messages, temperature=0.0, max_tokens=256)).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        verdict = Verdict(ok=not parsed["misused"], problem=parsed.get("problem") or "")
    except Exception as e:
        logger.warning(f"[verifier] check could not run: {e}")
        return None

    logger.info(f"[verifier] ok={verdict.ok} problem={verdict.problem[:120]}")
    return verdict
