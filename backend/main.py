"""
FastAPI application — single process, no framework beyond standard library + LLM client.
Entry point: uvicorn backend.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import logging

from backend.pipeline import handle_chat
from backend.session import session_store
from backend.guardrails import check_input
from backend.models import ChatResponse, ConfidenceLevel, CitationCheckOutcome

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Qanoony — Egyptian Law AI Assistant",
    version="1.0.0",
    description="AI-powered Egyptian legal assistant with grounded citations and explainability.",
)

# ── CORS (allow frontend dev server + deployed frontend) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas (Pydantic for FastAPI validation) ─────────

class ChatRequestBody(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: str = "ar"   # "ar" | "en"


class ChatResponseBody(BaseModel):
    answer: str
    citations: list[dict] = []
    graph_path: list[dict] = []
    confidence: str = "medium"
    citation_check_outcome: str = "passed"
    session_id: str = ""
    language: str = "ar"
    disclaimer: str = (
        "هذا المساعد يقدم معلومات قانونية عامة ولا يُغني عن استشارة محامٍ مختص. "
        "This assistant provides general legal information and is not a substitute "
        "for professional legal advice."
    )


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — used for deployment smoke tests."""
    return {"status": "ok", "service": "qanony"}


@app.post("/chat", response_model=ChatResponseBody)
async def chat(body: ChatRequestBody):
    """
    Main chat endpoint.
    Flow: guardrails → pipeline (planner → translation → retriever →
          context fusion → LLM → citation check) → response.
    """
    # Assign session ID if not provided
    session_id = body.session_id or str(uuid.uuid4())
    logger.info(f"[chat] session={session_id} lang={body.language} msg={body.message[:80]}...")

    # ── Input guardrails ─────────────────────────────────────────────
    guardrail_result = check_input(body.message)
    if not guardrail_result["safe"]:
        logger.warning(f"[guardrails] blocked: {guardrail_result['reason']}")
        return ChatResponseBody(
            answer=guardrail_result.get(
                "user_message",
                "عذراً، لا أستطيع معالجة هذا الطلب. / Sorry, I cannot process this request."
            ),
            session_id=session_id,
            language=body.language,
            confidence="low",
        )

    # ── Pipeline ─────────────────────────────────────────────────────
    try:
        result: ChatResponse = await handle_chat(
            message=body.message,
            session_id=session_id,
            language=body.language,
        )
    except Exception as e:
        logger.error(f"[pipeline] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal pipeline error.")

    # ── Serialize response ───────────────────────────────────────────
    return ChatResponseBody(
        answer=result.answer,
        citations=[
            {"law_name": c.law_name, "article_id": c.article_id, "text_snippet": c.text_snippet}
            for c in result.citations
        ],
        graph_path=[
            {"node_id": n.node_id, "label": n.label, "name": n.name, "relationship": n.relationship}
            for n in result.graph_path
        ],
        confidence=result.confidence.value,
        citation_check_outcome=result.citation_check_outcome.value,
        session_id=result.session_id,
        language=result.language,
        disclaimer=result.disclaimer,
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve conversation history for a session."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "turns": [{"role": t.role, "content": t.content} for t in session.turns],
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear a session."""
    session_store.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


from fastapi import UploadFile, File
@app.post("/sessions/{session_id}/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """Upload a document to pin to the session."""
    try:
        content = await file.read()
        # Decode as utf-8 (assuming text files for MVP)
        text = content.decode("utf-8", errors="ignore")
        session = session_store.get_or_create(session_id)
        session.pin_document(text)
        logger.info(f"[upload] pinned document to session={session_id}, length={len(text)}")
        return {"status": "success", "filename": file.filename, "length": len(text)}
    except Exception as e:
        logger.error(f"[upload] error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process file upload")
