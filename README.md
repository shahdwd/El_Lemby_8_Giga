# Qanony — Egyptian Law AI Assistant (قانوني)

AI-powered Egyptian legal assistant with grounded citations, graph-based explainability, and confidence scoring.

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Run the server:
```bash
# From project root
python -m uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Ingestion (Dev A)

```bash
# From project root — populate Neo4j + Qdrant from backend/data/laws/*.md
# (both scripts wipe and rebuild their database)
# LIMIT_LLM_EXTRACTION=0 → use only cached entity extractions (no LLM cost)
LIMIT_LLM_EXTRACTION=0 python -m backend.ingestion.load_neo4j
python -m backend.ingestion.load_qdrant
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Main chat endpoint |
| GET | `/sessions/{id}` | Get session history |
| DELETE | `/sessions/{id}` | Clear a session |
| POST | `/upload` | Upload a PDF/TXT/MD (multipart: `file`, `session_id`); its text is pinned to the session |
| DELETE | `/sessions/{id}/document` | Remove the uploaded document from a session |

A sample contract with two clauses that break the Labor Law (7 days of annual leave, dismissal without notice) is in [demo/sample_employment_contract.txt](demo/sample_employment_contract.txt) — upload it and ask "هل يوجد في هذا العقد بنود مخالفة لقانون العمل؟".

### POST /chat

**Request:**
```json
{
  "message": "ما هي مدة الإجازة السنوية للعامل؟",
  "session_id": "optional-uuid",
  "language": "ar"
}
```

**Response:**
```json
{
  "answer": "وفقاً للمادة 125 من قانون العمل، يحصل العامل على إجازة سنوية مدتها خمسة عشر يوماً...",
  "citations": [{"law_name": "قانون رقم ١٤ لسنة ٢٠٢٥ بإصدار قانون العمل", "article_id": "125", "text_snippet": "..."}],
  "graph_path": [{"node_id": "1", "label": "Law", "name": "قانون رقم ١٤ لسنة ٢٠٢٥ بإصدار قانون العمل", "relationship": "CONTAINS"}],
  "agent_steps": [{"tool": "search_law", "input": "...", "result": "..."}],
  "confidence": "high",
  "citation_check_outcome": "passed",
  "session_id": "uuid",
  "language": "ar",
  "disclaimer": "..."
}
```

## Architecture

```
User → React/Next.js → FastAPI → Pipeline Orchestrator
  ├── Guardrails → Planner (LLM)
  → Research agent (LLM + tools: search_law, get_article, explore_graph)
  → Citation Check → Answer Verifier (LLM) → Response
```

- **Research agent** (`backend/agent.py`): a tool-calling loop of up to 5 rounds that reads the law and writes the answer. Set `AGENT_ENABLED=false` in `env` to use the fixed retrieval flow, which also takes over if the agent fails.
- **Hybrid retrieval**: vector search (Qdrant) + keyword search (BM25 over the same articles), merged with Reciprocal Rank Fusion, then expanded through the Neo4j graph (articles sharing an extracted entity, and neighbouring articles). For an uploaded document, every clause is searched.
- **Citation check**: every `(law, article)` the answer cites must be in the retrieved context; otherwise the line is stripped, the answer regenerated once, or the raw sources shown.
- **Answer verifier** (`backend/answer_verifier.py`): an LLM check that each cited article says what the answer uses it for. A misused citation gets one rewrite, then the raw sources are shown; a document review keeps its answer with a warning and low confidence. If the check can't run, the answer goes through unverified.
- **Laws covered** (`backend/data/laws`): Labor Law 14/2025 (full), Civil Code 131/1948 (full), Penal Code (Articles 1–178), Real Estate Tax Law; Criminal Procedure and Commercial Code only partially.

## Evaluation

```bash
# With the backend running
python eval/eval_harness.py
```

Runs 11 legal questions with known correct articles plus off-topic, prompt-injection and 4 out-of-corpus cases; prints accuracy, citation-check and verifier outcomes and latency, and writes `eval/results.json`, including each answer.

## Team

- **Dev A**: Data & Retrieval (ingestion, Qdrant, Neo4j, retriever, context fusion)
- **Dev B**: Backend & Pipeline (FastAPI, orchestrator, LLM calls, session)
- **Dev C**: Safety & Frontend (citation check, guardrails, confidence, Next.js UI)
