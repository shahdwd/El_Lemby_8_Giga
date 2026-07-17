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
# From project root — populate Qdrant + Neo4j
python -m backend.ingestion.ingest
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Main chat endpoint |
| GET | `/sessions/{id}` | Get session history |
| DELETE | `/sessions/{id}` | Clear a session |

### POST /chat

**Request:**
```json
{
  "message": "ما هي عقوبة السرقة في القانون المصري؟",
  "session_id": "optional-uuid",
  "language": "ar"
}
```

**Response:**
```json
{
  "answer": "وفقاً للمادة 318 من قانون العقوبات...",
  "citations": [{"law_name": "قانون العقوبات", "article_id": "318", "text_snippet": "..."}],
  "graph_path": [{"node_id": "1", "label": "Law", "name": "قانون العقوبات", "relationship": "CONTAINS"}],
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
  ├── Planner (LLM) → Translation (LLM) → Retriever (Qdrant + Neo4j)
  → Context Fusion → Legal Reasoning (LLM) → Citation Check → Response
```

## Team

- **Dev A**: Data & Retrieval (ingestion, Qdrant, Neo4j, retriever, context fusion)
- **Dev B**: Backend & Pipeline (FastAPI, orchestrator, LLM calls, session)
- **Dev C**: Safety & Frontend (citation check, guardrails, confidence, Next.js UI)
