# Qanony — Egyptian Law AI Assistant: Current Architecture

This document describes the architecture as it exists on `main` today. It supersedes the earlier MVP/hackathon plan: the system now researches each question with a tool-calling agent, verifies citations against both retrieval and article text, and uses hybrid vector + keyword retrieval with graph expansion.

## 1. High-level architecture

```
User → Next.js frontend → FastAPI (backend/main.py) → Pipeline Orchestrator (backend/pipeline.py)
  ├── Input guardrails (backend/guardrails.py)
  ├── Planner / intent classifier (backend/planner.py)
  ├── Research agent (backend/agent.py) — LLM + tools: search_law, get_article, explore_graph
  │     (fixed retrieval pipeline in backend/pipeline.py::_fixed_answer as fallback)
  ├── Citation check (backend/citation_check.py) — deterministic grounding
  ├── Answer verifier (backend/answer_verifier.py) — LLM misuse check
  ├── Confidence score (backend/confidence.py)
  ├── Session memory (backend/session.py)
  └── Response → frontend
```

Storage:
- **Qdrant** — vector search over chunked articles (local multilingual MiniLM embeddings).
- **Neo4j Aura** — Law/Article/CONTAINS/REFERENCES/SemanticEntity graph.
- **In-memory** — sessions and pinned uploaded documents.

LLM access is through OpenRouter (`backend/llm_client.py`), using `LLM_MODEL` for reasoning/tool calls, `LLM_MODEL_CHEAP` for planner/translation, and `LLM_MODEL_LITE` for ingestion-time entity extraction.

## 2. Request flow (`backend/pipeline.py::handle_chat`)

1. **Guardrails** (`check_input`): length and prompt-injection checks.
2. **Planner** (`classify_intent`): `qa`, `document_explanation`, `case_guidance`, or `off_topic`. Off-topic stops here.
3. **Research + answer:**
   - If `AGENT_ENABLED` is true, run `run_agent` (tool-calling loop, up to 5 tool rounds / 10 calls). Everything the agent reads becomes its `RetrievalResult`.
   - If the agent is disabled or fails, use `_fixed_answer`: translate/rewrite the query, retrieve once, fuse context, generate a response.
4. **Citation check** (`check_citations`): every cited `(law, article)` must be in the retrieved context. On failure: strip ungrounded claims → regenerate once with a stricter prompt → transparent fallback showing raw sources.
5. **Answer verification** (`verify_answer`): each cited article must actually say what the answer uses it for. One rewrite, then fallback; document reviews are kept with a warning and low confidence.
6. **Confidence** (`compute_confidence`): derived from vector similarity, graph corroboration, citation outcome, and grounded-citation count. Zero citations is always low.
7. **Session update** and response build, including the graph path chosen to pass through a cited article.

For an uploaded document, its clauses are searched up front (agent seeds with them), and the document text is pinned to the session.

## 3. Backend components

| Component | File | Notes |
|---|---|---|
| FastAPI app / endpoints | `backend/main.py` | `/health`, `/chat`, `/sessions/{id}`, `/upload`, `/sessions/{id}/document` |
| Pipeline orchestrator | `backend/pipeline.py` | Agent + fixed fallback + citation check + verifier + confidence |
| Research agent | `backend/agent.py` | 3 tools, capped loop, self-check rewrite for citation-less answers |
| Answer verifier | `backend/answer_verifier.py` | Single cheap LLM call; returns `None` on failure so the answer goes through unverified |
| Citation check | `backend/citation_check.py` | Regex extraction of Arabic/English citations; set-membership grounding |
| Confidence | `backend/confidence.py` | 3-tier label from retrieval signals |
| Retriever | `backend/retriever.py` | Qdrant vectors + BM25 (RRF) + Neo4j graph expansion |
| Context fusion | `backend/context_fusion.py` | Dedupe by `(law, article)`, sort by score, cap to token budget |
| Keyword search (BM25) | `backend/keyword_search.py` | In-memory index over `backend/data/laws/*.md`; also `find_article`, `law_names` |
| Planner | `backend/planner.py` | Cheap LLM intent classification with a short-message heuristic |
| Translation / rewrite | `backend/translation.py` | `ensure_arabic_query`, `rewrite_for_retrieval` |
| Guardrails | `backend/guardrails.py` | Length + injection-pattern checks |
| Document upload | `backend/document_upload.py` | PDF (pypdf) / TXT / MD, NFKC normalization, 5 MB / 12k-char caps |
| Session store | `backend/session.py` | In-memory, last `SESSION_MAX_TURNS` turns, pinned document |
| Shared models | `backend/models.py` | `RetrievalResult`, `ChatResponse`, etc. (agreed retriever contract) |
| Config | `backend/config.py` | Loads `.env`/`env`; certifi bundle for Neo4j on Windows |

## 4. Retrieval details

- Articles are identified everywhere by `(law_name, article_id)`; article numbers repeat across laws.
- Vector search over-fetches (`top_k * 3`) and keeps each article's best chunk.
- Hybrid ranking uses Reciprocal Rank Fusion (`_RRF_K = 60`) of Qdrant and BM25 results.
- For documents, each clause is searched and results are merged round-robin so each clause gets a match.
- Graph expansion: articles sharing an LLM-extracted entity (ranked by shared count) plus neighbouring articles (`n-1`, `n+1`).
- Context fusion caps the merged set to `TOKEN_BUDGET` tokens.

## 5. Data and ingestion

- Source laws live in `backend/data/laws/*.md` (6 laws: Labor 14/2025, Civil Code 131/1948, Penal Code 1–178, Real Estate Tax, Criminal Procedure (29 articles), Commercial Code (20 articles)).
- `backend/ingestion/normalize.py` parses canonical Markdown into `{law_name, article_id, categories, text}`.
- `backend/ingestion/chunker.py` splits long articles (~512 tokens, overlap).
- `backend/ingestion/load_qdrant.py` embeds with the local MiniLM model and rebuilds the collection.
- `backend/ingestion/load_neo4j.py` builds the graph; runs LLM entity extraction (`extract_graph.py`) with `LIMIT_LLM_EXTRACTION` to control cost; results are cached in `backend/data/extracted_entities.json`.
- Both loaders wipe and rebuild their stores.

## 6. Frontend (`frontend/src/app/page.js`)

- Next.js 16 / React 19, Arabic/English toggle, RTL, mobile sidebar.
- Chat, example questions, coverage note, upload box, session switching.
- Renders Markdown-lite answers, confidence badge, grounding outcome badge, research steps, graph path, and collapsible citations.
- Talks to `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`).

## 7. Evaluation (`eval/eval_harness.py`)

- 17 cases: 11 legal questions with expected articles, plus off-topic, prompt-injection, and 4 out-of-corpus crime questions.
- Checks citation correctness, safety handling, citation-check outcomes, confidence, and latency; writes `eval/results.json`.
- Current recorded result: 17/17 passed, ~11.5s average latency.

## 8. Configuration and runtime

- Environment: `.env` or `env` at project root or `backend/../`. Required keys: OpenRouter, Qdrant, Neo4j. See `.env.example`-style entries in `README.md`.
- Python 3.11 is required; dependencies in `backend/requirements.txt`.
- Backend startup preloads the embedding model and BM25 index (~10s).
- `AGENT_ENABLED=false` switches to the fixed retrieval flow; the verifier still runs.
- Neo4j Aura and Qdrant free tiers pause when idle and must be resumed before a demo.

## 9. Deployment status

Deployment is **not done and is out of scope** for this project; it runs locally. CORS in `backend/main.py` allows `localhost:3000` and `*.vercel.app` for future use.

## 10. Known limitations / open work

- Contract review quality varies between runs (12–62s; one run missed the dismissal clause; Art. 156 applied to fixed-term contracts).
- Verifier tested via scripts/eval only; not yet exercised in the browser.
- Fallback lists loosely related articles instead of saying the law is missing.
- Research steps appear only after the answer and are English in the Arabic UI.
- `ask_user` and `calculate_entitlement` agent tools are not implemented.
- Penal Code is limited to Articles 1–178; Criminal Procedure and Commercial Code are small subsets.
- PDF upload with a real Arabic PDF is untested in the app.
- Graph path for uploaded documents sometimes shows a related Labor Law link instead of a cited article.
- Consider a stronger multilingual embedding model (requires full re-embed).
- Add more eval questions (English, Civil Code).
- Keep 2–3 saved good answers as a backup for demos.
