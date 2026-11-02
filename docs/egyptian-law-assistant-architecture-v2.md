# Egyptian Law AI Assistant — MVP Architecture & 24-Hour Build Plan (v2)

*Revised from v1 based on the team's updated architecture: plain pipeline instead of LangGraph, citation check kept as a deterministic function (not a full agent), and a defined fallback flow for validator failures.*

## 0. Knowledge base: datasets + MD files, combined

**Use both.** The two Hugging Face datasets (`egypt-legal-corpus` for the 25M-token legal text, `QA_LAW_Egyptian_dataset` as your eval set) remain your primary corpus. Legal documents ingested as `.md` files sit alongside them as a second source — this is fine as long as both sources are normalized to the **same metadata shape** before they hit Neo4j/Qdrant: `law_name`, `categories`, `article_id` (or equivalent). If the md-ingestion path produces documents with a different schema, Context Fusion has to special-case two formats, which is exactly the kind of subtle bug that surfaces mid-demo. Normalize at ingestion time, not at retrieval time.

The Flask PDF extractor remains a backup only — useful if you want to demo "ingest a new law live," not part of the core pipeline.

---

## 1. High-Level Architecture (revised)

```
USER
 │
 ▼
React / Next.js
 │
 ▼
FastAPI API
 │
 ▼
Pipeline Orchestrator  ← plain Python, no LangGraph (see Section 2)
 │
 ├──▶ Planner (LLM call)
 ├──▶ Translation (LLM call, single-purpose)
 └──▶ Session Memory (state, not an LLM call)
 │
 ▼
Retriever (function, not an agent)
 │
 ├──▶ Vector Search (Qdrant)   → similar articles, court cases
 └──▶ Graph Search (Neo4j)     → related articles, legal principles
 │
 ▼
Context Fusion (function — merge + dedupe + token-budget cap)
 │
 ▼
Legal Reasoning + Response Generation (single LLM call — see Section 2)
 │
 ▼
Citation Check (function — deterministic grounding check, see Section 4)
 │
 ├── pass ──▶ Final Answer + Graph Path + Confidence Score
 └── fail ──▶ Fallback flow (see Section 4)
 │
 ▼
React Frontend
```

---

## 2. Why no LangGraph — and what "agent" actually means here

Most of the boxes in the original diagram aren't agents in any meaningful sense — an agent implies an LLM deciding what to do next. Sorted honestly:

**Actually agentic (LLM reasoning required):**
- Planner — interprets the user's intent and decides the path (Q&A / document explanation / case guidance)
- Translation — single-purpose LLM call, not really "deciding" anything, but does need the model
- Legal Reasoning + Response Generation — **merged into one call** (see below)

**Not agents — deterministic functions:**
- Retriever (Qdrant + Neo4j query, no reasoning)
- Session Memory (state storage)
- Context Fusion (merge/dedupe logic)
- Citation Check (set-membership check against retrieved context)

Once you see the diagram this way, LangGraph (or any multi-agent framework) isn't just overkill — it's solving a coordination problem you don't have. You have a short, mostly-linear pipeline with one or two conditional branches (document uploaded vs. not; validator pass vs. fail). That's a plain Python `async def handle_chat(...)` function calling other functions in sequence, with an `if` statement for branches. This is:
- Faster to build under time pressure
- Easier for all 3 of you to debug live
- Easier for judges to review in your repo — a straight-line pipeline is more legible in a 5-minute code walkthrough than a graph of nodes and edges
- Zero framework-learning overhead mid-hackathon

**Merging Legal Reasoning + Response Generator:** these were two separate LLM calls in the original diagram. There's rarely a good reason to have one call reason about the law and a second call turn that reasoning into prose — a well-structured single prompt (retrieved context in, cited answer out) does both in one pass. This cuts one LLM hop off your latency budget, which matters live.

**Net LLM call count per turn**, revised: Planner → Translation (if needed) → Legal Reasoning/Response (merged) → done. Down from up to 4 hops to 2–3, and only 1 unconditionally required (the reasoning/response call — Planner and Translation can be short-circuited with cheap heuristics if you're tight on time; see Section 8 cut list).

---

## 3. Backend Components

| Component | Type | Responsibility | Owned by |
|---|---|---|---|
| Ingestion pipeline | script | Load HF datasets + md files → normalize schema → chunk → embed → write to Qdrant + Neo4j | Dev A |
| FastAPI backend | service | HTTP layer, session handling | Dev B |
| Retriever | function | Query Qdrant + Neo4j, merge, rank | Dev A + B shared interface |
| Planner | LLM call | Classify intent, pick path | Dev B |
| Translation | LLM call | Query→Arabic for retrieval if needed; response in user's language | Dev B |
| Context Fusion | function | Merge/dedupe/cap retrieved chunks | Dev A |
| Legal Reasoning + Response Generator | LLM call (merged) | Produce grounded, cited answer | Dev B |
| Citation Check | function | Verify cited articles exist in retrieved context | Dev C |
| Session Memory | state | Last 5–8 turns, session-scoped | Dev B |
| Guardrails (input) | function | Injection/jailbreak pattern check, validation | Dev C |
| Frontend | UI | Chat, upload, language toggle, disclaimer, graph path + confidence display | Dev C |
| Eval harness | script | Sample QA dataset, run through pipeline, score | Shared, whoever's free first |

No message queues, no microservices, no framework — one FastAPI process calling a chain of Python functions.

---

## 4. Citation Check — and what happens when it fails

The Citation Check is a **deterministic function**, not an agent: does every article/law ID cited in the generated answer actually appear in the context that was retrieved for that turn? Simple set-membership check, no extra LLM call, negligible latency.

**On failure, the flow is:**

1. **Try a cheap patch first.** If only some citations in the answer are ungrounded, strip just those claims/sentences and check whether the remaining answer still reads coherently. Handles the common case of one wrong detail in an otherwise-good answer.
2. **If the answer doesn't survive stripping** (e.g., the core claim depends on the bad citation), regenerate **once**, using the same retrieved context, with a stricter prompt reminder ("only cite articles present in the context below; if unsure, say so explicitly").
3. **If the regenerated answer still fails**, stop retrying — don't loop. Fall back to a transparent response: state that a confident, fully-grounded answer couldn't be generated, and surface the raw retrieved article titles/snippets directly so the user can read the source material themselves rather than trust an ungrounded synthesis.
4. **Never** silently let an ungrounded citation through, and never silently patch without any signal to the user that something was adjusted — for a legal product, an honest "low confidence, here's the source" beats a fluent but shaky answer.

**Cap retries at 1.** An unbounded regenerate-until-it-passes loop is a live-demo latency risk. One retry, then fall back — fast and honest beats slow and uncertain.

**Log every failure** (patched, retried, or fallen-back) to a simple counter during your eval pass. "Our citation check caught X% of ungrounded responses before reaching the user" is a concrete, quotable line for your 3-minute pitch — turn the safety feature into a judged differentiator, not just a defensive measure.

---

## 5. Confidence Score — definition

Don't use self-reported LLM confidence (unreliable, easy to fake, not measurable). Derive it instead from retrieval signal:
- Qdrant similarity score of the top hit(s)
- Number of corroborating graph hits from Neo4j (does the graph traversal reach the same articles the vector search found, or contradict them?)
- Whether the Citation Check passed on the first try, required a patch, or required a retry (from Section 4) — this alone is a strong confidence signal you already have "for free"

Combine these into a simple weighted score or even a 3-tier label (High / Medium / Low) rather than a precise-looking number — a fake-precise "87% confidence" invites more scrutiny than it survives.

---

## 6. Graph Path in the response

Keep this — it's a strong demo moment ("here's exactly how we traced your question to this article") and doubles as explainability, which tends to score well on compliance/legal-reasoning tracks. Implementation-wise, this is just: log the Neo4j traversal (starting node → hops → final nodes) during the Retriever step, and pass it through to the response payload unchanged. No extra computation needed beyond what the Retriever already does.

---

## 7. Conversation Memory (unchanged from v1)

Session-scoped, last 5–8 turns, passed directly into the LLM context window. No vector memory store, no summarization layer. If a document is uploaded, its extracted content stays pinned in that session's context alongside the running conversation.

---

## 8. MVP Tech Stack (unchanged from v1)

- **Frontend**: Next.js — good RTL/Arabic support, fast to scaffold, deploys to Vercel
- **Backend**: FastAPI — one process, no framework beyond the standard library + your LLM client
- **Vector DB**: Qdrant (Cloud free tier — skip self-hosting)
- **Graph DB**: Neo4j (Aura free tier — skip self-hosting)
- **LLM**: via OpenRouter, one capable multilingual model for the reasoning/response call; consider a cheaper/faster model for Planner and Translation if latency is tight
- **Auth**: none — session ID via cookie is enough
- **Orchestration**: plain Python function chain — no LangGraph, no agent framework

---

## 9. Task Breakdown — 3 Developers (revised)

**Dev A — Data & Retrieval**
- Normalize HF datasets + md files to one shared schema (`law_name`, `categories`, `article_id`)
- Build Neo4j graph (containment edges minimum; cross-reference edges where the text already states "as amended by...")
- Embed into Qdrant
- Own the Retriever function interface + Context Fusion merge logic

**Dev B — Backend & Pipeline**
- FastAPI scaffold, plain pipeline orchestration (no framework)
- Planner + Translation calls
- Merged Legal Reasoning + Response Generator call
- Session memory
- Document upload/explanation path

**Dev C — Safety & Frontend**
- Citation Check function + the 3-step fallback flow (patch → retry once → transparent fallback)
- Confidence score derivation from retrieval signals
- Input guardrails (injection pattern check, validation)
- Frontend: chat UI, upload, language toggle, disclaimer, graph path + confidence display
- Eval harness: sample the QA dataset, run end-to-end, log citation-check pass/patch/fail rates

**Integration point to agree on in hour 0:** the Retriever function's return shape (what fields it returns — chunks, article IDs, graph path, scores) — this is what Dev B and Dev C both build against.

---

## 10. 24-Hour Roadmap (unchanged structure from v1, updated to reflect the simplified pipeline)

**Hours 0–1**: Agree on Retriever interface + shared metadata schema. Confirm OpenRouter model choice.

**Hours 1–6 (parallel)**: Dev A builds ingestion + retrieval on a corpus subset first. Dev B scaffolds the plain pipeline against mocked retrieval results. Dev C builds frontend shell + Citation Check logic independently.

**Hours 6–10**: Wire real retrieval into the pipeline. Wire guardrails and Citation Check in. First real end-to-end question → cited answer.

**Hours 10–14**: Document upload/explanation. Session memory. Confidence score + graph path surfaced in the response. Full corpus ingestion running in background.

**Hours 14–18**: Eval pass against the QA dataset sample — this is where you get your citation-check pass-rate number for the pitch. Test both languages. Adversarial-test the guardrails.

**Hours 18–21**: Deploy (Qdrant Cloud + Neo4j Aura + Railway/Render for backend + Vercel for frontend). Smoke-test the deployed version, not just localhost.

**Hours 21–23**: Demo video, pitch prep. Know your Citation-Check pass rate and be ready to explain the fallback flow — it's your strongest technical talking point.

**Hour 23–24**: Buffer. No new features.

**Cut list, in order, if time runs short:**
1. Neo4j cross-reference edges — fall back to containment-only graph, or pure vector RAG
2. Document upload/explanation feature
3. Session memory beyond a single turn
4. Full corpus ingestion — a curated subset of commonly-referenced laws beats an incomplete full run
5. Planner as a separate LLM call — can be short-circuited with a simple heuristic (keyword/upload-presence check) if latency is tight

**Never cut**: the Citation Check and its fallback flow. It's cheap, and it's your best defense against your worst possible demo moment.

---

## 11. Deployment (unchanged from v1)

```
frontend (Next.js)  → Vercel
backend (FastAPI)   → Railway / Render / Fly.io
Qdrant              → Qdrant Cloud free tier
Neo4j               → Neo4j Aura free tier
```

Don't self-host the databases — managed free tiers remove an entire category of hackathon failure (a database container crashing mid-demo).

---

## 12. Critical Risks (updated)

- **Retrieval quality still matters more than model choice or framework choice.** Nothing in this revision changes that — protect Dev A's time.
- **The Citation Check is your safety centerpiece, not a nice-to-have.** With the fallback flow now defined (patch → retry once → transparent fallback), this is a concrete, demoable, and quotable feature — lead with it in your pitch.
- **Dropping LangGraph removes a failure mode, not just complexity.** Fewer moving parts means fewer things that can break in the last hours. The pipeline being "boring" is a feature for a 24-hour build, not a limitation.
- **Test Arabic/RTL rendering early**, not on hour 20.
- **Have 2–3 cached known-good exchanges as a live-demo fallback** in case of latency or API flakiness during the actual pitch.
- **Lead your pitch with the safety architecture.** Grounding check + honest fallback + confidence score is a genuine differentiator on a legal-AI track — say so explicitly, don't leave judges to infer it from the code.
