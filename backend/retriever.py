"""
Retriever — hybrid text search (Qdrant vectors + BM25 keywords) expanded
through the Neo4j graph.

Articles are identified by (law_name, article_id) everywhere: article numbers
repeat across laws, so an article_id on its own is ambiguous.
"""

import asyncio
import logging
from .config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    RETRIEVAL_TOP_K,
)
from .models import RetrievedChunk, RetrievalResult, GraphPathNode, Intent
from .ingestion.normalize import article_number
from .keyword_search import keyword_search

logger = logging.getLogger(__name__)

# ── Flags to check if DBs are configured ─────────────────────────────────
_qdrant_configured = bool(QDRANT_URL and QDRANT_API_KEY)
_neo4j_configured = bool(NEO4J_URI and NEO4J_PASSWORD)

# ── Lazy-initialized clients ─────────────────────────────────────────────
_qdrant_client = None
_neo4j_driver = None


def _get_qdrant_client():
    """Lazy-init Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None and _qdrant_configured:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("[retriever] Qdrant client initialized")
    return _qdrant_client


def _get_neo4j_driver():
    """Lazy-init Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is None and _neo4j_configured:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        logger.info("[retriever] Neo4j driver initialized")
    return _neo4j_driver


def _embed_query(query: str) -> list[float]:
    """
    Embed the query locally with the same model + normalization used at ingestion
    time, so query vectors live in the same space as the Qdrant collection.
    """
    from backend.ingestion.embed import embed_single
    return embed_single(query)


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR SEARCH (Qdrant)
# ═══════════════════════════════════════════════════════════════════════════

def _vector_search_sync(query: str, top_k: int) -> list[RetrievedChunk]:
    client = _get_qdrant_client()
    if client is None:
        logger.error("[retriever] Qdrant not configured")
        return []

    try:
        # Over-fetch: long articles are split into several chunks, and we want
        # top_k distinct articles, keeping each article's best-scoring chunk.
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=_embed_query(query),
            limit=top_k * 3,
            with_payload=True,
        )
    except Exception as e:
        logger.error(f"[retriever] Qdrant search failed: {e}")
        return []

    chunks: list[RetrievedChunk] = []
    seen: set[tuple[str, str]] = set()
    for hit in results:
        payload = hit.payload or {}
        key = (payload.get("law_name", ""), payload.get("article_id", ""))
        if key in seen:
            continue
        seen.add(key)
        chunks.append(RetrievedChunk(
            text=payload.get("text", ""),
            law_name=key[0],
            article_id=key[1],
            categories=payload.get("categories", []),
            source="qdrant",
            score=hit.score,
        ))
        if len(chunks) == top_k:
            break

    logger.info(f"[retriever] Qdrant returned {len(chunks)} articles (top_score={chunks[0].score if chunks else 0:.3f})")
    return chunks


async def vector_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
    """Search Qdrant for the top_k most similar articles."""
    return await asyncio.to_thread(_vector_search_sync, query, top_k)


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH SEARCH (Neo4j)
# ═══════════════════════════════════════════════════════════════════════════

# Articles that share an LLM-extracted entity (penalty, right, authority, ...)
# with a seed article, ranked by how many entities they share.
_SHARED_ENTITY_QUERY = """
UNWIND $seeds AS seed
MATCH (law:Law {name: seed.law_name})-[c:CONTAINS]->(a:Article {article_id: seed.article_id})
MATCH (a)-[r1]->(e:SemanticEntity)<-[r2]-(rel:Article)
WHERE rel <> a
WITH law, c, a, rel, collect({e: e, r1: r1, r2: r2}) AS via
RETURN law, c, a, rel, size(via) AS shared, via[0].e AS e, via[0].r1 AS r1, via[0].r2 AS r2
ORDER BY shared DESC
LIMIT $limit
"""

# Neighbouring articles (n-1, n+1) in the same law: related provisions are
# usually consecutive (e.g. the theft articles 311-318 of the Penal Code).
_NEIGHBOUR_QUERY = """
UNWIND $pairs AS pair
MATCH (law:Law {name: pair.law_name})-[c:CONTAINS]->(rel:Article {article_id: pair.article_id})
RETURN law, c, rel
"""


def _node_to_path(node, relationship: str = "") -> GraphPathNode:
    label = next((l for l in node.labels if l != "SemanticEntity"), "Unknown")
    if label == "Article":
        name = article_node_name(node.get("law_name", ""), node.get("article_id", ""))
    else:
        name = node.get("name", "")
    return GraphPathNode(node_id=node.element_id, label=label, name=name, relationship=relationship)


def _article_chunk(node, source: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=node.get("text", ""),
        law_name=node.get("law_name", ""),
        article_id=node.get("article_id", ""),
        categories=node.get("categories") or [],
        source=source,
        score=score,
    )


def article_node_name(law_name: str, article_id: str) -> str:
    """Display name of an Article node in a graph path."""
    return f"{law_name} — المادة {article_id}"


def _graph_search_sync(seeds: list[RetrievedChunk], limit: int) -> tuple[list[RetrievedChunk], list[list[GraphPathNode]]]:
    driver = _get_neo4j_driver()
    if driver is None:
        logger.error("[retriever] Neo4j not configured")
        return [], []
    if not seeds:
        return [], []

    seed_params = [{"law_name": s.law_name, "article_id": s.article_id} for s in seeds]
    neighbour_params = []
    for s in seeds:
        n = article_number(s.article_id)
        if n is None:
            continue
        for m in (int(n) - 1, int(n) + 1):
            if m > 0:
                neighbour_params.append({"law_name": s.law_name, "article_id": str(m)})

    chunks: list[RetrievedChunk] = []
    paths: list[list[GraphPathNode]] = []
    try:
        with driver.session() as session:
            for rec in session.run(_SHARED_ENTITY_QUERY, seeds=seed_params, limit=limit):
                chunks.append(_article_chunk(rec["rel"], "neo4j_entity", 0.5))
                # Law -CONTAINS-> seed Article -r1-> Entity <-r2- related Article
                paths.append([
                    _node_to_path(rec["law"], rec["c"].type),
                    _node_to_path(rec["a"], rec["r1"].type),
                    _node_to_path(rec["e"], rec["r2"].type),
                    _node_to_path(rec["rel"]),
                ])

            for rec in session.run(_NEIGHBOUR_QUERY, pairs=neighbour_params):
                chunks.append(_article_chunk(rec["rel"], "neo4j_neighbour", 0.45))
                paths.append([_node_to_path(rec["law"], rec["c"].type), _node_to_path(rec["rel"])])
    except Exception as e:
        logger.error(f"[retriever] Neo4j search failed: {e}")
        return [], []

    logger.info(f"[retriever] Neo4j returned {len(chunks)} related articles, {len(paths)} paths")
    return chunks, paths


async def graph_search(seeds: list[RetrievedChunk], limit: int = RETRIEVAL_TOP_K) -> tuple[list[RetrievedChunk], list[list[GraphPathNode]]]:
    """
    Expand the vector hits through the graph.

    Args:
        seeds: Articles found by vector search.
        limit: Max shared-entity expansions.

    Returns:
        (related_chunks, graph_paths) — paths strongest first
    """
    return await asyncio.to_thread(_graph_search_sync, seeds, limit)


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

_RRF_K = 60  # standard Reciprocal Rank Fusion constant


async def _hybrid_rank(query: str, per_query: int) -> tuple[list[RetrievedChunk], float]:
    """
    Hybrid retrieval for one query: vector (Qdrant) + keyword (BM25), merged with
    Reciprocal Rank Fusion — each list adds 1/(k + rank) per article, so an
    article both methods agree on rises to the top.

    Returns (chunks in fused order, best vector cosine score).
    """
    vector_hits = await vector_search(query, top_k=per_query)
    keyword_hits = await asyncio.to_thread(keyword_search, query, per_query)

    fused: dict[tuple[str, str], float] = {}
    chunks: dict[tuple[str, str], RetrievedChunk] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, chunk in enumerate(hits):
            key = (chunk.law_name, chunk.article_id)
            fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if key not in chunks:
                chunks[key] = chunk
            elif chunks[key].source != chunk.source:
                chunks[key].source = "hybrid"

    ranked = sorted(fused, key=fused.get, reverse=True)
    return [chunks[k] for k in ranked], (vector_hits[0].score if vector_hits else 0.0)


async def hybrid_search(queries: list[str], per_query: int, top_k: int) -> tuple[list[RetrievedChunk], float]:
    """
    Hybrid-rank every query, then merge round-robin (every query's #1, then every
    #2, ...) so each clause of a document gets its best match in, rather than
    articles that weakly match many clauses.

    Returns (top_k chunks, best vector cosine score). Chunk scores are set by
    position to 1.0-0.55 so they sort above graph expansions (0.45-0.5) in
    context fusion; the cosine score is returned separately for confidence.
    """
    rankings = []
    vector_top = 0.0
    for q in queries:
        ranked, top = await _hybrid_rank(q, per_query)
        rankings.append(ranked)
        vector_top = max(vector_top, top)

    result: list[RetrievedChunk] = []
    seen: set[tuple[str, str]] = set()
    for position in range(max((len(r) for r in rankings), default=0)):
        for ranked in rankings:
            if position < len(ranked) and len(result) < top_k:
                chunk = ranked[position]
                key = (chunk.law_name, chunk.article_id)
                if key not in seen:
                    seen.add(key)
                    result.append(chunk)

    for i, chunk in enumerate(result):
        chunk.score = 1.0 - 0.45 * i / max(1, len(result) - 1)
    return result, vector_top


async def retrieve(query: str, intent: Intent = Intent.QA, extra_queries: list[str] | None = None) -> RetrievalResult:
    """
    Run vector search, then expand its hits through the graph.
    This is the main entry point — the rest of the pipeline calls this.

    extra_queries: further texts to search (e.g. each clause of an uploaded
    contract); their hits are merged with the main query's.
    """
    if intent == Intent.OFF_TOPIC:
        logger.info("[retriever] off_topic — skipping retrieval")
        return RetrievalResult()

    queries = [query, *(extra_queries or [])]
    if len(queries) <= 2:
        # A question (plus its statutory-wording rewrite): full-depth search.
        per_query, top_k = RETRIEVAL_TOP_K, RETRIEVAL_TOP_K + 3
    else:
        # A document: search each clause and keep ~its top 2 (typically the vector
        # and the keyword winner); context fusion's token budget caps the total.
        per_query, top_k = 3, 2 * len(queries)
    seed_chunks, vector_top = await hybrid_search(queries, per_query=per_query, top_k=top_k)
    graph_chunks, graph_paths = await graph_search(seed_chunks)

    # Corroboration: articles the graph reached from a *different* seed that
    # the text search also found independently.
    seed_keys = {(c.law_name, c.article_id) for c in seed_chunks}
    graph_keys = {(c.law_name, c.article_id) for c in graph_chunks}
    corroboration = len(seed_keys & graph_keys)

    all_chunks = list(seed_chunks)
    seen = set(seed_keys)
    for gc in graph_chunks:
        key = (gc.law_name, gc.article_id)
        if key not in seen:
            all_chunks.append(gc)
            seen.add(key)

    result = RetrievalResult(
        chunks=all_chunks,
        graph_path=graph_paths[0] if graph_paths else [],
        graph_paths=graph_paths,
        vector_top_score=vector_top,
        graph_corroboration_count=corroboration,
    )

    logger.info(
        f"[retriever] combined: {len(all_chunks)} chunks, "
        f"top_score={result.vector_top_score:.3f}, "
        f"corroboration={corroboration}"
    )
    return result
