"""
Retriever — combined vector (Qdrant) + graph (Neo4j) retrieval.
Owned by Dev A. This scaffold provides the interface + mock data for Dev B to build against.
Dev A will replace the mock implementations with real Qdrant/Neo4j queries.
"""

import logging
from backend.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    RETRIEVAL_TOP_K,
    EMBEDDING_MODEL,
)
from backend.models import RetrievedChunk, RetrievalResult, GraphPathNode, Intent

logger = logging.getLogger(__name__)

# ── Flags to check if DBs are configured ─────────────────────────────────
_qdrant_configured = bool(QDRANT_URL and QDRANT_API_KEY)
_neo4j_configured = bool(NEO4J_URI and NEO4J_PASSWORD)

# ── Lazy-initialized clients (Dev A will set these up) ───────────────────
_qdrant_client = None
_neo4j_driver = None
_embedding_model = None


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


def _get_embedding_model():
    """Lazy-init embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"[retriever] Embedding model loaded: {EMBEDDING_MODEL}")
    return _embedding_model


def _embed_query(query: str) -> list[float]:
    """Generate embedding for a query string."""
    model = _get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


def _graph_seed_clause(has_article_ids: bool) -> str:
    """Build the seed predicate for the graph query."""
    if has_article_ids:
        return "a.article_id IN $article_ids"

    return (
        "toLower(a.text) CONTAINS toLower($query) "
        "OR toLower(a.law_name) CONTAINS toLower($query) "
        "OR toLower(a.article_id) = toLower($query)"
    )


def _graph_query_for_intent(intent: Intent, has_article_ids: bool) -> str:
    """Return the Cypher query shape best suited to the current intent."""
    seed_clause = _graph_seed_clause(has_article_ids)

    if intent == Intent.DOCUMENT_EXPLANATION:
        return f"""
        MATCH (a:Article)
        WHERE {seed_clause}
        MATCH path = (a)<-[:CONTAINS]-(law:Law)-[:CONTAINS]->(related:Article)
        RETURN a, related, path
        UNION
        MATCH (a:Article)
        WHERE {seed_clause}
        MATCH path = (a)-[:REFERENCES]->(ref_law:Law)<-[:CONTAINS]-(related:Article)
        RETURN a, related, path
        LIMIT 10
        """

    if intent == Intent.CASE_GUIDANCE:
        return f"""
        MATCH (a:Article)
        WHERE {seed_clause}
        MATCH path = (a)-[:IMPOSES|GRANTS|PRESCRIBES|EMPOWERS|MENTIONS]->(entity:SemanticEntity)
                      <-[:IMPOSES|GRANTS|PRESCRIBES|EMPOWERS|MENTIONS]-(related:Article)
        RETURN a, related, path
        UNION
        MATCH (a:Article)
        WHERE {seed_clause}
        MATCH path = (a)-[:REFERENCES]->(ref_law:Law)<-[:CONTAINS]-(related:Article)
        RETURN a, related, path
        LIMIT 10
        """

    return f"""
    MATCH (a:Article)
    WHERE {seed_clause}
    MATCH path = (a)-[:REFERENCES]->(ref_law:Law)<-[:CONTAINS]-(related:Article)
    RETURN a, related, path
    UNION
    MATCH (a:Article)
    WHERE {seed_clause}
    MATCH path = (a)-[:IMPOSES|GRANTS|PRESCRIBES|EMPOWERS|MENTIONS]->(entity:SemanticEntity)
                  <-[:IMPOSES|GRANTS|PRESCRIBES|EMPOWERS|MENTIONS]-(related:Article)
    RETURN a, related, path
    LIMIT 10
    """


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR SEARCH (Qdrant)
# ═══════════════════════════════════════════════════════════════════════════

async def vector_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
    """
    Search Qdrant for similar legal text chunks.
    Dev A: Replace the mock with real Qdrant queries.
    """
    client = _get_qdrant_client()

    if client is None:
        logger.warning("[retriever] Qdrant not configured — returning mock data")
        return _mock_vector_results()

    # ── Real Qdrant search ───────────────────────────────────────────
    try:
        query_vector = _embed_query(query)

        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append(RetrievedChunk(
                text=payload.get("text", ""),
                law_name=payload.get("law_name", ""),
                article_id=payload.get("article_id", ""),
                categories=payload.get("categories", []),
                source="qdrant",
                score=hit.score,
            ))

        logger.info(f"[retriever] Qdrant returned {len(chunks)} chunks (top_score={chunks[0].score if chunks else 0:.3f})")
        return chunks

    except Exception as e:
        logger.error(f"[retriever] Qdrant search failed: {e}")
        return _mock_vector_results()


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH SEARCH (Neo4j)
# ═══════════════════════════════════════════════════════════════════════════

async def graph_search(query: str, article_ids: list[str] | None = None, intent: Intent | None = None) -> tuple[list[RetrievedChunk], list[GraphPathNode]]:
    """
    Search Neo4j for related articles and graph traversal path.
    Dev A: Replace the mock with real Cypher queries.

    Args:
        query: The user's query (for keyword matching).
        article_ids: Article IDs from vector search (for graph expansion).
        intent: The classified intent (for filtering and context).

    Returns:
        (related_chunks, graph_path)
    """
    driver = _get_neo4j_driver()
    normalized_intent = intent or Intent.QA

    if normalized_intent == Intent.OFF_TOPIC:
        logger.info("[retriever] off_topic intent — skipping Neo4j search")
        return [], []

    if driver is None:
        logger.warning("[retriever] Neo4j not configured — returning mock data")
        return _mock_graph_results()

    # ── Real Neo4j search ────────────────────────────────────────────
    try:
        chunks = []
        graph_path = []
        seen_chunk_ids = set()
        seen_path_nodes = set()

        graph_query = _graph_query_for_intent(normalized_intent, bool(article_ids))
        query_params = {
            "query": query,
            "article_ids": article_ids or [],
        }

        with driver.session() as session:
            result = session.run(graph_query, **query_params)

            for record in result:
                article = record.get("related") or record.get("a")
                if article:
                    article_id = article.get("article_id", "")
                    if not article_id or article_id not in seen_chunk_ids:
                        if article_id:
                            seen_chunk_ids.add(article_id)
                        chunks.append(RetrievedChunk(
                            text=article.get("text", ""),
                            law_name=article.get("law_name", ""),
                            article_id=article.get("article_id", ""),
                            categories=article.get("categories", []),
                            source="neo4j",
                            score=0.5,  # graph hits don't have similarity scores
                        ))

                # Build graph path from traversal
                if record.get("path"):
                    for node in record["path"].nodes:
                        node_id = str(node.element_id)
                        if node_id in seen_path_nodes:
                            continue
                        seen_path_nodes.add(node_id)
                        graph_path.append(GraphPathNode(
                            node_id=node_id,
                            label=list(node.labels)[0] if node.labels else "Unknown",
                            name=node.get("name", node.get("article_id", "")),
                            relationship="",
                        ))

        # Add relationship labels to graph path
        for i in range(len(graph_path) - 1):
            graph_path[i].relationship = "RELATED_TO"

        logger.info(f"[retriever] Neo4j returned {len(chunks)} chunks, {len(graph_path)} path nodes")
        return chunks, graph_path

    except Exception as e:
        logger.error(f"[retriever] Neo4j search failed: {e}")
        return _mock_graph_results()


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

async def retrieve(query: str, intent: Intent) -> RetrievalResult:
    """
    Run both vector and graph search, combine results.
    This is the main entry point — the rest of the pipeline calls this.
    """
    if intent == Intent.OFF_TOPIC:
        logger.info("[retriever] off_topic intent — returning empty retrieval")
        return RetrievalResult()

    # Step 1: Vector search
    vector_chunks = await vector_search(query)

    # Step 2: Graph search (using article IDs from vector search for expansion)
    vector_article_ids = list(set(c.article_id for c in vector_chunks if c.article_id))
    graph_chunks, graph_path = await graph_search(query, article_ids=vector_article_ids, intent=intent)

    # Step 3: Compute corroboration (how many graph hits overlap with vector hits)
    vector_ids = set(c.article_id for c in vector_chunks)
    graph_ids = set(c.article_id for c in graph_chunks)
    corroboration = len(vector_ids & graph_ids)

    # Step 4: Combine (vector first, then unique graph results)
    all_chunks = list(vector_chunks)
    seen_ids = set(c.article_id for c in vector_chunks)
    for gc in graph_chunks:
        if gc.article_id not in seen_ids:
            all_chunks.append(gc)
            seen_ids.add(gc.article_id)

    result = RetrievalResult(
        chunks=all_chunks,
        graph_path=graph_path,
        vector_top_score=vector_chunks[0].score if vector_chunks else 0.0,
        graph_corroboration_count=corroboration,
    )

    logger.info(
        f"[retriever] combined: {len(all_chunks)} chunks, "
        f"top_score={result.vector_top_score:.3f}, "
        f"corroboration={corroboration}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MOCK DATA (used when DBs aren't configured — Dev B can work against these)
# ═══════════════════════════════════════════════════════════════════════════

def _mock_vector_results() -> list[RetrievedChunk]:
    """Mock vector results for development without Qdrant."""
    return [
        RetrievedChunk(
            text="كل من اختلس منقولاً مملوكاً لغيره فهو سارق، ويعاقب بالحبس مع الشغل مدة لا تتجاوز سنتين.",
            law_name="قانون العقوبات",
            article_id="318",
            categories=["عقوبات", "سرقة"],
            source="qdrant",
            score=0.87,
        ),
        RetrievedChunk(
            text="يعاقب بالسجن المشدد على السرقات التي ترتكب في الطرق العامة أو في إحدى وسائل النقل.",
            law_name="قانون العقوبات",
            article_id="315",
            categories=["عقوبات", "سرقة"],
            source="qdrant",
            score=0.82,
        ),
        RetrievedChunk(
            text="إذا وقعت السرقة ليلاً من شخصين فأكثر يكون أحدهم على الأقل حاملاً سلاحاً ظاهراً أو مخبأً.",
            law_name="قانون العقوبات",
            article_id="316",
            categories=["عقوبات", "سرقة"],
            source="qdrant",
            score=0.78,
        ),
    ]


def _mock_graph_results() -> tuple[list[RetrievedChunk], list[GraphPathNode]]:
    """Mock graph results for development without Neo4j."""
    chunks = [
        RetrievedChunk(
            text="الشروع في الجنايات المنصوص عليها في المواد السابقة يعاقب عليه بالسجن.",
            law_name="قانون العقوبات",
            article_id="321",
            categories=["عقوبات", "سرقة", "شروع"],
            source="neo4j",
            score=0.5,
        ),
    ]
    path = [
        GraphPathNode(node_id="1", label="Law", name="قانون العقوبات", relationship="CONTAINS"),
        GraphPathNode(node_id="2", label="Article", name="المادة 318", relationship="REFERENCES"),
        GraphPathNode(node_id="3", label="Article", name="المادة 321", relationship=""),
    ]
    return chunks, path
