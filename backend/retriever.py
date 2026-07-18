"""
Retriever — combined vector (Qdrant) + graph (Neo4j) retrieval.
Owned by Dev A. This scaffold provides the interface + mock data for Dev B to build against.
Dev A will replace the mock implementations with real Qdrant/Neo4j queries.
"""

import logging
from .config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    RETRIEVAL_TOP_K,
    EMBEDDING_MODEL,
    HF_TOKEN
)
from .models import RetrievedChunk, RetrievalResult, GraphPathNode

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


import httpx
import json

import os
from huggingface_hub import InferenceClient

def _embed_query(query: str) -> list[float]:
    """
    Generate 384-dim multilingual Arabic embeddings using the Hugging Face Inference SDK.
    Bypasses local PyTorch setup and guarantees compatibility with Qdrant collection rules.
    """
    # Grab token dynamically from your environment variables
    hf_token = os.getenv("HF_TOKEN", "")
    
    # Initialize the serverless provider client
    client = InferenceClient(
        provider="hf-inference",
        api_key=hf_token if hf_token else None, # Works tokenless for public models at low scale
    )
    
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        # feature_extraction retrieves the dense vector layer directly from the model
        embedding_output = client.feature_extraction(
            text=query,
            model=model_name
        )
        
        # Some versions of feature_extraction return numpy arrays or nested structures; 
        # ensure it maps cleanly to a linear list of floats
        if hasattr(embedding_output, "tolist"):
            embedding = embedding_output.tolist()
        elif isinstance(embedding_output, list):
            # If it's a batch return variation (matrix instead of vector), flatten it out
            if len(embedding_output) > 0 and isinstance(embedding_output[0], list):
                embedding = embedding_output[0]
            else:
                embedding = embedding_output
        else:
            embedding = list(embedding_output)

        # Confirm the structural output perfectly aligns with Qdrant index settings
        if len(embedding) == 384:
            logger.info(f"[retriever] HF SDK embedding success! Generated {len(embedding)} dims.")
            return [float(x) for x in embedding]
        else:
            logger.error(f"[retriever] Dimension mismatch! Expected 384, got {len(embedding)}")
            
    except Exception as e:
        logger.error(f"[retriever] Critical HF InferenceClient pipeline failure: {e}")
        
    # Return structural 384-dimensional fallback zero-vector to keep the system running
    return [0.0] * 384

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
        logger.error("[retriever] Qdrant not configured")
        return []

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
        return []


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH SEARCH (Neo4j)
# ═══════════════════════════════════════════════════════════════════════════

async def graph_search(query: str, article_ids: list[str] | None = None) -> tuple[list[RetrievedChunk], list[GraphPathNode]]:
    """
    Search Neo4j for related articles and graph traversal path.
    Dev A: Replace the mock with real Cypher queries.

    Args:
        query: The user's query (for keyword matching).
        article_ids: Article IDs from vector search (for graph expansion).

    Returns:
        (related_chunks, graph_path)
    """
    driver = _get_neo4j_driver()

    if driver is None:
        logger.error("[retriever] Neo4j not configured")
        return [], []

    # ── Real Neo4j search ────────────────────────────────────────────
    try:
        chunks = []
        graph_path = []

        with driver.session() as session:
            logger.info(f"article_ids: {article_ids}")
            if article_ids:
                # Existing structural lookup using vector IDs
                check = session.run("MATCH (a:Article) RETURN count(a) AS c").single()
                logger.info(f"[debug_neo4j] Total nodes with label :Article in DB: {check['c'] if check else 0}")

                # Clean strings on both sides using trim() to avoid whitespace traps
                result = session.run("""
                    MATCH (a:Article)
                    WHERE trim(a.article_id) IN [id IN $article_ids | trim(id)]
                    OPTIONAL MATCH path = (a)-[r:CONTAINS|REFERENCES*1..2]-(related:Article)
                    RETURN a, related, path
                    LIMIT 10
                """, article_ids=article_ids)
            else:
                # Fallback keyword match query if no vector IDs came through
                logger.warning("[retriever] Vector search returned empty. Falling back to Neo4j text property matching.")
                result = session.run("""
                    MATCH (a:Article)
                    WHERE a.text CONTAINS $search_term OR a.law_name CONTAINS $search_term
                    OPTIONAL MATCH path = (a)-[r:CONTAINS|REFERENCES*1..1]-(related:Article)
                    RETURN a, related, path
                    LIMIT 5
                """, search_term=query)

            # ── UN-INDENTED OUTSIDE IF/ELSE: This now processes records for both paths ──
            for record in result:
                # 1. ALWAYS extract the base article found by the match
                base_article = record.get("a")
                if base_article:
                    chunks.append(RetrievedChunk(
                        text=base_article.get("text", ""),
                        law_name=base_article.get("law_name", ""),
                        article_id=base_article.get("article_id", ""),
                        categories=base_article.get("categories", []),
                        source="neo4j_base",
                        score=0.6
                    ))
                    
                # 2. Extract the related expanded article if it exists
                related_article = record.get("related")
                if related_article:
                    chunks.append(RetrievedChunk(
                        text=related_article.get("text", ""),
                        law_name=related_article.get("law_name", ""),
                        article_id=related_article.get("article_id", ""),
                        categories=related_article.get("categories", []),
                        source="neo4j_expanded",
                        score=0.5
                    ))

                # 3. Process path nodes safely
                path = record.get("path")
                if path:
                    for node in path.nodes:
                        graph_path.append(GraphPathNode(
                            node_id=str(node.element_id),
                            label=list(node.labels)[0] if node.labels else "Unknown",
                            name=node.get("name", node.get("article_id", "")),
                            relationship=""
                        ))

            # Add relationship labels to graph path
            for i in range(len(graph_path) - 1):
                graph_path[i].relationship = "RELATED_TO"

        logger.info(f"[retriever] Neo4j returned {len(chunks)} chunks, {len(graph_path)} path nodes")
        return chunks, graph_path

    except Exception as e:
        logger.error(f"[retriever] Neo4j search failed: {e}")
        return [], []


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

async def retrieve(query: str) -> RetrievalResult:
    """
    Run both vector and graph search, combine results.
    This is the main entry point — the rest of the pipeline calls this.
    """
    # Step 1: Vector search
    vector_chunks = await vector_search(query)

    # Step 2: Graph search (using article IDs from vector search for expansion)
    vector_article_ids = list(set(c.article_id for c in vector_chunks if c.article_id))

    logger.info(f"[debug] Extracted raw chunk attributes: {[getattr(c, 'article_id', None) for c in vector_chunks]}")
    vector_article_ids = list(set(c.article_id for c in vector_chunks if c.article_id))
    logger.info(f"[debug] Passing to graph search: {vector_article_ids}")

    graph_chunks, graph_path = await graph_search(query, article_ids=vector_article_ids)

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