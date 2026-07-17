"""
Main Ingestion Script — end-to-end pipeline:
  Load HF datasets → Normalize → Chunk → Embed → Upload to Qdrant + Neo4j

Run from project root:
    python -m backend.ingestion.ingest

Dev A owns this file.
"""

import sys
import logging
import uuid
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run the full ingestion pipeline."""

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE — START")
    logger.info("=" * 60)

    # ── Step 1: Load HF Datasets ─────────────────────────────────────
    logger.info("[1/5] Loading HuggingFace datasets...")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        sys.exit(1)

    # TODO (Dev A): Replace with actual HF dataset IDs
    # Example: load_dataset("org/egypt-legal-corpus")
    try:
        # Primary legal corpus
        legal_corpus = load_dataset(
            "ahmed-moustafa/egypt-legal-corpus",  # TODO: verify actual ID
            split="train",
        )
        logger.info(f"  Legal corpus: {len(legal_corpus)} records")
    except Exception as e:
        logger.error(f"  Failed to load legal corpus: {e}")
        logger.info("  Continuing with mock data for testing...")
        legal_corpus = None

    try:
        # QA eval dataset
        qa_dataset = load_dataset(
            "ahmed-moustafa/QA_LAW_Egyptian_dataset",  # TODO: verify actual ID
            split="train",
        )
        logger.info(f"  QA dataset: {len(qa_dataset)} records")
    except Exception as e:
        logger.error(f"  Failed to load QA dataset: {e}")
        qa_dataset = None

    # ── Step 2: Normalize ────────────────────────────────────────────
    logger.info("[2/5] Normalizing records...")

    from backend.ingestion.normalize import normalize_record

    all_records = []

    if legal_corpus:
        for record in legal_corpus:
            normalized = normalize_record(dict(record), "egypt-legal-corpus")
            if normalized and normalized.get("text"):
                all_records.append(normalized)

    if qa_dataset:
        for record in qa_dataset:
            normalized = normalize_record(dict(record), "QA_LAW_Egyptian_dataset")
            if normalized and normalized.get("text"):
                all_records.append(normalized)

    logger.info(f"  Normalized: {len(all_records)} records")

    if not all_records:
        logger.error("  No records to process! Check dataset IDs and field mappings.")
        sys.exit(1)

    # ── Step 3: Chunk ────────────────────────────────────────────────
    logger.info("[3/5] Chunking documents...")

    from backend.ingestion.chunker import chunk_all

    all_chunks = chunk_all(all_records)
    logger.info(f"  Total chunks: {len(all_chunks)}")

    # ── Step 4: Embed & Upload to Qdrant ─────────────────────────────
    logger.info("[4/5] Embedding and uploading to Qdrant...")

    from backend.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
    from backend.ingestion.embed import embed_texts

    if not QDRANT_URL or not QDRANT_API_KEY:
        logger.error("  QDRANT_URL and QDRANT_API_KEY must be set in .env")
        logger.info("  Skipping Qdrant upload.")
    else:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Get embedding dimension
        texts = [c["text"] for c in all_chunks]
        embeddings = embed_texts(texts)
        embedding_dim = len(embeddings[0])

        # Create collection (recreate if exists)
        try:
            client.delete_collection(QDRANT_COLLECTION)
            logger.info(f"  Deleted existing collection: {QDRANT_COLLECTION}")
        except Exception:
            pass

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        logger.info(f"  Created collection: {QDRANT_COLLECTION} (dim={embedding_dim})")

        # Upload in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "text": chunk["text"],
                        "law_name": chunk["law_name"],
                        "article_id": chunk["article_id"],
                        "categories": chunk["categories"],
                        "chunk_id": chunk.get("chunk_id", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                    },
                )
                for chunk, emb in zip(batch_chunks, batch_embeddings)
            ]

            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            logger.info(f"  Uploaded batch {i // batch_size + 1} ({len(points)} points)")

        logger.info(f"  Qdrant upload complete: {len(all_chunks)} points")

    # ── Step 5: Build Neo4j Graph ────────────────────────────────────
    logger.info("[5/5] Building Neo4j graph...")

    from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    if not NEO4J_URI or not NEO4J_PASSWORD:
        logger.error("  NEO4J_URI and NEO4J_PASSWORD must be set in .env")
        logger.info("  Skipping Neo4j graph build.")
    else:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("  Cleared existing graph data")

            # Create Law nodes
            laws = set(r["law_name"] for r in all_records if r["law_name"])
            for law_name in laws:
                session.run(
                    "CREATE (l:Law {name: $name})",
                    name=law_name,
                )
            logger.info(f"  Created {len(laws)} Law nodes")

            # Create Article nodes with CONTAINS edges
            article_count = 0
            for record in all_records:
                if record["law_name"] and record["article_id"]:
                    session.run(
                        """
                        MATCH (l:Law {name: $law_name})
                        CREATE (a:Article {
                            article_id: $article_id,
                            law_name: $law_name,
                            text: $text,
                            categories: $categories
                        })
                        CREATE (l)-[:CONTAINS]->(a)
                        """,
                        law_name=record["law_name"],
                        article_id=record["article_id"],
                        text=record["text"][:500],  # store first 500 chars in graph
                        categories=record["categories"],
                    )
                    article_count += 1

            logger.info(f"  Created {article_count} Article nodes with CONTAINS edges")

            # TODO (Dev A — P2): Add cross-reference REFERENCES edges
            # Parse "as amended by..." or "المعدل بموجب..." text patterns
            # to create REFERENCES edges between articles

        driver.close()
        logger.info("  Neo4j graph build complete")

    # ── Done ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE — COMPLETE")
    logger.info(f"  Records: {len(all_records)}")
    logger.info(f"  Chunks: {len(all_chunks)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
