"""
Main Ingestion Script — end-to-end pipeline:
  Load HF datasets + local canonical .md files → Normalize → Chunk → Embed
  → Upload to Qdrant + Neo4j

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

# Directory containing the canonical .md law files (one file per law).
# TODO (Dev A): point this at wherever the canonical files actually live.
MD_LAWS_DIR = Path("backend/data/laws")


def main():
    """Run the full ingestion pipeline."""

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE — START")
    logger.info("=" * 60)

    # ── Step 1: Load HF Datasets ─────────────────────────────────────
    logger.info("[1/6] Loading HuggingFace datasets...")

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
    logger.info("[2/6] Normalizing records...")

    from backend.ingestion.normalize import normalize_record, normalize_md_document

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

    logger.info(f"  Normalized (HF datasets): {len(all_records)} records")

    # ── Step 2b: Load + normalize local canonical .md law files ─────
    logger.info("[2b/6] Loading canonical .md law files...")

    md_records = []
    if MD_LAWS_DIR.exists():
        md_files = sorted(MD_LAWS_DIR.glob("*.md"))
        for md_path in md_files:
            try:
                content = md_path.read_text(encoding="utf-8")
                parsed = normalize_md_document(str(md_path), content)
                md_records.extend(parsed)
            except Exception as e:
                logger.error(f"  Failed to parse {md_path}: {e}")
        logger.info(f"  Parsed {len(md_files)} md file(s) → {len(md_records)} article records")
    else:
        logger.info(f"  {MD_LAWS_DIR} not found — skipping md ingestion")

    all_records.extend(r for r in md_records if r.get("text"))
    logger.info(f"  Total normalized records (HF + md): {len(all_records)}")

    if not all_records:
        logger.error("  No records to process! Check dataset IDs, field mappings, and MD_LAWS_DIR.")
        sys.exit(1)

    # ── Step 3: Chunk ────────────────────────────────────────────────
    logger.info("[3/6] Chunking documents...")

    from backend.ingestion.chunker import chunk_all

    all_chunks = chunk_all(all_records)
    logger.info(f"  Total chunks: {len(all_chunks)}")

    # ── Step 4: Embed & Upload to Qdrant ─────────────────────────────
    logger.info("[4/6] Embedding and uploading to Qdrant...")

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
    logger.info("[5/6] Building Neo4j graph (Law/Article nodes + CONTAINS edges)...")

    from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    if not NEO4J_URI or not NEO4J_PASSWORD:
        logger.error("  NEO4J_URI and NEO4J_PASSWORD must be set in .env")
        logger.info("  Skipping Neo4j graph build.")
    else:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                # 1. Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("  Cleared existing Neo4j graph data")

                # 2. Create Law nodes
                logger.info("  Importing Law nodes...")
                laws = list(set(r["law_name"] for r in all_records if r.get("law_name")))
                law_query = """
                UNWIND $laws AS law_name
                MERGE (l:Law {name: law_name})
                """
                session.run(law_query, laws=laws)
                logger.info(f"  Created {len(laws)} Law nodes")

                # 3. Create Article nodes and CONTAINS edges in batches
                logger.info("  Importing Article nodes and CONTAINS edges...")
                batch_size = 1000
                articles_payload = [
                    {
                        "law_name": r["law_name"],
                        "article_id": r["article_id"],
                        "text": r["text"][:500],  # store first 500 chars in graph
                        "categories": r["categories"]
                    }
                    for r in all_records if r.get("law_name") and r.get("article_id")
                ]

                article_query = """
                UNWIND $rows AS row
                MATCH (l:Law {name: row.law_name})
                MERGE (a:Article {article_id: row.article_id, law_name: row.law_name})
                SET a.text = row.text,
                    a.categories = row.categories
                MERGE (l)-[:CONTAINS]->(a)
                """

                for i in range(0, len(articles_payload), batch_size):
                    batch = articles_payload[i:i + batch_size]
                    session.run(article_query, rows=batch)
                    logger.info(f"  Created Articles batch {i // batch_size + 1} ({len(batch)} nodes)")

                # ── Step 6: REFERENCES edges ──────────────────────────────
                logger.info("[6/6] Building REFERENCES edges...")
                refs_payload = []
                for r in all_records:
                    refs = r.get("references") or []
                    if not refs or not r.get("article_id") or not r.get("law_name"):
                        continue
                    for ref_law_name in refs:
                        refs_payload.append({
                            "law_name": r["law_name"],
                            "article_id": r["article_id"],
                            "target_law_name": ref_law_name
                        })

                ref_query = """
                UNWIND $refs AS ref
                MATCH (a:Article {article_id: ref.article_id, law_name: ref.law_name})
                MERGE (target:Law {name: ref.target_law_name})
                MERGE (a)-[:REFERENCES]->(target)
                """

                for i in range(0, len(refs_payload), batch_size):
                    batch = refs_payload[i:i + batch_size]
                    session.run(ref_query, refs=batch)
                    logger.info(f"  Created REFERENCES batch {i // batch_size + 1} ({len(batch)} edges)")

        except Exception as e:
            logger.error(f"  Neo4j graph build failed: {e}")
        finally:
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