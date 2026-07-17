"""
Qdrant Vector Loader Runner — batch-imports canonical law records into Qdrant.
Generates embeddings using sentence-transformers and uploads:
  1. Chunks (embedded vectors)
  2. Metadata payload (text, law_name, article_id, categories)

Run:
    python -m backend.ingestion.load_qdrant
"""
import os
import sys
import logging
import uuid
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Add workspace root to path for standalone execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from backend.ingestion.normalize import normalize_md_document
from backend.ingestion.chunker import chunk_all
from backend.ingestion.embed import embed_texts

# Resolve MD_LAWS_DIR relative to the script file location so it works from any CWD
MD_LAWS_DIR = Path("../data/laws")
BATCH_SIZE = 100


def load_local_records() -> list[dict]:
    """Parse local canonical markdown law files."""
    records = []
    if not MD_LAWS_DIR.exists():
        logger.error(f"Directory {MD_LAWS_DIR} does not exist.")
        return []
        
    md_files = sorted(MD_LAWS_DIR.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files in {MD_LAWS_DIR}")
    
    for md_path in md_files:
        if md_path.name == "index.md":
            continue
        try:
            content = md_path.read_text(encoding="utf-8")
            parsed = normalize_md_document(str(md_path), content)
            records.extend(r for r in parsed if r.get("text"))
        except Exception as e:
            logger.error(f"Failed to parse {md_path}: {e}")
            
    logger.info(f"Loaded {len(records)} total article records from local files.")
    return records


def build_qdrant(chunks: list[dict]):
    """Embed and upload chunks to Qdrant vector database in batches."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        logger.error("QDRANT_URL and QDRANT_API_KEY not found in environment config.")
        sys.exit(1)
        
    logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
        
        # 1. Generate embeddings
        logger.info("Generating embeddings for all chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        embedding_dim = len(embeddings[0])
        logger.info(f"Embeddings generated successfully. Dimension: {embedding_dim}")
        
        # 2. Recreate collection
        logger.info(f"Recreating Qdrant collection '{QDRANT_COLLECTION}'...")
        try:
            client.delete_collection(QDRANT_COLLECTION)
            logger.info(f"Deleted existing collection: {QDRANT_COLLECTION}")
        except Exception:
            pass
            
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {QDRANT_COLLECTION} (dim={embedding_dim})")
        
        # 3. Upload points in batches
        logger.info(f"Uploading {len(chunks)} chunks to Qdrant in batches of {BATCH_SIZE}...")
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            batch_embeddings = embeddings[i:i + BATCH_SIZE]
            
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
            
            client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=False)
            logger.info(f"Uploaded batch {i // BATCH_SIZE + 1} ({len(points)} points)")
            
        logger.info("Qdrant Vector Store Build Completed Successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during Qdrant import: {e}")


def main():
    records = load_local_records()
    if not records:
        logger.warning("No records to import. Exiting.")
        return
        
    logger.info("Chunking records...")
    chunks = chunk_all(records)
    logger.info(f"Generated {len(chunks)} chunks from {len(records)} records.")
    
    build_qdrant(chunks)


if __name__ == "__main__":
    main()
