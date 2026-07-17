"""
Neo4j Graph Loader Runner — batch-imports canonical law records into Neo4j.
Uses optimized Cypher queries with UNWIND batching to import:
  1. Law nodes
  2. Article nodes and CONTAINS edges
  3. REFERENCES cross-reference edges

Run:
    python -m backend.ingestion.load_neo4j
"""
import os
import sys
import logging
import asyncio
from pathlib import Path
from neo4j import GraphDatabase

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Add workspace root to path for standalone execution
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from backend.ingestion.normalize import normalize_md_document
from backend.ingestion.extract_graph import process_articles, get_cache_key

MD_LAWS_DIR = Path("../data/laws")
BATCH_SIZE = 1000


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


def build_graph(records: list[dict], extracted_data: dict = None):
    """Optimized batch graph builder using UNWIND strategy."""
    if not NEO4J_URI or not NEO4J_PASSWORD:
        logger.error("NEO4J_URI and NEO4J_PASSWORD not found in environment config.")
        sys.exit(1)
        
    logger.info(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # 1. Clear database
            logger.info("Clearing existing Neo4j graph database...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully.")
            
            # Setup constraint/index for fast semantic entity lookup
            try:
                session.run("CREATE CONSTRAINT semantic_entity_name_unique IF NOT EXISTS FOR (s:SemanticEntity) REQUIRE s.name IS UNIQUE")
            except Exception:
                try:
                    session.run("CREATE INDEX semantic_entity_name_idx IF NOT EXISTS FOR (s:SemanticEntity) ON (s.name)")
                except Exception:
                    pass
            
            # 2. Extract and create Law nodes
            logger.info("Importing Law nodes...")
            laws = list(set(r["law_name"] for r in records if r.get("law_name")))
            
            law_query = """
            UNWIND $laws AS law_name
            MERGE (l:Law {name: law_name})
            """
            session.run(law_query, laws=laws)
            logger.info(f"Successfully merged {len(laws)} Law nodes.")
            
            # 3. Create Article nodes and CONTAINS edges in batches
            logger.info("Importing Article nodes and CONTAINS edges...")
            articles_payload = [
                {
                    "law_name": r["law_name"],
                    "article_id": r["article_id"],
                    "text": r["text"][:500],  # cap length for graph UI display
                    "categories": r["categories"]
                }
                for r in records if r.get("law_name") and r.get("article_id")
            ]
            
            article_query = """
            UNWIND $rows AS row
            MATCH (l:Law {name: row.law_name})
            MERGE (a:Article {article_id: row.article_id, law_name: row.law_name})
            SET a.text = row.text,
                a.categories = row.categories
            MERGE (l)-[:CONTAINS]->(a)
            """
            
            for i in range(0, len(articles_payload), BATCH_SIZE):
                batch = articles_payload[i:i + BATCH_SIZE]
                session.run(article_query, rows=batch)
                logger.info(f"Merged Articles batch {i // BATCH_SIZE + 1} ({len(batch)} nodes)")
                
            # 4. Create REFERENCES cross-reference edges
            logger.info("Importing REFERENCES edges...")
            refs_payload = []
            for r in records:
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
            
            for i in range(0, len(refs_payload), BATCH_SIZE):
                batch = refs_payload[i:i + BATCH_SIZE]
                session.run(ref_query, refs=batch)
                logger.info(f"Merged REFERENCES batch {i // BATCH_SIZE + 1} ({len(batch)} edges)")
                
            # 5. Import Semantic Entities and Relationships from LLM
            if extracted_data:
                logger.info("Importing Semantic Entities and Relationships from LLM extraction...")
                
                # Split entities by type
                entity_types = {
                    "Authority": [],
                    "Obligation": [],
                    "Right": [],
                    "Penalty": [],
                    "Concept": []
                }
                seen_entities = set()
                
                # Gather all entities from the cache
                for record in records:
                    key = f"{record['law_name']}_{record['article_id']}"
                    art_data = extracted_data.get(key)
                    if not art_data:
                        continue
                    
                    for ent in art_data.get("entities") or []:
                        ent_type = ent.get("type")
                        ent_name = ent.get("name")
                        if not ent_name or not ent_type or ent_type not in entity_types:
                            continue
                        
                        # Dedup by name
                        if ent_name not in seen_entities:
                            seen_entities.add(ent_name)
                            entity_types[ent_type].append({
                                "name": ent_name,
                                "description": ent.get("description", "")
                            })
                
                # Create Semantic Entities by Label
                for ent_type, entities in entity_types.items():
                    if not entities:
                        continue
                    
                    # Merge nodes with dual labels: the specific type and generic SemanticEntity
                    entity_query = f"""
                    UNWIND $entities AS ent
                    MERGE (e:{ent_type}:SemanticEntity {{name: ent.name}})
                    SET e.description = ent.description
                    """
                    session.run(entity_query, entities=entities)
                    logger.info(f"Merged {len(entities)} {ent_type} nodes.")
                
                # Prepare relationship queries
                article_rels = {}  # rel_type -> list
                entity_rels = {}   # rel_type -> list
                
                for record in records:
                    key = f"{record['law_name']}_{record['article_id']}"
                    art_data = extracted_data.get(key)
                    if not art_data:
                        continue
                    
                    for rel in art_data.get("relationships") or []:
                        source = rel.get("source")
                        target = rel.get("target")
                        rel_type = rel.get("type")
                        
                        if not source or not target or not rel_type:
                            continue
                        
                        # Clean relationship type to keep alphanumeric + underscore, uppercase
                        rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_").upper()
                        if not rel_type:
                            rel_type = "RELATED_TO"
                            
                        if source == "ARTICLE_NODE":
                            if rel_type not in article_rels:
                                article_rels[rel_type] = []
                            article_rels[rel_type].append({
                                "law_name": record["law_name"],
                                "article_id": record["article_id"],
                                "target_name": target
                            })
                        else:
                            if rel_type not in entity_rels:
                                entity_rels[rel_type] = []
                            entity_rels[rel_type].append({
                                "source_name": source,
                                "target_name": target
                            })
                
                # Merge Article-to-Entity Relationships
                for rel_type, rels in article_rels.items():
                    rel_query = f"""
                    UNWIND $rels AS rel
                    MATCH (a:Article {{article_id: rel.article_id, law_name: rel.law_name}})
                    MATCH (t:SemanticEntity {{name: rel.target_name}})
                    MERGE (a)-[:{rel_type}]->(t)
                    """
                    session.run(rel_query, rels=rels)
                    logger.info(f"Merged {len(rels)} relationships of type ARTICLE_NODE -[:{rel_type}]-> Entity.")
                
                # Merge Entity-to-Entity Relationships
                for rel_type, rels in entity_rels.items():
                    rel_query = f"""
                    UNWIND $rels AS rel
                    MATCH (s:SemanticEntity {{name: rel.source_name}})
                    MATCH (t:SemanticEntity {{name: rel.target_name}})
                    MERGE (s)-[:{rel_type}]->(t)
                    """
                    session.run(rel_query, rels=rels)
                    logger.info(f"Merged {len(rels)} relationships of type Entity -[:{rel_type}]-> Entity.")
                
            logger.info("Neo4j Graph Build Completed Successfully!")
            
    except Exception as e:
        logger.error(f"An error occurred during graph import: {e}")
    finally:
        driver.close()


def main():
    records = load_local_records()
    if not records:
        logger.warning("No records to import. Exiting.")
        return
        
    limit_env = os.getenv("LIMIT_LLM_EXTRACTION")
    limit = int(limit_env) if limit_env else None
    
    logger.info("Starting LLM entity extraction...")
    extracted_data = asyncio.run(process_articles(records, limit=limit))
    
    build_graph(records, extracted_data)


if __name__ == "__main__":
    main()
