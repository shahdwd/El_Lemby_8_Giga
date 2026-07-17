import os
import json
import logging
import asyncio
import hashlib
from pathlib import Path
from backend.config import LLM_MODEL_LITE
from backend.llm_client import call_llm

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "extracted_entities.json"

SYSTEM_PROMPT = """You are an expert legal analyst specializing in Egyptian law.
Analyze the provided Arabic legal article text and extract entities and relationships to build a semantic legal graph.

You MUST extract entities of the following types:
1. "Authority" (e.g., جهة حكومية، وزير، محكمة، مأمور الضبط) - Government bodies, officials, courts, or law enforcement entities.
2. "Obligation" (e.g., التزام بتقديم إقرار، دفع الضريبة) - Mandatory legal duties, actions, or requirements.
3. "Right" (e.g., حق الإعفاء، حق التظلم) - Entitlements, exemptions, or rights granted to individuals/entities.
4. "Penalty" (e.g., عقوبة الحبس، غرامة مالية) - Fines, imprisonment, or other legal sanctions.
5. "Concept" (e.g., الضريبة العقارية، القيمة الإيجارية، مكلف) - Key legal terms, concepts, or subject matter.

You MUST extract relationships. The relationships should connect the Article (refer to it as "ARTICLE_NODE") or connect the extracted entities to each other:
- (ARTICLE_NODE)-[:EMPOWERS]->(Authority)
- (ARTICLE_NODE)-[:IMPOSES]->(Obligation)
- (ARTICLE_NODE)-[:GRANTS]->(Right)
- (ARTICLE_NODE)-[:PRESCRIBES]->(Penalty)
- (ARTICLE_NODE)-[:MENTIONS]->(Concept)
- Custom relationships between extracted entities are also allowed (e.g., Authority-[:ENFORCES]->Obligation, Penalty-[:APPLIES_TO]->Obligation).

Output ONLY a valid JSON object with the following structure (do not include any conversational filler, markdown formatting block markers like ```json, or comments):
{
  "entities": [
    {"type": "Authority|Obligation|Right|Penalty|Concept", "name": "Entity Name in Arabic", "description": "Brief description of the entity in Arabic"}
  ],
  "relationships": [
    {"source": "ARTICLE_NODE or Entity Name", "type": "EMPOWERS|IMPOSES|GRANTS|PRESCRIBES|MENTIONS|ENFORCES|APPLIES_TO", "target": "Entity Name"}
  ]
}
"""

def get_cache_key(law_name: str, article_id: str) -> str:
    """Generate a clean, readable cache key for the article."""
    return f"{law_name}_{article_id}"

def load_cache() -> dict:
    """Load the extracted entities cache from disk."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load cache file: {e}")
    return {}

def save_cache(cache: dict):
    """Save the updated cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(cache)} entries to entity extraction cache.")
    except Exception as e:
        logger.error(f"Failed to save cache file: {e}")

async def extract_for_article(article_text: str, sem: asyncio.Semaphore) -> dict:
    """Call Gemini to extract entities and relationships for a single article."""
    async with sem:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text:\n{article_text}"}
        ]
        
        for attempt in range(3):
            try:
                response = await call_llm(messages, model=LLM_MODEL_LITE, temperature=0.1, max_tokens=4096)
                
                # Robust JSON cleaning: strip backticks and whitespace
                cleaned_resp = response.strip()
                if cleaned_resp.startswith("```"):
                    # Strip ```json or ```
                    lines = cleaned_resp.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned_resp = "\n".join(lines).strip()
                
                parsed = json.loads(cleaned_resp)
                
                # Basic validation
                if "entities" not in parsed or "relationships" not in parsed:
                    raise ValueError("JSON response missing required keys")
                
                return parsed
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to parse extract for article. Error: {e}")
                await asyncio.sleep(1)
        
        # Return empty schema if failed after retries
        return {"entities": [], "relationships": []}

async def process_articles(records: list[dict], limit: int | None = None) -> dict:
    """Process articles in batches, using cached results where available and querying the LLM for the rest."""
    cache = load_cache()
    sem = asyncio.Semaphore(5)  # Limit concurrent calls to 5 to prevent rate limits
    
    # Filter records to process
    to_process = []
    for r in records:
        if not r.get("text") or not r.get("law_name") or not r.get("article_id"):
            continue
        key = get_cache_key(r["law_name"], r["article_id"])
        if key not in cache:
            to_process.append(r)
            
    if limit is not None:
        to_process = to_process[:limit]
        
    if to_process:
        logger.info(f"Extracting entities for {len(to_process)} new articles (out of {len(records)} total) using {LLM_MODEL_LITE}...")
        
        batch_size = 50
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]
            logger.info(f"Processing extraction batch {i // batch_size + 1}/{(len(to_process) + batch_size - 1) // batch_size} ({len(batch)} articles)...")
            
            tasks = [extract_for_article(r["text"], sem) for r in batch]
            results = await asyncio.gather(*tasks)
            
            # Merge new results into cache
            for r, res in zip(batch, results):
                key = get_cache_key(r["law_name"], r["article_id"])
                cache[key] = res
                
            # Save cache incrementally
            save_cache(cache)
    else:
        logger.info("All articles found in entity extraction cache. No LLM calls needed.")
        
    return cache
