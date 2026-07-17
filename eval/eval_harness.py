"""
Evaluation Harness — run QA sample tests through the pipeline API
and report grounding verification metrics (passed, patched, retried, fallback).

Run from project root:
    python eval/eval_harness.py
"""

import httpx
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/chat"

# Sample Egyptian legal QA test cases
TEST_CASES = [
    {
        "query": "ما هي عقوبة السرقة البسيطة في قانون العقوبات المصري؟",
        "expected_article": "318",
        "language": "ar"
    },
    {
        "query": "ما هي عقوبة السرقات التي تقع في الطرق العامة؟",
        "expected_article": "315",
        "language": "ar"
    },
    {
        "query": "ما هو تعريف السارق في القانون المصري؟",
        "expected_article": "318",
        "language": "ar"
    },
    {
        "query": "What is the penalty for simple theft under Egyptian law?",
        "expected_article": "318",
        "language": "en"
    }
]


def run_evaluation():
    logger.info("=" * 60)
    logger.info("EVALUATION HARNESS — STARTING")
    logger.info(f"Targeting: {API_URL}")
    logger.info("=" * 60)

    stats = {
        "total": 0,
        "success": 0,
        "failures": 0,
        "outcomes": {
            "passed": 0,
            "patched": 0,
            "retried": 0,
            "fallback": 0
        },
        "latencies": []
    }

    client = httpx.Client(timeout=30.0)

    for i, test in enumerate(TEST_CASES, 1):
        logger.info(f"Test {i}/{len(TEST_CASES)}: Query: '{test['query']}'")
        start_time = time.time()
        
        try:
            response = client.post(
                API_URL,
                json={
                    "message": test["query"],
                    "language": test["language"],
                    "session_id": f"eval-session-{i}"
                }
            )
            
            latency = time.time() - start_time
            stats["latencies"].append(latency)

            if response.status_code != 200:
                logger.error(f"  ❌ HTTP Error: {response.status_code}")
                stats["failures"] += 1
                continue

            data = response.json()
            outcome = data.get("citation_check_outcome", "passed")
            confidence = data.get("confidence", "medium")
            citations = data.get("citations", [])
            
            stats["total"] += 1
            stats["success"] += 1
            stats["outcomes"][outcome] = stats["outcomes"].get(outcome, 0) + 1

            logger.info(f"  ✅ Complete in {latency:.2f}s")
            logger.info(f"     Outcome: {outcome.upper()} | Confidence: {confidence.upper()}")
            logger.info(f"     Citations found: {[c.get('article_id') for c in citations]}")
            
        except Exception as e:
            logger.error(f"  ❌ Request failed: {e}")
            stats["failures"] += 1

    client.close()

    # ── Report results ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION HARNESS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total queries run successfully: {stats['success']}")
    logger.info(f"Total connection errors: {stats['failures']}")
    
    if stats["success"] > 0:
        avg_latency = sum(stats["latencies"]) / len(stats["latencies"])
        logger.info(f"Average latency: {avg_latency:.2f} seconds")
        
        logger.info("Grounding Verification breakdown:")
        for outcome, count in stats["outcomes"].items():
            percentage = (count / stats["success"]) * 100
            logger.info(f"  - {outcome.upper()}: {count} ({percentage:.1f}%)")
            
        # Grounding safety rate target
        safety_rate = ((stats["outcomes"]["passed"] + stats["outcomes"]["patched"] + stats["outcomes"]["retried"]) / stats["success"]) * 100
        logger.info(f"Deterministic Citation Grounding safety rate: {safety_rate:.1f}%")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    run_evaluation()
