"""
Schema Normalization — normalize HF datasets to a unified schema.
Every document must have: law_name, categories, article_id, text.

Dev A owns this file.
"""

import re
import logging

logger = logging.getLogger(__name__)


def normalize_record(record: dict, dataset_name: str) -> dict | None:
    """
    Normalize a single record from any dataset to the shared schema.

    Expected output:
    {
        "law_name": str,
        "article_id": str,
        "categories": list[str],
        "text": str,
    }

    Dev A: Update the field mappings below based on the actual HF dataset schema.
    """

    try:
        if dataset_name == "egypt-legal-corpus":
            # TODO (Dev A): Inspect actual column names and update these mappings
            return {
                "law_name": record.get("law_name", record.get("title", "")),
                "article_id": str(record.get("article_id", record.get("article_number", ""))),
                "categories": _extract_categories(record),
                "text": record.get("text", record.get("content", "")),
            }

        elif dataset_name == "QA_LAW_Egyptian_dataset":
            # This is the eval set — normalize Q&A pairs
            return {
                "law_name": record.get("law_name", ""),
                "article_id": str(record.get("article_id", "")),
                "categories": _extract_categories(record),
                "text": record.get("question", "") + "\n" + record.get("answer", ""),
            }

        else:
            logger.warning(f"Unknown dataset: {dataset_name}")
            return None

    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return None


def _extract_categories(record: dict) -> list[str]:
    """Extract categories from various possible field names."""
    cats = record.get("categories", record.get("category", record.get("topic", [])))
    if isinstance(cats, str):
        return [c.strip() for c in cats.split(",")]
    if isinstance(cats, list):
        return cats
    return []


def normalize_md_document(filepath: str, content: str) -> list[dict]:
    """
    Parse a markdown legal document into article-level records.
    
    Dev A: Adjust parsing logic based on actual .md file structure.
    """
    records = []

    # Try to extract law name from filename or first heading
    law_name = filepath.split("/")[-1].replace(".md", "").replace("_", " ")

    # Split by article headers (common patterns)
    # Pattern: "## المادة 5" or "## مادة (5)" or "## Article 5"
    article_pattern = re.compile(
        r"^##?\s*(?:ال)?(?:مادة|Article)\s*\(?\s*(\d+)\s*\)?",
        re.MULTILINE | re.UNICODE,
    )

    splits = article_pattern.split(content)

    if len(splits) <= 1:
        # No article headers found — treat entire doc as one record
        records.append({
            "law_name": law_name,
            "article_id": "0",
            "categories": [],
            "text": content.strip(),
        })
    else:
        # splits alternates: [preamble, article_num_1, text_1, article_num_2, text_2, ...]
        for i in range(1, len(splits), 2):
            article_id = splits[i]
            text = splits[i + 1].strip() if i + 1 < len(splits) else ""
            if text:
                records.append({
                    "law_name": law_name,
                    "article_id": article_id,
                    "categories": [],
                    "text": text,
                })

    logger.info(f"[normalize] parsed {filepath}: {len(records)} articles")
    return records
