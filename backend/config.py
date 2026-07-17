# Try loading ".env" or "env" from project root or current working dir
import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
possible_paths = [
    Path(".env"),
    Path("env"),
    project_root / ".env",
    project_root / "env"
]

loaded = False
for p in possible_paths:
    if p.exists() and p.is_file():
        load_dotenv(dotenv_path=str(p))
        loaded = True
        break

if not loaded:
    load_dotenv()



# ── LLM ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
LLM_MODEL_CHEAP: str = os.getenv("LLM_MODEL_CHEAP", LLM_MODEL)
LLM_MODEL_LITE: str = os.getenv("LLM_MODEL_LITE", "google/gemini-2.5-flash-lite")

# ── Qdrant ───────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "egyptian_law")

# ── Neo4j ────────────────────────────────────────────────────────────────
NEO4J_URI: str = os.getenv("NEO4J_URI", "")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

# ── Embedding ────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# ── Retrieval / Pipeline ────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
TOKEN_BUDGET: int = int(os.getenv("TOKEN_BUDGET", "3000"))
SESSION_MAX_TURNS: int = int(os.getenv("SESSION_MAX_TURNS", "8"))
