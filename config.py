"""Shared settings. Used by ingest.py, rag.py and (later) the FastAPI backend."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------- Paths ----------
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
DEFAULT_PDF = DATA_DIR / "document.pdf"
COLLECTION_NAME = "document"

# ---------- Ingestion ----------
# Small, fast English embedding model (384 dims), runs on CPU via ONNX.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000       # max characters per chunk (~200-250 tokens)
CHUNK_OVERLAP = 150     # characters carried over from the previous chunk
MIN_CHUNK_CHARS = 50    # skip near-empty chunks (page numbers, stray headers)

# ---------- Retrieval ----------
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.5"))  # tune with: python ingest.py --query "..."

# ---------- LLM ----------
# "groq" for the online version, "ollama" for running fully locally.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL") or (
    "openai/gpt-oss-120b" if LLM_PROVIDER == "groq" else "llama3.2:3b"
)
TEMPERATURE = 0.2
MAX_HISTORY_TURNS = 3   # previous Q&A pairs sent to the LLM for follow-up questions
# ---------- Rate limiting (protects the LLM quota on the public demo) ----------
RATE_LIMIT_PER_IP = int(os.getenv("RATE_LIMIT_PER_IP", "10"))          # questions per minute, per visitor
RATE_LIMIT_GLOBAL = int(os.getenv("RATE_LIMIT_GLOBAL", "200"))         # questions per hour, all visitors together