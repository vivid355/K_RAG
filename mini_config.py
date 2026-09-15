from pathlib import Path

from langchain_qdrant import RetrievalMode


BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "markdown_docs"

QDRANT_PATH = BASE_DIR / "qdrant_db_stage2"
PARENT_STORE_PATH = BASE_DIR / "parent_store_stage2.json"
COLLECTION = "my_rag_stage2"

DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

MIN_PARENT_SIZE = 800
PARENT_SIZE = 1600

CHILD_SIZE = 500
CHILD_OVERLAP = 100

TOP_K = 7
SCORE_THRESHOLD = 0.0

MAX_ITERATIONS = 5
MAX_TOOL_CALLS = 8

BASE_TOKEN_THRESHOLD = 8000
TOKEN_GROWTH_FACTOR = 0.9
MAIN_HISTORY_MESSAGES_TO_KEEP = 4

SPARSE_MODEL = "Qdrant/bm25"
SPARSE_VECTOR_NAME = "sparse"
QUERY_MODE = RetrievalMode.HYBRID
