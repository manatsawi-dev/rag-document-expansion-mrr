import os
from typing import Final
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=False)

# ML, AI Model names, Configs
GOOGLE_GEMINI_API_KEY: Final[str] = os.getenv("GOOGLE_GEMINI_API_KEY", "")
GOOGLE_GEMINI_MODEL_NAME: Final[str] = "gemini-2.0-flash"
FASTEMBED_DENSE_MODEL_NAME: Final[str] = "BAAI/bge-small-en-v1.5"
FASTEMBED_BM25_MODEL_NAME: Final[str] = "Qdrant/bm25"

# Qdrant configs
QDRANT_HTTP_HOST: Final[str] = "http://localhost"
QDRANT_HOST: Final[str] = "localhost"
QDRANT_PORT: Final[int] = 6333
VECTOR_SIZE: Final[int] = 384  # for bge-small-en-v1.5
TEST_COLLECTION_NAME: Final[str] = "test_collection"
TEST_HYBRID_COLLECTION_NAME: Final[str] = "test_hybrid_collection"
ORIGINAL_TEXT_COLLECTION_NAME: Final[str] = "original_text"
ORIGINAL_TEXT_AND_EXPANDED_COLLECTION_NAME: Final[str] = "original_and_expanded_text"
ONLY_EXPANDED_COLLECTION_NAME: Final[str] = "only_expanded_text"


# BAAI/bge-small-en-v1.5: 384 dimensions
# BAAI/bge-base-en-v1.5: 768 dimensions
# BAAI/bge-large-en-v1.5: 1024 dimensions
