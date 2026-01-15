import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DAT_DIR = os.path.join(PROJECT_ROOT, "data")
DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
USER_DATA_DIR = os.path.join(PROJECT_ROOT, "user_data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MODEL_NAME = "gpt-5-mini"
USE_OPENAI_EMBEDDINGS = False
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
TEMPERATURE = 1.0
