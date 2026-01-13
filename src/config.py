import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DAT_DIR = os.path.join(PROJECT_ROOT, "data")
DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
USER_DATA_DIR = os.path.join(PROJECT_ROOT, "user_data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 1.0
