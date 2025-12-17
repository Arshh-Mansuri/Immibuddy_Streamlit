import os
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment")

# ---------------- MODELS ----------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# ---------------- VECTOR STORE ----------------
PINECONE_INDEX_NAME = "rpl-mvp"
PINECONE_DIMENSION = 1536
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# ---------------- CHUNKING CONFIG ----------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# ---------------- DATA PATHS ----------------
DATA_DIR = "data"

PDF_SOURCES = [
    {
        "filename": "UTS_RPL.pdf",
        "topic": "RPL",
        "institution": "UTS",
        "source_type": "University policy",
    },
    {
        "filename": "MQ_RPL.pdf",
        "topic": "RPL",
        "institution": "Macquarie University",
        "source_type": "University policy",
    },
    {
        "filename": "My_RPL_Experience.pdf",
        "topic": "RPL",
        "institution": "Personal",
        "source_type": "User experience",
    },
]