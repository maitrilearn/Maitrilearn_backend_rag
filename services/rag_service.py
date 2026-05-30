import os
import re
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Load embedding model once at startup
# all-MiniLM-L6-v2: free, fast, 384 dimensions, runs on Render
print("[rag] Loading embedding model...")
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("[rag] Embedding model ready")


def embed(text: str) -> list:
    """Generate 384-dim embedding for a text string."""
    return _model.encode(text, normalize_embeddings=True).tolist()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks by word count.
    chunk_size=500 words, overlap=50 words between chunks.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap  # overlap for context continuity

    return chunks


def ingest_text(text: str, topic: str, source: str) -> dict:
    """
    Chunk text, embed each chunk, store in Supabase pgvector.
    Returns count of chunks stored.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY not set")

    # Clean text
    text = re.sub(r"\s+", " ", text).strip()

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No content extracted from document")

    stored = 0
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal"
    }

    for chunk in chunks:
        vector = embed(chunk)
        payload = {
            "topic":     topic,
            "content":   chunk,
            "embedding": vector,
            "source":    source
        }
        res = requests.post(
            f"{SUPABASE_URL}/rest/v1/documents",
            headers=headers,
            json=payload,
            timeout=15
        )
        if res.status_code in (200, 201):
            stored += 1
        else:
            print(f"[rag] Insert failed: {res.status_code} {res.text[:100]}")

    return {"chunks_stored": stored, "total_chunks": len(chunks)}


def search_chunks(query: str, topic: str = None, top_k: int = 5, threshold: float = 0.3) -> list:
    """
    Search pgvector for chunks most relevant to query.
    Optionally filter by topic.
    Returns list of content strings.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    query_vector = embed(query)

    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json"
    }

    payload = {
        "query_embedding": query_vector,
        "match_threshold": threshold,
        "match_count":     top_k
    }

    res = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_documents",
        headers=headers,
        json=payload,
        timeout=15
    )

    if res.status_code != 200:
        print(f"[rag] Search failed: {res.status_code} {res.text[:200]}")
        return []

    results = res.json()

    # Filter by topic if specified
    if topic:
        topic_lower = topic.lower()
        results = [r for r in results if topic_lower in (r.get("topic","")).lower()]

    return [r["content"] for r in results]
