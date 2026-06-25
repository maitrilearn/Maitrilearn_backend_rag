import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
LOCAL_EMBED_URL = os.getenv("LOCAL_EMBED_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def embed(text: str) -> list:
    """Generate embedding — tries local container, then Jina, then HF fallback."""

    # 1. Local container (fastest, free)
    if LOCAL_EMBED_URL:
        try:
            res = requests.post(
                f"{LOCAL_EMBED_URL}/embed",
                json={"text": text}, timeout=15
            )
            if res.status_code == 200:
                return res.json()["embedding"]
        except Exception as e:
            print(f"[rag] Local embed failed: {e}")

    # 2. Jina AI
    if JINA_API_KEY:
        try:
            res = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model":      "jina-embeddings-v3",
                    "input":      [text],
                    "dimensions": 384,        # match pgvector table vector(384)
                    "task":       "retrieval.query"
                },
                timeout=30
            )
            if res.status_code == 200:
                return res.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"[rag] Jina embed failed: {e}")

    # 3. HuggingFace fallback
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    res = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30
    )
    if res.status_code != 200:
        raise ValueError(f"All embedding providers failed. Last error: {res.status_code}")

    data = res.json()
    if isinstance(data, list) and isinstance(data[0], list):
        return data[0]
    raise ValueError(f"Unexpected HF response: {str(data)[:100]}")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 40) -> list:
    """
    Split text into overlapping chunks.
    TUNED: 300 words (was 500) — smaller chunks = more precise retrieval
    TUNED: 40 words overlap (was 50) — proportional to smaller size
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
        start = end - overlap
    return chunks


def ingest_text(text: str, topic: str, source: str) -> dict:
    """Chunk, embed and store in Supabase pgvector."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY not set")

    text   = re.sub(r"\s+", " ", text).strip()
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No content extracted from document")

    stored  = 0
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal"
    }

    for chunk in chunks:
        try:
            vector = embed(chunk)
        except Exception as e:
            print(f"[rag] Embedding failed for chunk: {e}")
            continue

        res = requests.post(
            f"{SUPABASE_URL}/rest/v1/documents",
            headers=headers,
            json={"topic": topic, "content": chunk, "embedding": vector, "source": source},
            timeout=15
        )
        if res.status_code in (200, 201):
            stored += 1
        else:
            print(f"[rag] Insert failed: {res.status_code} {res.text[:100]}")

    return {"chunks_stored": stored, "total_chunks": len(chunks)}


def search_chunks(
    query:     str,
    topic:     str  = None,
    top_k:     int  = 8,       # TUNED: was 5
    threshold: float = 0.2     # TUNED: was 0.25 — lower = more results
) -> list:
    """
    Search pgvector for relevant chunks.
    Returns list of dicts with content + source for citation.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    try:
        query_vector = embed(query)
    except Exception as e:
        print(f"[rag] Query embedding failed: {e}")
        return []

    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json"
    }

    res = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_documents",
        headers=headers,
        json={
            "query_embedding": query_vector,
            "match_threshold": threshold,
            "match_count":     top_k
        },
        timeout=15
    )

    if res.status_code != 200:
        print(f"[rag] Search failed: {res.status_code} {res.text[:200]}")
        return []

    results = res.json()

    # Filter by topic if specified
    if topic:
        topic_lower = topic.lower()
        results = [r for r in results
                   if topic_lower in (r.get("topic", "")).lower()]

    # Return full objects (content + source) for citation
    return results
