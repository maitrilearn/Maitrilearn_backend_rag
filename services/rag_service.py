import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_KEY      = os.getenv("SUPABASE_KEY")
JINA_API_KEY      = os.getenv("JINA_API_KEY", "")
LOCAL_EMBED_URL   = os.getenv("LOCAL_EMBED_URL", "")

def embed(text: str) -> list:
    """
    Generate embedding.
    Priority:
    1. Local embedding service
    2. Jina AI
    """

    # Local embedding service (optional)
    if LOCAL_EMBED_URL:
        try:
            res = requests.post(
                f"{LOCAL_EMBED_URL}/embed",
                json={"text": text},
                timeout=15
            )

            if res.status_code == 200:
                return res.json()["embedding"]

            print(f"[rag] Local embed failed: {res.status_code}")

        except Exception as e:
            print(f"[rag] Local embed unreachable: {e}")

    # Jina AI fallback
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    res = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers=headers,
        json={
            "model": "jina-embeddings-v5-text-small",
            "input": [text]
        },
        timeout=30
    )

    if res.status_code != 200:
        raise ValueError(
            f"Jina API error {res.status_code}: {res.text[:200]}"
        )

    data = res.json()

    return data["data"][0]["embedding"]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
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
            print(f"[rag] Embedding failed: {e}")
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


def search_chunks(query: str, topic: str = None, top_k: int = 5, threshold: float = 0.3) -> list:
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
        json={"query_embedding": query_vector, "match_threshold": threshold, "match_count": top_k},
        timeout=15
    )

    if res.status_code != 200:
        print(f"[rag] Search failed: {res.status_code}")
        return []

    results = res.json()
    if topic:
        results = [r for r in results if topic.lower() in r.get("topic", "").lower()]

    return [r["content"] for r in results]
