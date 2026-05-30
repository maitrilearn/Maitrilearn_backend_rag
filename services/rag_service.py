import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN     = os.getenv("HF_TOKEN", "")  # optional but increases rate limit

# Hugging Face Inference API — no model loaded locally, zero RAM cost
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def embed(text: str) -> list:
    """
    Generate 384-dim embedding via HuggingFace Inference API.
    No model loaded locally — solves Render 512MB RAM limit.
    """
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
        raise ValueError(f"HF embedding API error {res.status_code}: {res.text[:200]}")

    data = res.json()

    # HF returns [[...vector...]] — unwrap
    if isinstance(data, list) and isinstance(data[0], list):
        return data[0]

    raise ValueError(f"Unexpected HF response format: {str(data)[:100]}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping word-based chunks."""
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
    """Chunk, embed, store in Supabase pgvector."""
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
    """Search pgvector for chunks most relevant to query."""
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

    if topic:
        topic_lower = topic.lower()
        results = [r for r in results if topic_lower in (r.get("topic", "")).lower()]

    return [r["content"] for r in results]
