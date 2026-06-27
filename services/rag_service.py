import os
import re
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# Environment Variables
# ==========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

LOCAL_EMBED_URL = os.getenv("LOCAL_EMBED_URL", "").strip()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

HF_API_URL = (
    "https://api-inference.huggingface.co/"
    "pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 40))

# ==========================================================
# Logging
# ==========================================================

logger = logging.getLogger("maitrilearn.rag")

# ==========================================================
# Retry Helper
# ==========================================================

def request_with_retry(
    method,
    url,
    retries=3,
    delay=2,
    **kwargs
):
    """
    Retry HTTP requests for temporary failures.
    """

    last_error = None

    for attempt in range(1, retries + 1):

        try:
            response = requests.request(
                method,
                url,
                timeout=30,
                **kwargs
            )

            if response.status_code < 500:
                return response

            logger.warning(
                f"[Retry {attempt}/{retries}] "
                f"{url} returned {response.status_code}"
            )

        except Exception as e:
            last_error = e
            logger.warning(
                f"[Retry {attempt}/{retries}] {url} -> {e}"
            )

        if attempt < retries:
            time.sleep(delay)

    if last_error:
        raise last_error

    raise Exception(f"Request failed after {retries} retries")

# ==========================================================
# Embedding
# ==========================================================

def embed(text: str) -> list:
    """
    Embedding Provider Priority

    1. Local
    2. Jina AI
    3. HuggingFace
    """

    # ------------------------------------------------------
    # LOCAL
    # ------------------------------------------------------

    if LOCAL_EMBED_URL:

        logger.info("[RAG] Trying LOCAL embedding")

        try:

            res = request_with_retry(
                "POST",
                f"{LOCAL_EMBED_URL}/embed",
                json={
                    "text": text
                }
            )

            logger.info(
                f"[RAG] Local Status {res.status_code}"
            )

            if res.status_code == 200:

                data = res.json()

                if "embedding" in data:

                    logger.info(
                        "[RAG] ✅ Using LOCAL Embedding"
                    )

                    return data["embedding"]

                logger.warning(
                    "[RAG] Local response missing embedding"
                )

        except Exception as e:

            logger.warning(
                f"[RAG] Local failed : {e}"
            )

    # ------------------------------------------------------
    # JINA
    # ------------------------------------------------------

    if JINA_API_KEY:

        logger.info("[RAG] Trying Jina Embedding")

        try:

            res = request_with_retry(
                "POST",
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "jina-embeddings-v3",
                    "input": [text],
                    "task": "retrieval.query",
                    "dimensions": 384
                }
            )

            logger.info(
                f"[RAG] Jina Status {res.status_code}"
            )

            if res.status_code == 200:

                body = res.json()

                if (
                    "data" in body
                    and len(body["data"]) > 0
                    and "embedding" in body["data"][0]
                ):

                    logger.info(
                        "[RAG] ✅ Using Jina Embedding"
                    )

                    return body["data"][0]["embedding"]

                logger.warning(
                    "[RAG] Invalid Jina Response"
                )

        except Exception as e:

            logger.warning(
                f"[RAG] Jina failed : {e}"
            )

    # ------------------------------------------------------
    # HuggingFace
    # ------------------------------------------------------

    logger.info("[RAG] Trying HuggingFace Embedding")

    headers = {
        "Content-Type": "application/json"
    }

    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    res = request_with_retry(
        "POST",
        HF_API_URL,
        headers=headers,
        json={
            "inputs": text,
            "options": {
                "wait_for_model": True
            }
        }
    )

    logger.info(
        f"[RAG] HF Status {res.status_code}"
    )

    if res.status_code != 200:
        raise Exception(
            f"HuggingFace failed : {res.text}"
        )

    data = res.json()

    if (
        isinstance(data, list)
        and len(data) > 0
        and isinstance(data[0], list)
    ):

        logger.info(
            "[RAG] ✅ Using HuggingFace Embedding"
        )

        return data[0]

    raise Exception("All embedding providers failed")

# ==========================================================
# Chunk Text
# ==========================================================

def chunk_text(text: str) -> list:
    """
    Split text into overlapping chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    logger.info(
        f"[RAG] Created {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    return chunks


# ==========================================================
# Ingest Document
# ==========================================================

def ingest_text(
    text: str,
    topic: str,
    source: str
):
    """
    Chunk → Embed → Store in Supabase
    """

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("Supabase not configured")

    text = re.sub(r"\s+", " ", text).strip()

    chunks = chunk_text(text)

    if len(chunks) == 0:
        raise Exception("No chunks generated")

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    stored = 0

    skipped = 0

    for i, chunk in enumerate(chunks, start=1):

        logger.info(
            f"[RAG] Processing chunk "
            f"{i}/{len(chunks)}"
        )

        # Skip tiny chunks
        if len(chunk.strip()) < 30:
            skipped += 1
            logger.info(
                f"[RAG] Skipping tiny chunk {i}"
            )
            continue

        try:

            vector = embed(chunk)

        except Exception as e:

            logger.error(
                f"[RAG] Embedding failed "
                f"for chunk {i}: {e}"
            )

            continue

        payload = {
            "topic": topic,
            "content": chunk,
            "embedding": vector,
            "source": source
        }

        try:

            res = request_with_retry(
                "POST",
                f"{SUPABASE_URL}/rest/v1/documents",
                headers=headers,
                json=payload
            )

            if res.status_code in (200, 201):

                stored += 1

                logger.info(
                    f"[RAG] ✅ Stored chunk "
                    f"{i}/{len(chunks)}"
                )

            else:

                logger.error(
                    f"[RAG] Insert failed "
                    f"{res.status_code}"
                )

                logger.error(res.text)

        except Exception as e:

            logger.error(
                f"[RAG] Database insert failed "
                f"for chunk {i}: {e}"
            )

    logger.info(
        f"""
=============================
RAG INGEST COMPLETE
Topic      : {topic}
Source     : {source}
Chunks     : {len(chunks)}
Stored     : {stored}
Skipped    : {skipped}
=============================
"""
    )

    return {
        "chunks_stored": stored,
        "total_chunks": len(chunks),
        "skipped": skipped
    }

    # ==========================================================
# Search Chunks
# ==========================================================

def search_chunks(
    query: str,
    topic: str = None,
    top_k: int = 8,
    threshold: float = 0.20
):
    """
    Semantic search using pgvector.

    Returns:
        [
            {
                "id": ...,
                "topic": ...,
                "content": ...,
                "source": ...,
                "similarity": ...
            }
        ]
    """

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("[RAG] Supabase not configured")
        return []

    logger.info("=" * 60)
    logger.info("[RAG] SEARCH STARTED")
    logger.info(f"[RAG] Query : {query}")
    logger.info(f"[RAG] Topic : {topic}")
    logger.info("=" * 60)

    # ------------------------------------------------------
    # Generate query embedding
    # ------------------------------------------------------

    try:

        query_vector = embed(query)

    except Exception as e:

        logger.error(f"[RAG] Query embedding failed : {e}")

        return []

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "query_embedding": query_vector,
        "match_threshold": threshold,
        "match_count": top_k
    }

    try:

        res = request_with_retry(
            "POST",
            f"{SUPABASE_URL}/rest/v1/rpc/match_documents",
            headers=headers,
            json=payload
        )

    except Exception as e:

        logger.error(f"[RAG] RPC failed : {e}")

        return []

    if res.status_code != 200:

        logger.error(
            f"[RAG] Search Error "
            f"{res.status_code}"
        )

        logger.error(res.text)

        return []

    results = res.json()

    logger.info(
        f"[RAG] RPC returned "
        f"{len(results)} chunks"
    )

    # ------------------------------------------------------
    # Optional Topic Filter
    # ------------------------------------------------------

    if topic:

        topic = topic.lower().strip()

        filtered = []

        for row in results:

            row_topic = (
                row.get("topic", "")
                .lower()
                .strip()
            )

            if row_topic == topic:

                filtered.append(row)

        logger.info(
            f"[RAG] Topic Filter : "
            f"{len(filtered)} chunks"
        )

        results = filtered

    # ------------------------------------------------------
    # Sort by similarity
    # ------------------------------------------------------

    results.sort(
        key=lambda x: x.get(
            "similarity",
            0
        ),
        reverse=True
    )

    # ------------------------------------------------------
    # Debug Output
    # ------------------------------------------------------

    logger.info("=" * 60)

    logger.info(
        f"[RAG] Returning "
        f"{len(results)} chunks"
    )

    for index, row in enumerate(results, start=1):

        logger.info(
            f"""
Result {index}

Topic      : {row.get("topic")}
Source     : {row.get("source")}
Similarity : {round(row.get("similarity",0),3)}

Preview:
{row.get("content","")[:150]}
"""
        )

    logger.info("=" * 60)

    return results

def document_exists(source: str):

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    res = requests.get(
        f"{SUPABASE_URL}/rest/v1/documents?source=eq.{source}&select=id&limit=1",
        headers=headers,
        timeout=10
    )

    if res.status_code != 200:
        return False

    return len(res.json()) > 0

