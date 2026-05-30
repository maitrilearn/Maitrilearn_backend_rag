import os
import io
import requests
from flask import Blueprint, request
from services.rag_service import ingest_text, search_chunks
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

rag_bp = Blueprint("rag", __name__)


# ── INGEST FROM SUPABASE BUCKET ───────────────────────────────────────────────
@rag_bp.route("/rag/ingest", methods=["POST"])
def ingest():
    """
    Admin endpoint.
    Body: { "filename": "docker-notes.pdf", "topic": "Docker", "bucket": "notes" }
    Downloads file from Supabase bucket, extracts text, chunks, embeds, stores.
    """
    data = request.json
    if not data:
        return {"error": "Request body required"}, 400

    filename = data.get("filename", "").strip()
    topic    = data.get("topic", "").strip()
    bucket   = data.get("bucket", "notes").strip()

    if not filename or not topic:
        return {"error": "filename and topic are required"}, 400

    # Download file from Supabase Storage
    file_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
    try:
        file_res = requests.get(file_url, timeout=30)
        if file_res.status_code != 200:
            return {"error": f"Could not download file from bucket: {file_res.status_code}"}, 404
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}, 500

    # Extract text based on file type
    fname_lower = filename.lower()
    try:
        if fname_lower.endswith(".pdf"):
            text = extract_pdf(file_res.content)
        elif fname_lower.endswith((".txt", ".md")):
            text = file_res.content.decode("utf-8", errors="ignore")
        else:
            return {"error": "Unsupported file type. Use PDF, TXT, or MD."}, 400
    except Exception as e:
        return {"error": f"Text extraction failed: {str(e)}"}, 500

    if not text or len(text.strip()) < 50:
        return {"error": "File appears to be empty or unreadable"}, 400

    # Chunk, embed, store
    try:
        result = ingest_text(text, topic, source=filename)
        return {
            "success": True,
            "filename": filename,
            "topic": topic,
            "chunks_stored": result["chunks_stored"],
            "total_chunks": result["total_chunks"],
            "message": f"Successfully ingested {result['chunks_stored']} chunks from {filename}"
        }
    except Exception as e:
        print(f"[rag/ingest] Error: {e}")
        return {"error": f"Ingestion failed: {str(e)}"}, 500


def extract_pdf(content: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages  = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except ImportError:
        raise ValueError("pypdf not installed — add it to requirements.txt")
    except Exception as e:
        raise ValueError(f"PDF parse error: {e}")


# ── SEARCH (used internally by whiteboard) ─────────────────────────────────────
@rag_bp.route("/rag/search", methods=["POST"])
def search():
    """
    Internal + debug endpoint.
    Body: { "query": "how does docker work", "topic": "Docker", "top_k": 5 }
    Returns relevant chunks.
    """
    data = request.json
    if not data or not data.get("query"):
        return {"error": "query is required"}, 400

    query   = data["query"]
    topic   = data.get("topic", "")
    top_k   = int(data.get("top_k", 5))

    try:
        chunks = search_chunks(query, topic=topic, top_k=top_k)
        return {"chunks": chunks, "count": len(chunks)}
    except Exception as e:
        print(f"[rag/search] Error: {e}")
        return {"error": str(e)}, 500


# ── LIST INGESTED TOPICS ───────────────────────────────────────────────────────
@rag_bp.route("/rag/topics", methods=["GET"])
def list_topics():
    """Returns distinct topics in the knowledge base."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "Supabase not configured"}, 500

    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    res = requests.get(
        f"{SUPABASE_URL}/rest/v1/documents?select=topic&order=topic",
        headers=headers, timeout=10
    )
    if res.status_code != 200:
        return {"error": "Could not fetch topics"}, 500

    rows   = res.json()
    topics = sorted(set(r["topic"] for r in rows if r.get("topic")))
    return {"topics": topics, "total_chunks": len(rows)}


# ── DELETE TOPIC ───────────────────────────────────────────────────────────────
@rag_bp.route("/rag/delete", methods=["DELETE"])
def delete_topic():
    """Delete all chunks for a topic. Body: { "topic": "Docker" }"""
    data = request.json
    if not data or not data.get("topic"):
        return {"error": "topic is required"}, 400

    topic   = data["topic"]
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    res = requests.delete(
        f"{SUPABASE_URL}/rest/v1/documents?topic=eq.{topic}",
        headers=headers, timeout=10
    )
    if res.status_code in (200, 204):
        return {"success": True, "deleted_topic": topic}
    return {"error": f"Delete failed: {res.status_code}"}, 500
