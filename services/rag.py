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


@rag_bp.route("/rag/ingest", methods=["POST"])
def ingest():
    data = request.json
    if not data:
        return {"error": "Request body required"}, 400

    filename = data.get("filename", "").strip()
    topic    = data.get("topic", "").strip()
    bucket   = data.get("bucket", "notes").strip()
    # Max pages to process — prevents timeout on large PDFs
    max_pages = int(data.get("max_pages", 30))

    if not filename or not topic:
        return {"error": "filename and topic are required"}, 400

    # Download file from Supabase Storage
    file_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
    try:
        file_res = requests.get(file_url, timeout=30)
        if file_res.status_code != 200:
            return {"error": f"Could not download file: {file_res.status_code}. Make sure bucket is PUBLIC."}, 404
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}, 500

    # Extract text
    fname_lower = filename.lower()
    try:
        if fname_lower.endswith(".pdf"):
            text, pages_processed = extract_pdf_safe(file_res.content, max_pages)
        elif fname_lower.endswith((".txt", ".md")):
            text = file_res.content.decode("utf-8", errors="ignore")
            pages_processed = 1
        else:
            return {"error": "Unsupported file type. Use PDF, TXT, or MD."}, 400
    except Exception as e:
        return {"error": f"Text extraction failed: {str(e)}"}, 500

    if not text or len(text.strip()) < 50:
        return {"error": "File appears empty or unreadable. Try a text file instead."}, 400

    # Chunk, embed, store
    try:
        result = ingest_text(text, topic, source=filename)
        return {
            "success":       True,
            "filename":      filename,
            "topic":         topic,
            "pages_processed": pages_processed,
            "chunks_stored": result["chunks_stored"],
            "total_chunks":  result["total_chunks"],
            "message":       f"Ingested {result['chunks_stored']} chunks from {pages_processed} pages of {filename}"
        }
    except Exception as e:
        print(f"[rag/ingest] Error: {e}")
        return {"error": f"Ingestion failed: {str(e)}"}, 500


def extract_pdf_safe(content: bytes, max_pages: int = 30) -> tuple:
    """
    Extract text from PDF page by page.
    Stops at max_pages to prevent timeout on large PDFs.
    Returns (text, pages_processed).
    """
    try:
        import pypdf
    except ImportError:
        raise ValueError("pypdf not installed")

    reader = pypdf.PdfReader(io.BytesIO(content))
    total  = len(reader.pages)
    limit  = min(total, max_pages)
    pages  = []

    for i in range(limit):
        try:
            # Extract one page at a time — avoids memory spike
            t = reader.pages[i].extract_text()
            if t and t.strip():
                pages.append(t.strip())
        except Exception as e:
            # Skip unreadable pages — don't crash
            print(f"[rag] Skipping page {i+1}: {e}")
            continue

    if not pages:
        raise ValueError(
            f"Could not extract text from PDF ({total} pages). "
            "PDF may be scanned/image-based. Try uploading a text file instead."
        )

    text = "\n\n".join(pages)
    return text, limit


@rag_bp.route("/rag/search", methods=["POST"])
def search():
    data = request.json
    if not data or not data.get("query"):
        return {"error": "query is required"}, 400

    query = data["query"]
    topic = data.get("topic", "")
    top_k = int(data.get("top_k", 5))

    try:
        chunks = search_chunks(query, topic=topic, top_k=top_k)
        return {"chunks": chunks, "count": len(chunks)}
    except Exception as e:
        print(f"[rag/search] Error: {e}")
        return {"error": str(e)}, 500


@rag_bp.route("/rag/topics", methods=["GET"])
def list_topics():
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


@rag_bp.route("/rag/delete", methods=["DELETE"])
def delete_topic():
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
