import os
import io
import logging
import requests
from flask import Blueprint, request
from services.rag_service import ingest_text, search_chunks
from utils.validator import validate_filename, validate_topic, validate_text, ValidationError
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

rag_bp = Blueprint("rag", __name__)
logger = logging.getLogger("maitrilearn")


@rag_bp.route("/rag/ingest", methods=["POST"])
def ingest():
    data = request.get_json(silent=True) or {}

    try:
        filename = validate_filename(data.get("filename", ""))
        topic    = validate_topic(data.get("topic", ""))
        bucket   = validate_text(data.get("bucket", "notes"),
                                 field="bucket", min_len=1, max_len=50)
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    max_pages = min(int(data.get("max_pages", 30)), 50)  # cap at 50

    file_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
    try:
        file_res = requests.get(file_url, timeout=30)
        if file_res.status_code != 200:
            return {"error": f"File not found in bucket '{bucket}'. Make sure bucket is PUBLIC and filename is correct."}, 404
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}, 500

    fname_lower = filename.lower()
    try:
        if fname_lower.endswith(".pdf"):
            text, pages = extract_pdf_safe(file_res.content, max_pages)
        else:
            text  = file_res.content.decode("utf-8", errors="ignore")
            pages = 1
    except Exception as e:
        return {"error": f"Text extraction failed: {str(e)}"}, 500

    if not text or len(text.strip()) < 50:
        return {"error": "File appears empty or unreadable. Try a text file instead."}, 400

    try:
        result = ingest_text(text, topic, source=filename)
        logger.info(f"[rag/ingest] topic={topic} file={filename} chunks={result['chunks_stored']}")
        return {
            "success":       True,
            "filename":      filename,
            "topic":         topic,
            "pages":         pages,
            "chunks_stored": result["chunks_stored"],
            "total_chunks":  result["total_chunks"],
            "message":       f"Ingested {result['chunks_stored']} chunks from {filename}"
        }
    except Exception as e:
        logger.error(f"[rag/ingest] Error: {e}")
        return {"error": f"Ingestion failed: {str(e)}"}, 500


def extract_pdf_safe(content: bytes, max_pages: int = 30):
    try:
        import pypdf
    except ImportError:
        raise ValueError("pypdf not installed")
    reader = pypdf.PdfReader(io.BytesIO(content))
    limit  = min(len(reader.pages), max_pages)
    pages  = []
    for i in range(limit):
        try:
            t = reader.pages[i].extract_text()
            if t and t.strip():
                pages.append(t.strip())
        except Exception as e:
            logger.warning(f"[rag] Skipping page {i+1}: {e}")
    if not pages:
        raise ValueError("Could not extract text. PDF may be image-based — try a text file.")
    return "\n\n".join(pages), limit


@rag_bp.route("/rag/search", methods=["POST"])
def search():
    data = request.get_json(silent=True) or {}

    try:
        query = validate_text(data.get("query", ""), field="query", min_len=2, max_len=500)
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    topic = data.get("topic", "")
    top_k = min(int(data.get("top_k", 5)), 10)

    try:
        raw_results = search_chunks(query, topic=topic or None, top_k=top_k)
        # Normalize: handle both string list and dict list
        if raw_results and isinstance(raw_results[0], dict):
            contents = [r.get("content","") for r in raw_results if r.get("content")]
        else:
            contents = [r for r in raw_results if isinstance(r, str) and r.strip()]
        return {
            "chunks":   contents,
            "count":    len(contents),
            "no_match": len(contents) == 0,
            "message":  "No matching notes found" if len(contents) == 0 else f"Found {len(contents)} relevant chunks"
        }
    except Exception as e:
        logger.error(f"[rag/search] Error: {e}")
        return {"error": str(e)}, 500


@rag_bp.route("/rag/topics", methods=["GET"])
def list_topics():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "Supabase not configured"}, 500
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
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
    data = request.get_json(silent=True) or {}
    try:
        topic = validate_topic(data.get("topic", ""))
    except ValidationError as e:
        return {"error": e.message}, 400
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    res = requests.delete(
        f"{SUPABASE_URL}/rest/v1/documents?topic=eq.{topic}",
        headers=headers, timeout=10
    )
    if res.status_code in (200, 204):
        logger.info(f"[rag/delete] Deleted topic={topic}")
        return {"success": True, "deleted_topic": topic}
    return {"error": f"Delete failed: {res.status_code}"}, 500
