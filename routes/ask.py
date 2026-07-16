import logging
from flask import Blueprint, request
from services.groq_service import ask_ai
from services.rag_service import search_chunks
from utils.validator import validate_question, validate_text, ValidationError
from utils.limiter import limiter

ask_bp = Blueprint("ask", __name__)
logger = logging.getLogger("maitrilearn")


def _parse_chunks(results):
    """Normalize search_chunks() output into (chunks, sources) lists."""
    if results and isinstance(results[0], dict):
        chunks  = [r.get("content", "") for r in results if r.get("content")]
        sources = [r.get("source", "notes") for r in results if r.get("content")]
    else:
        chunks  = [r for r in results if isinstance(r, str) and r.strip()]
        sources = ["notes"] * len(chunks)
    return chunks, sources


def _clean_sources(raw_sources: list) -> list:
    """Strip numeric upload-id prefixes like '1234567890_docker-notes.pdf' -> 'docker-notes.pdf'."""
    cleaned = []
    for s in set(raw_sources):
        parts = s.split("_", 1)
        cleaned.append(parts[1] if len(parts) > 1 and parts[0].isdigit() else s)
    return cleaned


# QA audit P0: /ask was inheriting the blueprint-wide default limit
# (200/day, 50/hour, 20/minute — utils/limiter.py) with no route-specific
# override. A stress test showed 10 concurrent requests all hit 429 within
# 500ms, and a classroom of ~30 students would exhaust the 50/hour bucket
# in minutes. override_defaults=True replaces (not stacks on top of) the
# blueprint default for this route specifically.
@ask_bp.route("/ask", methods=["POST"])
@limiter.limit("30 per minute;400 per hour", override_defaults=True)
def ask():
    data = request.get_json(silent=True) or {}

    try:
        question = validate_question(data.get("question", ""))
        subject  = validate_text(data.get("subject", "General"),
                                 field="subject", min_len=0, max_len=100) \
                   if data.get("subject") else "General"
        topic    = validate_text(data.get("topic", ""),
                                 field="topic", min_len=0, max_len=200) \
                   if data.get("topic") else ""
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    # ── RAG retrieval: search the student's uploaded notes first ──────────────
    rag_chunks  = []
    rag_sources = []
    chunks_found = 0
    try:
        results = search_chunks(question, topic=topic or None, top_k=5, threshold=0.35)
        if results:
            rag_chunks, rag_sources = _parse_chunks(results)
            chunks_found = len(rag_chunks)
            logger.info(f"[ask] RAG found {chunks_found} chunks for question")
        else:
            logger.info("[ask] No RAG chunks found — falling back to AI knowledge")
    except Exception as e:
        # Non-fatal: fall back to plain LLM answer if retrieval fails
        logger.warning(f"[ask] RAG error (non-fatal): {e}")

    # ── Build prompt, grounded in notes when available ─────────────────────────
    header = ""
    if subject and subject != "General" or topic:
        header = f"Subject: {subject}\nTopic: {topic}\n"

    if rag_chunks:
        context = "\n\n---\n\n".join(rag_chunks)[:4000]
        prompt = (
            f"{header}"
            f"STUDENT'S UPLOADED NOTES (use these as your PRIMARY source):\n"
            f"---\n{context}\n---\n\n"
            f"Q: {question}\n\n"
            "Answer clearly in 2-4 sentences, grounded in the notes above. "
            "If the notes don't fully cover the question, say so and add relevant general knowledge."
        )
    else:
        prompt = f"{header}Q: {question}\n\nAnswer clearly in 2-4 sentences."

    try:
        answer = ask_ai(prompt, route="ask")
        logger.info(f"[ask] subject={subject} rag={bool(rag_chunks)} "
                    f"chunks={chunks_found} len={len(answer)}")

        clean_sources = _clean_sources(rag_sources) if rag_sources else []
        return {
            "answer":     answer,
            "rag_used":   bool(rag_chunks),
            "chunks":     rag_chunks,
            "sources":    clean_sources,
            "no_match":   chunks_found == 0,
        }
    except Exception as e:
        logger.error(f"[ask] Error: {e}")
        error_msg = str(e)
        if "Rate limit" in error_msg or "429" in error_msg:
            return {"error": "Too many requests — please wait 10 seconds and try again."}, 429
        return {"error": "AI service unavailable. Please try again in a moment."}, 503
