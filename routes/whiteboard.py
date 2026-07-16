import logging
import json
import re
import time as _time
from threading import Lock
from flask import Blueprint, request
from services.groq_service import ask_ai
from services.rag_service import search_chunks
from utils.limiter import limiter

whiteboard_bp = Blueprint("whiteboard", __name__)
logger = logging.getLogger("maitrilearn")

# ── LESSON CACHE ─────────────────────────────────────────────────────────────
# Whiteboard is by far the heaviest Groq consumer (1000+ tokens/call) and the
# most rate-limit-exposed endpoint — production logs showed an 82% fallback
# rate under repeated requests for the same handful of common topics (Docker,
# Python, SQL...). Caching successful (non-degraded) lessons by topic means a
# second student asking about "Docker" gets an instant, full-quality response
# instead of competing for the same Groq quota and likely degrading.
# NOTE: this is per-worker-process memory, not shared across gunicorn workers
# or restarts — a real production fix would move this to Redis/Supabase, but
# even per-process it meaningfully cuts duplicate Groq calls for popular topics.
_lesson_cache = {}
_lesson_cache_lock = Lock()
_CACHE_TTL_SECONDS = 30 * 60   # 30 minutes
_CACHE_MAX_ENTRIES = 200


def _cache_key(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.strip().lower())


def _cache_get(topic: str):
    key = _cache_key(topic)
    with _lesson_cache_lock:
        entry = _lesson_cache.get(key)
        if not entry:
            return None
        cached_at, payload = entry
        if _time.time() - cached_at > _CACHE_TTL_SECONDS:
            del _lesson_cache[key]
            return None
        return payload


def _cache_set(topic: str, payload: dict):
    key = _cache_key(topic)
    with _lesson_cache_lock:
        if len(_lesson_cache) >= _CACHE_MAX_ENTRIES and key not in _lesson_cache:
            oldest_key = min(_lesson_cache, key=lambda k: _lesson_cache[k][0])
            del _lesson_cache[oldest_key]
        _lesson_cache[key] = (_time.time(), payload)


def clean_json(raw: str) -> str:
    raw   = re.sub(r"```json|```", "", raw).strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1:
        return raw[start:end+1]
    return raw


def detect_subject_type(topic: str) -> str:
    t = topic.lower()
    if any(k in t for k in ["docker","kubernetes","k8s","linux","git","ci/cd",
                              "terraform","ansible","aws","bash","nginx","yaml","helm"]):
        return "devops"
    if any(k in t for k in ["reactor","nuclear","engine","turbine","circuit","machine",
                              "system","mechanism","power plant","factory","pipeline"]):
        return "engineering"
    if any(k in t for k in ["photosynthesis","biology","chemistry","physics","atom",
                              "molecule","cell","mitosis","reaction","force","gravity"]):
        return "science"
    if any(k in t for k in ["calculus","algebra","geometry","trigonometry","equation",
                              "theorem","derivative","integral","matrix","probability"]):
        return "math"
    if any(k in t for k in ["war","revolution","empire","history","ancient",
                              "medieval","independence","century","civilization"]):
        return "history"
    if any(k in t for k in ["python","javascript","java","algorithm","data structure",
                              "sorting","recursion","api","database","sql","neural"]):
        return "cs"
    return "general"


def build_prompt(topic: str, subject_type: str,
                 rag_chunks: list, rag_sources: list) -> str:

    subject_hints = {
        "devops":      "Include architecture diagrams, real CLI commands, Dockerfiles/YAMLs, deployment timelines.",
        "engineering": "Include a DETAILED multi-component architecture diagram showing the full system (5-7 components minimum), a sequential process flow showing exactly how energy/material moves through the system, and a chalkboard step listing critical safety/operating facts.",
        "science":     "Include diagrams showing structures, process flows, and real-world examples.",
        "math":        "Include step-by-step worked examples and formula explanations.",
        "history":     "Include timelines of key events, comparisons, and cause-effect flows.",
        "cs":          "Include code examples, algorithm flow diagrams, and complexity comparisons.",
        "general":     "Include diagrams, real-world analogies, step-by-step explanations.",
    }
    hint = subject_hints.get(subject_type, subject_hints["general"])

    # Complex topics need MORE detail and bigger diagrams
    complexity_note = ""
    if subject_type in ("engineering", "science"):
        complexity_note = """
COMPLEXITY REQUIREMENT: This is a complex technical/scientific topic.
- The architecture diagram MUST have 5-7 components minimum (not 3)
- Each component needs a clear, specific description (not generic "Part A")
- Connections between components must be labeled with WHAT flows between them
  (e.g. "steam", "electricity", "coolant", "neutrons" — not just "leads to")
- Include a "chalkboard" step type with 4-6 critical facts the student must memorize
"""

    if rag_chunks:
        combined = "\n\n---\n\n".join(rag_chunks)
        sources  = ", ".join(set(rag_sources)) if rag_sources else "uploaded notes"
        context_section = f"""
STUDENT'S UPLOADED NOTES (use these as your PRIMARY source):
Source files: {sources}
---
{combined[:4000]}
---

IMPORTANT INSTRUCTIONS FOR RAG MODE:
- Base your lesson DIRECTLY on the content above
- Preserve key terms, definitions and examples from the notes
- You MAY expand with additional real-world examples to make it clearer
- Do NOT contradict the notes
- Cite which part of the notes you're teaching from in your narration
"""
    else:
        context_section = """
(No uploaded notes found for this topic — teaching from AI knowledge)

INSTRUCTIONS FOR FALLBACK MODE:
- Use your best, most ACCURATE and DETAILED knowledge to teach this topic
- Be specific — use real component names, real numbers, real processes
- Suggest the student uploads relevant notes for more personalized teaching
"""

    return f"""You are an expert classroom teacher and technical illustrator for ALL subjects — including complex engineering and scientific topics like nuclear reactors, engines, biological systems.
Create a RICH, DETAILED, visual step-by-step lesson for: "{topic}"

{context_section}

SUBJECT GUIDANCE: {hint}
{complexity_note}

TEACHING RULES:
- Each step teaches ONE concept only
- Explanations should be 1-3 sentences — detailed enough to be useful, not a wall of text
- Always use real, SPECIFIC examples (real component names, real numbers, real terms)
- Narration must sound like a warm, encouraging teacher
- If RAG notes provided, reference them naturally in narration
- NEVER use generic placeholders like "Part A", "Component B" — use REAL names
  (e.g. for nuclear reactor: "Fuel Rods", "Control Rods", "Coolant Loop", "Steam Generator", "Turbine", "Generator", "Containment Vessel")

Return ONLY valid JSON — no markdown, no backticks, no text outside JSON.

JSON SAFETY RULES (critical — output must be parseable):
- Any code example must be SHORT: 12 lines maximum. Summarize longer algorithms with comments instead of full implementations.
- Inside JSON string values, all newlines MUST be escaped as \n and all double quotes MUST be escaped as \".
- Do not use raw/literal line breaks inside a string value — every line break inside a "code" or "narration" field must be the two characters backslash-n.
- Keep every individual string field under 500 characters.

Available step types:
- title: lesson opener
- concept: definition + analogy
- steps: numbered process list
- architecture: detailed component diagram with labeled connections (5-7 nodes for complex topics)
- flowdiagram: sequential process flow (use for energy/material flow through a system)
- chalkboard: IMPORTANT — critical facts written like chalk notes, used for safety facts, key numbers, must-remember points
- example: real example with optional code/formula
- codefile: full code file (CS/DevOps only)
- timeline: stages with durations
- comparison: before vs after / type A vs type B
- summary: a short recap of everything covered in the lesson so far, in 3-5 plain sentences/bullets
- remember: 2-4 concise must-remember facts distilled from the whole lesson (like exam-cram notes)
- keypoints: final warm summary with icons and encouragement

{{
  "title": "lesson title",
  "subject": "subject area",
  "rag_mode": {"true" if rag_chunks else "false"},
  "steps": [
    {{
      "type": "title",
      "text": "{topic}",
      "subtitle": "one sentence hook — why this matters",
      "narration": "warm intro (2 sentences)"
    }},
    {{
      "type": "concept",
      "heading": "What is {topic}?",
      "definition": "clear, specific, accurate definition",
      "analogy": "real-world analogy that makes it click",
      "narration": "teacher explains using the analogy"
    }},
    {{
      "type": "architecture",
      "heading": "How {topic} is Built — Full Architecture",
      "nodes": [
        {{"id": "n1", "label": "REAL component name", "description": "specific what it does", "role": "input"}},
        {{"id": "n2", "label": "REAL component name", "description": "specific what it does", "role": "process"}},
        {{"id": "n3", "label": "REAL component name", "description": "specific what it does", "role": "process"}},
        {{"id": "n4", "label": "REAL component name", "description": "specific what it does", "role": "process"}},
        {{"id": "n5", "label": "REAL component name", "description": "specific what it does", "role": "output"}}
      ],
      "connections": [
        {{"from": "n1", "to": "n2", "label": "what flows here e.g. heat/steam/data"}},
        {{"from": "n2", "to": "n3", "label": "what flows here"}},
        {{"from": "n3", "to": "n4", "label": "what flows here"}},
        {{"from": "n4", "to": "n5", "label": "what flows here"}}
      ],
      "narration": "teacher walks through the FULL system component by component"
    }},
    {{
      "type": "flowdiagram",
      "heading": "Step-by-Step Process",
      "nodes": [
        {{"label": "Specific stage name", "sublabel": "what exactly happens"}},
        {{"label": "Specific stage name", "sublabel": "what exactly happens"}},
        {{"label": "Specific stage name", "sublabel": "what exactly happens"}},
        {{"label": "Specific stage name", "sublabel": "what exactly happens"}}
      ],
      "narration": "teacher walks through each stage of the process in order"
    }},
    {{
      "type": "chalkboard",
      "heading": "Important Points to Remember",
      "points": [
        "Critical fact 1 — specific and exam-worthy",
        "Critical fact 2 — specific number or term",
        "Critical fact 3 — common misconception corrected",
        "Critical fact 4 — safety or key operating principle"
      ],
      "narration": "teacher emphasizes these are the points to remember most"
    }},
    {{
      "type": "comparison",
      "heading": "Types / Variants Comparison",
      "left_label": "Type A or Without",
      "left_points": ["specific point 1", "specific point 2", "specific point 3"],
      "right_label": "Type B or With",
      "right_points": ["specific point 1", "specific point 2", "specific point 3"],
      "narration": "teacher explains the key differences"
    }},
    {{
      "type": "summary",
      "heading": "Let's Recap",
      "points": [
        "Plain-language recap sentence 1 covering an earlier concept step",
        "Plain-language recap sentence 2 covering another earlier step",
        "Plain-language recap sentence 3 tying it all together"
      ],
      "narration": "teacher briefly recaps everything covered so far"
    }},
    {{
      "type": "remember",
      "heading": "Remember This",
      "points": [
        "Concise must-remember fact 1 — exam-worthy, specific",
        "Concise must-remember fact 2 — specific number or term",
        "Concise must-remember fact 3 — common mistake to avoid"
      ],
      "narration": "teacher stresses these are the facts to memorize"
    }},
    {{
      "type": "keypoints",
      "heading": "You've Got This!",
      "points": [
        {{"icon": "⚡", "text": "most important single fact about this topic"}},
        {{"icon": "🎯", "text": "key practical application or insight"}},
        {{"icon": "💡", "text": "common mistake or exam tip"}}
      ],
      "narration": "teacher summarizes warmly and encourages the student"
    }}
  ]
}}

Generate 8-10 steps. MUST start with title.
MUST include a 'summary' step that recaps the lesson in plain language.
MUST include a 'remember' step with 2-4 concise must-remember facts.
MUST end with keypoints.
MUST include architecture with 5+ REAL named components for engineering/science/devops topics.
MUST include chalkboard step for important facts.
Topic: {topic}
"""


def build_fallback_lesson(topic: str, ai_explanation: str = None) -> dict:
    """
    Used when the AI fails to produce parseable JSON after both attempts.
    If ai_explanation is provided (from the plain-text rescue attempt in
    whiteboard_lesson), the student still gets real teaching content in a
    minimal 3-step shape — not a "please try again" apology. Only falls back
    to the apology message if EVERY generation attempt failed, including
    the plain-text one.
    """
    if ai_explanation:
        return {
            "title": topic,
            "subject": "general",
            "rag_mode": False,
            "steps": [
                {
                    "type": "title",
                    "text": topic,
                    "subtitle": "Let's explore this topic together",
                    "narration": f"Today we're learning about {topic}. Let's dive in!"
                },
                {
                    "type": "concept",
                    "heading": f"Understanding {topic}",
                    "definition": ai_explanation,
                    "analogy": "",
                    "narration": "Here's the key idea — the full illustrated lesson "
                                 "with diagrams is temporarily unavailable, but here's "
                                 "what you need to know."
                },
                {
                    "type": "keypoints",
                    "heading": "Want the Full Lesson?",
                    "points": [
                        {"icon": "🔄", "text": "Try again in a moment for the full illustrated lesson"},
                        {"icon": "📝", "text": "Uploading notes on this topic also helps"}
                    ],
                    "narration": "That's the essential idea — try again shortly for the full visual walkthrough!"
                }
            ]
        }

    return {
        "title": topic,
        "subject": "general",
        "rag_mode": False,
        "steps": [
            {
                "type": "title",
                "text": topic,
                "subtitle": "Let's explore this topic together",
                "narration": f"Today we're learning about {topic}. Let's dive in!"
            },
            {
                "type": "concept",
                "heading": f"What is {topic}?",
                "definition": (
                    f"We had trouble generating the full visual lesson for '{topic}' just now. "
                    "This can happen briefly during high load — please try again in a moment "
                    "for the complete step-by-step lesson with diagrams and examples."
                ),
                "analogy": "",
                "narration": "Please try regenerating this lesson for the full experience."
            },
            {
                "type": "keypoints",
                "heading": "Try Again",
                "points": [
                    {"icon": "🔄", "text": "Click 'Start Lesson' again to retry generating the full lesson"},
                    {"icon": "📝", "text": "Try uploading notes on this topic for a more grounded lesson"}
                ],
                "narration": "Sorry about that — one more try should do it!"
            }
        ]
    }


@whiteboard_bp.route("/whiteboard/lesson", methods=["POST"])
@limiter.limit("8 per minute")
def whiteboard_lesson():
    data = request.get_json(silent=True) or {}
    if not data.get("topic"):
        return {"error": "topic is required"}, 400

    topic = data["topic"].strip()

    cached = _cache_get(topic)
    if cached:
        logger.info(f"[whiteboard] Cache hit for '{topic}'")
        return {**cached, "cached": True}

    rag_chunks  = []
    rag_sources = []
    chunks_found = 0

    try:
        # SPEED FIX: single RAG call with broad search (no topic filter first)
        # topic filter was causing double DB roundtrip — broad search is fast enough
        results = search_chunks(topic, topic=None, top_k=8, threshold=0.35)
        if results:
            # Handle both dict results {content, source} and plain string results
            if results and isinstance(results[0], dict):
                rag_chunks  = [r.get("content", "") for r in results if r.get("content")]
                rag_sources = [r.get("source", "notes") for r in results]
            else:
                rag_chunks  = [r for r in results if isinstance(r, str) and r.strip()]
                rag_sources = ["notes"] * len(rag_chunks)
            chunks_found = len(rag_chunks)
            if chunks_found > 0:
                logger.info(f"[whiteboard] RAG found {chunks_found} chunks for '{topic}'")
            else:
                logger.info(f"[whiteboard] RAG results empty after parsing for '{topic}'")
        else:
            logger.info(f"[whiteboard] No RAG chunks for '{topic}' — using AI knowledge")
    except Exception as e:
        logger.warning(f"[whiteboard] RAG error (non-fatal): {e}")

    subject_type = detect_subject_type(topic)
    prompt       = build_prompt(topic, subject_type, rag_chunks, rag_sources)

    raw = ""
    lesson = None
    try:
        # json_mode=True forces the Groq API's structured JSON output mode,
        # which guarantees syntactically valid JSON (handles escaping of
        # newlines/quotes inside code examples automatically). This was the
        # main cause of 500s on code-heavy topics like "Sorting Algorithms",
        # where free-text generation produced raw newlines inside JSON
        # string values.
        raw   = ask_ai(prompt, json_mode=True, route='whiteboard')
        clean = clean_json(raw)

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning(f"[whiteboard] First JSON parse failed for '{topic}' — retrying")
            fix_prompt = (
                "The following is almost valid JSON but has syntax errors "
                "(likely an unescaped newline or quote inside a string value). "
                "Fix ONLY the JSON syntax and return the corrected, complete JSON object. "
                "Do not change any content, just fix syntax:\n\n" + clean[:8000]
            )
            try:
                raw2   = ask_ai(fix_prompt, json_mode=True, route='whiteboard')
                clean2 = clean_json(raw2)
                parsed = json.loads(clean2)
            except Exception as retry_err:
                logger.error(f"[whiteboard] JSON repair also failed for '{topic}': {retry_err}")
                parsed = None

        if parsed:
            candidate = parsed.get("lesson", parsed)
            if candidate.get("steps"):
                lesson = candidate

    except Exception as e:
        logger.error(f"[whiteboard] Generation error for '{topic}': {e}")

    # ── Graceful degradation: never hard-fail the student with a raw 500 ──────
    degraded = False
    if not lesson:
        # Before falling all the way back to a canned apology, try ONE plain
        # text generation. Structured JSON output is the hardest thing to
        # get right from a strained/fallback provider (schema compliance +
        # correct escaping); plain prose has neither constraint, so it's
        # meaningfully more likely to succeed even when JSON mode just
        # failed twice. This means a "degraded" response usually still
        # teaches the student something real instead of just apologizing.
        ai_explanation = None
        try:
            simple_prompt = (
                f"Explain '{topic}' to a student in 3-4 clear, informative "
                f"sentences. Plain text only, no markdown, no JSON, no headers."
            )
            text = ask_ai(simple_prompt, json_mode=False, route='whiteboard')
            if text and text.strip():
                ai_explanation = text.strip()[:800]
        except Exception as rescue_err:
            logger.warning(f"[whiteboard] Plain-text rescue also failed for '{topic}': {rescue_err}")

        logger.warning(f"[whiteboard] Falling back to minimal lesson for '{topic}' "
                        f"(rescue_content={'yes' if ai_explanation else 'no'})")
        lesson   = build_fallback_lesson(topic, ai_explanation)
        degraded = True

    unique_sources = list(set(rag_sources)) if rag_sources else []
    citation = None
    if unique_sources:
        clean_sources = []
        for s in unique_sources:
            parts = s.split("_", 1)
            name  = parts[1] if len(parts) > 1 and parts[0].isdigit() else s
            clean_sources.append(name)
        citation = {
            "sources":     clean_sources,
            "raw_sources": unique_sources,
            "chunks_used": chunks_found,
            "message":     f"This lesson was built from your uploaded notes: {', '.join(clean_sources)}"
        }

    logger.info(f"[whiteboard] Lesson generated: topic={topic} subject_type={subject_type} "
               f"rag={bool(rag_chunks)} chunks={chunks_found} steps={len(lesson['steps'])} "
               f"degraded={degraded}")

    result = {
        "lesson":       lesson,
        "rag_used":     bool(rag_chunks),
        "chunks":       chunks_found,
        "citation":     citation,
        "subject_type": subject_type,
        "degraded":     degraded
    }

    # Only cache full-quality lessons — never cache a degraded fallback,
    # or every subsequent request for that topic would get stuck serving
    # the minimal 3-step version until the TTL expires.
    if not degraded:
        _cache_set(topic, result)

    return result
