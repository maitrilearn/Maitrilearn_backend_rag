import logging
import json
import re
from flask import Blueprint, request
from services.groq_service import ask_ai
from services.rag_service import search_chunks

whiteboard_bp = Blueprint("whiteboard", __name__)
logger = logging.getLogger("maitrilearn")


def clean_json(raw: str) -> str:
    """Strip markdown fences and extract first JSON object."""
    raw   = re.sub(r"```json|```", "", raw).strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1:
        return raw[start:end+1]
    return raw


def detect_subject_type(topic: str) -> str:
    """Detect subject category for subject-aware teaching."""
    t = topic.lower()
    if any(k in t for k in ["docker","kubernetes","k8s","linux","git","ci/cd",
                              "terraform","ansible","aws","bash","nginx","yaml","helm"]):
        return "devops"
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
    """
    Build subject-aware prompt.
    If RAG chunks exist → AI uses them as base and expands.
    If no chunks → AI uses own knowledge only.
    """

    subject_hints = {
        "devops":  "Include architecture diagrams, real CLI commands, Dockerfiles/YAMLs, deployment timelines.",
        "science": "Include diagrams showing structures, process flows, and real-world examples.",
        "math":    "Include step-by-step worked examples and formula explanations.",
        "history": "Include timelines of key events, comparisons, and cause-effect flows.",
        "cs":      "Include code examples, algorithm flow diagrams, and complexity comparisons.",
        "general": "Include diagrams, real-world analogies, step-by-step explanations.",
    }
    hint = subject_hints.get(subject_type, subject_hints["general"])

    if rag_chunks:
        # ── RAG MODE — use notes as base, expand with examples ──────────────
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
  e.g. "According to your notes...", "Your notes explain that...", "As mentioned in your material..."
"""
    else:
        # ── FALLBACK MODE — pure AI knowledge ───────────────────────────────
        context_section = """
(No uploaded notes found for this topic — teaching from AI knowledge)

INSTRUCTIONS FOR FALLBACK MODE:
- Use your best knowledge to teach this topic accurately
- Suggest the student uploads relevant notes for more personalized teaching
"""

    return f"""You are an expert classroom teacher for ALL subjects.
Create a rich visual step-by-step lesson for: "{topic}"

{context_section}

SUBJECT GUIDANCE: {hint}

TEACHING RULES:
- Each step teaches ONE concept only
- SHORT explanations (1-2 sentences max)
- Always use real examples relevant to the topic
- Narration must sound like a warm, encouraging teacher
- If RAG notes provided, reference them naturally in narration

Return ONLY valid JSON — no markdown, no backticks, no text outside JSON.

{{
  "title": "lesson title",
  "subject": "subject area",
  "rag_mode": {"true" if rag_chunks else "false"},
  "steps": [
    {{
      "type": "title",
      "text": "{topic}",
      "subtitle": "one sentence why this matters",
      "narration": "warm intro — if RAG: mention you're teaching from their notes"
    }},
    {{
      "type": "concept",
      "heading": "What is {topic}?",
      "definition": "clear definition — use notes if available",
      "analogy": "real-world analogy",
      "narration": "teacher explains — reference notes if available"
    }},
    {{
      "type": "flowdiagram",
      "heading": "How it works",
      "nodes": [
        {{"label": "Stage 1", "sublabel": "detail"}},
        {{"label": "Stage 2", "sublabel": "detail"}},
        {{"label": "Stage 3", "sublabel": "detail"}},
        {{"label": "Stage 4", "sublabel": "detail"}}
      ],
      "narration": "teacher walks through each stage"
    }},
    {{
      "type": "architecture",
      "heading": "Structure / Architecture",
      "nodes": [
        {{"id": "a", "label": "Part A", "description": "what it does", "role": "input"}},
        {{"id": "b", "label": "Part B", "description": "what it does", "role": "process"}},
        {{"id": "c", "label": "Part C", "description": "what it does", "role": "output"}}
      ],
      "connections": [
        {{"from": "a", "to": "b", "label": "leads to"}},
        {{"from": "b", "to": "c", "label": "produces"}}
      ],
      "narration": "teacher explains each component"
    }},
    {{
      "type": "example",
      "heading": "Real Example",
      "scenario": "concrete scenario from notes or real world",
      "code": "command or formula if applicable else empty string",
      "lang": "bash or python or text",
      "explanation": "what this example shows",
      "narration": "teacher explains — reference notes if used"
    }},
    {{
      "type": "timeline",
      "heading": "Key Stages",
      "nodes": [
        {{"label": "Stage 1", "sublabel": "what happens", "duration": "time/phase"}},
        {{"label": "Stage 2", "sublabel": "what happens", "duration": "time/phase"}},
        {{"label": "Stage 3", "sublabel": "what happens", "duration": "time/phase"}},
        {{"label": "Stage 4", "sublabel": "what happens", "duration": "time/phase"}}
      ],
      "narration": "teacher explains progression"
    }},
    {{
      "type": "comparison",
      "heading": "Key Contrast",
      "left_label": "Without / Before",
      "left_points": ["point 1", "point 2", "point 3"],
      "right_label": "With / After",
      "right_points": ["point 1", "point 2", "point 3"],
      "narration": "teacher explains the contrast"
    }},
    {{
      "type": "keypoints",
      "heading": "Remember This",
      "points": [
        {{"icon": "⚡", "text": "most important thing — from notes if available"}},
        {{"icon": "🎯", "text": "key insight or application"}},
        {{"icon": "💡", "text": "common mistake or exam tip"}}
      ],
      "narration": "teacher wraps up — if RAG: encourage student to review their notes"
    }}
  ]
}}

Generate 6-8 steps. MUST start with title. MUST end with keypoints.
Topic: {topic}
"""


@whiteboard_bp.route("/whiteboard/lesson", methods=["POST"])
def whiteboard_lesson():
    data = request.get_json(silent=True) or {}
    if not data.get("topic"):
        return {"error": "topic is required"}, 400

    topic = data["topic"].strip()

    # ── RAG SEARCH ────────────────────────────────────────────────────────────
    rag_chunks  = []
    rag_sources = []
    chunks_found = 0

    try:
        # Search with topic filter first
        results = search_chunks(topic, topic=topic, top_k=8, threshold=0.2)

        # Broader search if nothing found
        if not results:
            results = search_chunks(topic, topic=None, top_k=5, threshold=0.25)

        if results:
            rag_chunks   = [r["content"] for r in results]
            rag_sources  = [r.get("source", "notes") for r in results]
            chunks_found = len(results)
            logger.info(f"[whiteboard] RAG found {chunks_found} chunks for '{topic}' "
                       f"from {set(rag_sources)}")
        else:
            logger.info(f"[whiteboard] No RAG chunks for '{topic}' — using AI knowledge")

    except Exception as e:
        logger.warning(f"[whiteboard] RAG error (non-fatal): {e}")

    # ── DETECT SUBJECT ────────────────────────────────────────────────────────
    subject_type = detect_subject_type(topic)

    # ── BUILD PROMPT ──────────────────────────────────────────────────────────
    prompt = build_prompt(topic, subject_type, rag_chunks, rag_sources)

    # ── CALL AI (no json_mode — avoids Groq 400 errors) ──────────────────────
    raw = ""
    try:
        raw   = ask_ai(prompt, json_mode=False)
        clean = clean_json(raw)

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Retry — ask AI to fix its own JSON
            logger.warning("[whiteboard] First JSON parse failed — retrying")
            fix_prompt = (
                "The following is almost valid JSON but has syntax errors. "
                "Fix ONLY the JSON syntax and return the corrected JSON object. "
                "Do not change any content, just fix syntax:\n\n"
                + clean[:4000]
            )
            raw2  = ask_ai(fix_prompt, json_mode=False)
            clean = clean_json(raw2)
            parsed = json.loads(clean)

        lesson = parsed.get("lesson", parsed)

        if "steps" not in lesson or not lesson["steps"]:
            raise ValueError("No steps in lesson")

        # ── BUILD SOURCE CITATION ─────────────────────────────────────────────
        unique_sources = list(set(rag_sources)) if rag_sources else []
        citation = None
        if unique_sources:
            # Clean up filenames for display
            clean_sources = []
            for s in unique_sources:
                # Remove timestamp prefix if present (e.g. 1234567890_docker.pdf → docker.pdf)
                parts = s.split("_", 1)
                name  = parts[1] if len(parts) > 1 and parts[0].isdigit() else s
                clean_sources.append(name)
            citation = {
                "sources":      clean_sources,
                "raw_sources":  unique_sources,
                "chunks_used":  chunks_found,
                "message":      f"This lesson was built from your uploaded notes: {', '.join(clean_sources)}"
            }

        logger.info(f"[whiteboard] Lesson generated: topic={topic} "
                   f"rag={bool(rag_chunks)} chunks={chunks_found} steps={len(lesson['steps'])}")

        return {
            "lesson":    lesson,
            "rag_used":  bool(rag_chunks),
            "chunks":    chunks_found,
            "citation":  citation       # NEW — source reference for frontend display
        }

    except json.JSONDecodeError as e:
        logger.error(f"[whiteboard] JSON error: {e}\nRaw: {raw[:400]}")
        return {"error": "Could not parse lesson. Please try again."}, 500
    except Exception as e:
        logger.error(f"[whiteboard] Error: {e}")
        return {"error": "AI service unavailable. Please try again."}, 503
