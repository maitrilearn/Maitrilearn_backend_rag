from flask import Blueprint, request
from services.groq_service import ask_ai
from services.rag_service import search_chunks
import json, re

whiteboard_bp = Blueprint("whiteboard", __name__)


def clean_json(raw: str) -> str:
    """Strip markdown fences and extract first JSON object."""
    raw = re.sub(r"```json|```", "", raw).strip()
    # Find first { and last } to extract JSON object
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1:
        return raw[start:end+1]
    return raw


def detect_subject_type(topic: str) -> str:
    """
    Detect what kind of subject the topic is so we can
    generate appropriate step types (code for CS, diagrams for science, etc.)
    """
    topic_lower = topic.lower()

    devops_keywords = ["docker","kubernetes","k8s","linux","git","ci/cd","pipeline",
                       "terraform","ansible","aws","nginx","bash","shell","yaml","helm"]
    science_keywords = ["photosynthesis","biology","chemistry","physics","atom","molecule",
                        "cell","mitosis","osmosis","reaction","force","gravity","newton"]
    math_keywords    = ["calculus","algebra","geometry","trigonometry","equation","theorem",
                        "derivative","integral","matrix","vector","probability","statistics"]
    history_keywords = ["war","revolution","empire","civilization","history","ancient",
                        "medieval","colonial","independence","century"]
    cs_keywords      = ["python","javascript","java","algorithm","data structure","sorting",
                        "recursion","api","database","sql","machine learning","neural"]

    for k in devops_keywords:
        if k in topic_lower: return "devops"
    for k in science_keywords:
        if k in topic_lower: return "science"
    for k in math_keywords:
        if k in topic_lower: return "math"
    for k in history_keywords:
        if k in topic_lower: return "history"
    for k in cs_keywords:
        if k in topic_lower: return "cs"
    return "general"


def build_prompt(topic: str, subject_type: str, rag_context: str) -> str:
    """Build subject-aware prompt with RAG context."""

    context_section = f"""
USE THIS REFERENCE MATERIAL from the student's uploaded notes (prioritize this over general knowledge):
---
{rag_context[:3000]}
---
""" if rag_context else "(No uploaded notes found — use your best knowledge to teach this topic)"

    # Subject-specific instructions
    subject_hints = {
        "devops":  "Include architecture diagrams, real CLI commands, Dockerfiles/YAMLs, and deployment timelines.",
        "science": "Include diagrams showing biological/chemical structures, process flows (e.g. photosynthesis steps), and real-world examples.",
        "math":    "Include step-by-step worked examples, formula explanations, and visual diagrams. Use plain text for formulas.",
        "history": "Include timelines of key events, comparison of before/after, key figures, and cause-effect flows.",
        "cs":      "Include code examples with syntax highlighting, algorithm flow diagrams, and complexity comparisons.",
        "general": "Include diagrams, real-world analogies, step-by-step explanations, and key takeaways."
    }

    hint = subject_hints.get(subject_type, subject_hints["general"])

    return f"""You are an expert classroom teacher for ALL subjects — science, math, history, DevOps, programming, and more.
Create a rich, visual, step-by-step lesson for: "{topic}"

{context_section}

SUBJECT GUIDANCE: {hint}

TEACHING RULES:
- Each step teaches ONE concept only
- SHORT explanations (1-2 sentences max per step)
- Always use real examples relevant to the topic
- Narration must sound like a warm, encouraging teacher
- If RAG material is provided, base your lesson on it

IMPORTANT: Return ONLY a valid JSON object. No markdown, no backticks, no text before or after the JSON.

Use these step types (pick the most appropriate mix for the subject):
- title: lesson opener with subtitle
- concept: definition + real-world analogy
- steps: numbered process (how something works)
- example: real example with optional code
- codefile: full code file (for CS/DevOps only)
- architecture: component boxes + arrows (for systems/science)
- flowdiagram: left-to-right process flow
- timeline: stages with durations (great for history, CI/CD, biology cycles)
- comparison: before vs after / without vs with
- keypoints: final summary

Return this exact JSON structure:
{{
  "title": "lesson title",
  "subject": "subject area e.g. Biology / Docker / World History",
  "steps": [
    {{
      "type": "title",
      "text": "{topic}",
      "subtitle": "one sentence why this matters",
      "narration": "warm engaging intro in 2 sentences"
    }},
    {{
      "type": "concept",
      "heading": "What is {topic}?",
      "definition": "clear 1-sentence definition",
      "analogy": "relatable real-world analogy",
      "narration": "teacher explains using the analogy"
    }},
    {{
      "type": "flowdiagram",
      "heading": "How it works",
      "nodes": [
        {{"label": "Stage 1", "sublabel": "brief detail"}},
        {{"label": "Stage 2", "sublabel": "brief detail"}},
        {{"label": "Stage 3", "sublabel": "brief detail"}},
        {{"label": "Stage 4", "sublabel": "brief detail"}}
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
      "scenario": "concrete real-world scenario",
      "code": "command or formula or code if applicable else empty string",
      "lang": "bash or python or text",
      "explanation": "what this example shows",
      "narration": "teacher explains the example"
    }},
    {{
      "type": "timeline",
      "heading": "Key Stages / Timeline",
      "nodes": [
        {{"label": "Stage 1", "sublabel": "what happens", "duration": "time or phase"}},
        {{"label": "Stage 2", "sublabel": "what happens", "duration": "time or phase"}},
        {{"label": "Stage 3", "sublabel": "what happens", "duration": "time or phase"}},
        {{"label": "Stage 4", "sublabel": "what happens", "duration": "time or phase"}}
      ],
      "narration": "teacher explains the progression"
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
        {{"icon": "⚡", "text": "most important thing to remember"}},
        {{"icon": "🎯", "text": "key insight or application"}},
        {{"icon": "💡", "text": "common mistake or exam tip"}}
      ],
      "narration": "teacher wraps up warmly"
    }}
  ]
}}

Generate 6-8 steps. MUST start with title, MUST end with keypoints.
Pick step types that make sense for: {topic}
"""


@whiteboard_bp.route("/whiteboard/lesson", methods=["POST"])
def whiteboard_lesson():
    data = request.json
    if not data or not data.get("topic"):
        return {"error": "topic is required"}, 400

    topic = data["topic"].strip()

    # ── RAG: search across ALL topics (no topic filter for universal search) ──
    rag_context  = ""
    chunks_found = 0
    try:
        # First try with topic filter
        chunks = search_chunks(topic, topic=topic, top_k=5, threshold=0.25)
        # If nothing found, try without topic filter (broader search)
        if not chunks:
            chunks = search_chunks(topic, topic=None, top_k=3, threshold=0.3)
        if chunks:
            rag_context  = "\n\n".join(chunks)
            chunks_found = len(chunks)
            print(f"[whiteboard] RAG found {chunks_found} chunks for '{topic}'")
        else:
            print(f"[whiteboard] No RAG chunks found for '{topic}' — using AI knowledge")
    except Exception as e:
        print(f"[whiteboard] RAG error (non-fatal): {e}")

    # Detect subject type for better prompt
    subject_type = detect_subject_type(topic)
    prompt       = build_prompt(topic, subject_type, rag_context)

    # ── BUG FIX: Don't use json_mode=True — it causes Groq 400 errors ────────
    # Parse JSON manually instead — more reliable
    raw = ""
    try:
        raw   = ask_ai(prompt, json_mode=False)
        clean = clean_json(raw)

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Second attempt — ask AI to fix its own JSON
            print(f"[whiteboard] First parse failed, retrying...")
            fix_prompt = f"""The following is almost valid JSON but has errors. Fix it and return ONLY the corrected JSON object, nothing else:

{clean[:4000]}"""
            raw2  = ask_ai(fix_prompt, json_mode=False)
            clean = clean_json(raw2)
            parsed = json.loads(clean)

        lesson = parsed.get("lesson", parsed)

        if "steps" not in lesson or not lesson["steps"]:
            raise ValueError("No steps in lesson")

        return {
            "lesson":   lesson,
            "rag_used": bool(rag_context),
            "chunks":   chunks_found
        }

    except json.JSONDecodeError as e:
        print(f"[whiteboard] JSON error after retry: {e}\nRaw: {raw[:400]}")
        return {"error": "Could not parse lesson. Please try again."}, 500
    except Exception as e:
        print(f"[whiteboard] Error: {e}")
        return {"error": "AI service unavailable. Please try again."}, 503
