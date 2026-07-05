import logging
import json
import re
from flask import Blueprint, request
from services.groq_service import ask_ai
from services.rag_service import search_chunks

whiteboard_bp = Blueprint("whiteboard", __name__)
logger = logging.getLogger("maitrilearn")


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
- keypoints: final summary

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
      "type": "keypoints",
      "heading": "Remember This",
      "points": [
        {{"icon": "⚡", "text": "most important single fact about this topic"}},
        {{"icon": "🎯", "text": "key practical application or insight"}},
        {{"icon": "💡", "text": "common mistake or exam tip"}}
      ],
      "narration": "teacher summarizes warmly and encourages the student"
    }}
  ]
}}

Generate 7-9 steps. MUST start with title. MUST end with keypoints.
MUST include architecture with 5+ REAL named components for engineering/science/devops topics.
MUST include chalkboard step for important facts.
Topic: {topic}
"""


@whiteboard_bp.route("/whiteboard/lesson", methods=["POST"])
def whiteboard_lesson():
    data = request.get_json(silent=True) or {}
    if not data.get("topic"):
        return {"error": "topic is required"}, 400

    topic = data["topic"].strip()

    rag_chunks  = []
    rag_sources = []
    chunks_found = 0

    try:
        results = search_chunks(topic, topic=topic, top_k=8, threshold=0.2)
        if not results:
            results = search_chunks(topic, topic=None, top_k=5, threshold=0.25)
        if results:
            rag_chunks   = [r["content"] for r in results]
            rag_sources  = [r.get("source", "notes") for r in results]
            chunks_found = len(results)
            logger.info(f"[whiteboard] RAG found {chunks_found} chunks for '{topic}'")
        else:
            logger.info(f"[whiteboard] No RAG chunks for '{topic}' — using AI knowledge")
    except Exception as e:
        logger.warning(f"[whiteboard] RAG error (non-fatal): {e}")

    subject_type = detect_subject_type(topic)
    prompt       = build_prompt(topic, subject_type, rag_chunks, rag_sources)

    raw = ""
    try:
        raw   = ask_ai(prompt, json_mode=False)
        clean = clean_json(raw)

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("[whiteboard] First JSON parse failed — retrying")
            fix_prompt = (
                "The following is almost valid JSON but has syntax errors. "
                "Fix ONLY the JSON syntax and return the corrected JSON object. "
                "Do not change any content, just fix syntax:\n\n" + clean[:4000]
            )
            raw2  = ask_ai(fix_prompt, json_mode=False)
            clean = clean_json(raw2)
            parsed = json.loads(clean)

        lesson = parsed.get("lesson", parsed)

        if "steps" not in lesson or not lesson["steps"]:
            raise ValueError("No steps in lesson")

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
                   f"rag={bool(rag_chunks)} chunks={chunks_found} steps={len(lesson['steps'])}")

        return {
            "lesson":       lesson,
            "rag_used":     bool(rag_chunks),
            "chunks":       chunks_found,
            "citation":     citation,
            "subject_type": subject_type
        }

    except json.JSONDecodeError as e:
        logger.error(f"[whiteboard] JSON error: {e}\nRaw: {raw[:400]}")
        return {"error": "Could not parse lesson. Please try again."}, 500
    except Exception as e:
        logger.error(f"[whiteboard] Error: {e}")
        return {"error": "AI service unavailable. Please try again."}, 503
