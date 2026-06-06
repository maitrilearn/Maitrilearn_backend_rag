from flask import Blueprint, request
from services.groq_service import ask_ai
from services.rag_service import search_chunks
import json, re

whiteboard_bp = Blueprint("whiteboard", __name__)


@whiteboard_bp.route("/whiteboard/lesson", methods=["POST"])
def whiteboard_lesson():
    data = request.json
    if not data or not data.get("topic"):
        return {"error": "topic is required"}, 400

    topic = data["topic"]

    # RAG search
    rag_context = ""
    chunks_found = 0
    try:
        chunks = search_chunks(topic, topic=topic, top_k=5, threshold=0.25)
        if chunks:
            rag_context = "\n\n".join(chunks)
            chunks_found = len(chunks)
    except Exception as e:
        print(f"[whiteboard] RAG error: {e}")

    context_section = f"""
USE THIS REFERENCE MATERIAL (prefer it over general knowledge):
---
{rag_context[:3000]}
---
""" if rag_context else "(No reference material — use your own knowledge)"

    prompt = f"""You are an expert classroom teacher and visual explainer.
Create a rich, visual, step-by-step lesson for: "{topic}"

{context_section}

TEACHING RULES:
- Each step teaches ONE concept
- Use SHORT explanations (1-2 sentences)
- Always include real DevOps examples with actual commands/code
- Make narration sound like a real teacher — warm, conversational
- For DevOps topics: always include architecture diagram, code examples, and timeline

Return ONLY valid JSON. No markdown, no backticks.

Available step types (use a mix for rich visual experience):

1. title — lesson opener
2. concept — definition + analogy
3. steps — numbered process steps
4. example — real example with code
5. codefile — full code file with syntax highlighting
6. architecture — component diagram with connections
7. flowdiagram — process flow with nodes
8. timeline — pipeline/process timeline
9. comparison — before vs after
10. keypoints — summary

Return this JSON structure:
{{
  "title": "lesson title",
  "subject": "subject area",
  "steps": [
    {{
      "type": "title",
      "text": "topic name",
      "subtitle": "why this matters in one sentence",
      "narration": "warm intro (2 sentences)"
    }},
    {{
      "type": "concept",
      "heading": "What is {topic}?",
      "definition": "one clear sentence",
      "analogy": "real-world analogy",
      "narration": "teacher explains with analogy"
    }},
    {{
      "type": "architecture",
      "heading": "How {topic} is structured",
      "nodes": [
        {{"id": "a", "label": "Component A", "description": "what it does", "role": "input"}},
        {{"id": "b", "label": "Component B", "description": "what it does", "role": "process"}},
        {{"id": "c", "label": "Component C", "description": "what it does", "role": "output"}}
      ],
      "connections": [
        {{"from": "a", "to": "b", "label": "sends to"}},
        {{"from": "b", "to": "c", "label": "outputs"}}
      ],
      "narration": "teacher describes architecture components"
    }},
    {{
      "type": "flowdiagram",
      "heading": "Process Flow",
      "nodes": [
        {{"label": "Step 1", "sublabel": "detail", "type": "process"}},
        {{"label": "Step 2", "sublabel": "detail", "type": "process"}},
        {{"label": "Step 3", "sublabel": "detail", "type": "process"}},
        {{"label": "Step 4", "sublabel": "detail", "type": "process"}}
      ],
      "narration": "teacher walks through the flow"
    }},
    {{
      "type": "codefile",
      "heading": "Real Code Example",
      "filename": "Dockerfile or docker-compose.yml or relevant filename",
      "lang": "dockerfile or yaml or bash",
      "description": "what this file does",
      "code": "actual real code here\\nwith newlines",
      "explanation": "what each part does",
      "narration": "teacher explains the code"
    }},
    {{
      "type": "example",
      "heading": "Try It Yourself",
      "scenario": "concrete scenario",
      "code": "$ actual command here",
      "lang": "bash",
      "explanation": "what happens when you run this",
      "narration": "teacher explains the example"
    }},
    {{
      "type": "timeline",
      "heading": "Pipeline / Process Timeline",
      "nodes": [
        {{"label": "Stage 1", "sublabel": "what happens", "duration": "~30s"}},
        {{"label": "Stage 2", "sublabel": "what happens", "duration": "~2min"}},
        {{"label": "Stage 3", "sublabel": "what happens", "duration": "~5min"}},
        {{"label": "Stage 4", "sublabel": "what happens", "duration": "~1min"}}
      ],
      "narration": "teacher explains each stage"
    }},
    {{
      "type": "comparison",
      "heading": "Without vs With {topic}",
      "left_label": "Without {topic}",
      "left_points": ["problem 1", "problem 2", "problem 3"],
      "right_label": "With {topic}",
      "right_points": ["solution 1", "solution 2", "solution 3"],
      "narration": "teacher explains the contrast"
    }},
    {{
      "type": "keypoints",
      "heading": "Remember This",
      "points": [
        {{"icon": "⚡", "text": "most important insight"}},
        {{"icon": "🎯", "text": "practical tip"}},
        {{"icon": "💡", "text": "common mistake to avoid"}}
      ],
      "narration": "teacher summarizes warmly"
    }}
  ]
}}

Generate 7-9 steps for: {topic}
MUST include: title, at least one of (architecture OR flowdiagram), at least one timeline, at least one codefile, keypoints.
Make code examples REAL and accurate for the topic.
"""

    try:
        raw    = ask_ai(prompt, json_mode=True)
        clean  = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(clean)
        lesson = parsed.get("lesson", parsed)

        if "steps" not in lesson or not lesson["steps"]:
            raise ValueError("No steps returned")

        return {
            "lesson":   lesson,
            "rag_used": bool(rag_context),
            "chunks":   chunks_found
        }

    except json.JSONDecodeError as e:
        print(f"[whiteboard] JSON error: {e}\nRaw: {raw[:400]}")
        return {"error": "AI returned invalid format. Try again."}, 500
    except Exception as e:
        print(f"[whiteboard] Error: {e}")
        return {"error": "AI service unavailable. Try again."}, 503
