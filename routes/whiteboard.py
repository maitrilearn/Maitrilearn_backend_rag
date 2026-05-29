from flask import Blueprint, request
from services.groq_service import ask_ai
import json, re

whiteboard_bp = Blueprint("whiteboard", __name__)


@whiteboard_bp.route("/whiteboard/lesson", methods=["POST"])
def whiteboard_lesson():
    data = request.json
    if not data or not data.get("topic"):
        return {"error": "topic is required"}, 400

    topic = data["topic"]

    prompt = f"""You are an experienced classroom teacher. Create a step-by-step lesson for: "{topic}"

TEACHING RULES:
- Each step teaches ONE concept only
- Explanations must be SHORT (1-2 sentences max)
- Always give a real-world analogy or example
- Narration must sound like a real teacher — conversational and warm
- Never use jargon without explaining it first

Return ONLY valid JSON. No markdown, no backticks, no text outside JSON.

{{
  "title": "lesson title",
  "subject": "subject area",
  "steps": [
    {{
      "type": "title",
      "text": "topic name",
      "subtitle": "one sentence hook — why this matters",
      "narration": "warm greeting + why student should care (2 sentences)"
    }},
    {{
      "type": "concept",
      "heading": "concept name",
      "definition": "one clear sentence definition",
      "analogy": "real-world analogy that makes it click",
      "narration": "teacher explains using the analogy (2 sentences)"
    }},
    {{
      "type": "steps",
      "heading": "how it works",
      "items": [
        {{"step": "Step name", "detail": "what happens"}},
        {{"step": "Step name", "detail": "what happens"}},
        {{"step": "Step name", "detail": "what happens"}},
        {{"step": "Step name", "detail": "what happens"}}
      ],
      "narration": "teacher walks through the steps"
    }},
    {{
      "type": "example",
      "heading": "Real Example",
      "scenario": "concrete real-world scenario in 1 sentence",
      "code": "actual command or code if applicable, empty string if not",
      "explanation": "what the example shows in 1-2 sentences",
      "narration": "teacher explains the example"
    }},
    {{
      "type": "diagram",
      "heading": "diagram title",
      "elements": [
        {{"label": "Component A", "description": "what it does", "role": "input"}},
        {{"label": "Component B", "description": "what it does", "role": "process"}},
        {{"label": "Component C", "description": "what it does", "role": "output"}}
      ],
      "narration": "teacher describes the structure"
    }},
    {{
      "type": "comparison",
      "heading": "Without vs With {topic}",
      "left_label": "Without {topic}",
      "left_points": ["problem 1", "problem 2", "problem 3"],
      "right_label": "With {topic}",
      "right_points": ["solution 1", "solution 2", "solution 3"],
      "narration": "teacher explains why this technology exists"
    }},
    {{
      "type": "keypoints",
      "heading": "Remember This",
      "points": [
        {{"icon": "⚡", "text": "most important thing to remember"}},
        {{"icon": "🎯", "text": "second key insight"}},
        {{"icon": "💡", "text": "practical tip or common mistake to avoid"}}
      ],
      "narration": "teacher summarizes warmly and encourages student"
    }}
  ]
}}

Generate 6-8 steps. Always start with title. Always end with keypoints.
Include at least: 1 concept, 1 steps, 1 example, 1 diagram or comparison.
Topic: {topic}
"""

    try:
        raw   = ask_ai(prompt)
        clean = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(clean)
        lesson = parsed.get("lesson", parsed)
        if "steps" not in lesson or not lesson["steps"]:
            raise ValueError("No steps returned")
        return {"lesson": lesson}

    except json.JSONDecodeError as e:
        print(f"[whiteboard] JSON error: {e}\nRaw: {raw[:400]}")
        return {"error": "AI returned invalid format. Try again."}, 500
    except Exception as e:
        print(f"[whiteboard] Error: {e}")
        return {"error": "AI service unavailable. Try again."}, 503
