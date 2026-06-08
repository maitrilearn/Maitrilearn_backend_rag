from flask import Blueprint, request
from services.groq_service import ask_ai
import json, re

terminal_bp = Blueprint("terminal", __name__)


@terminal_bp.route("/terminal/run", methods=["POST"])
def run_command():
    """
    Student types a command → AI generates realistic terminal output.
    Also evaluates if command is correct for active challenge.
    """
    data    = request.json
    if not data or not data.get("command"):
        return {"error": "command is required"}, 400

    command   = data["command"].strip()
    context   = data.get("context", "general")   # docker / linux / git / k8s
    challenge = data.get("challenge", "")         # current challenge description
    history   = data.get("history", [])           # previous commands in session

    # Build history string (last 5 commands for context)
    history_str = ""
    if history:
        history_str = "\n".join([
            f"$ {h['cmd']}\n{h['out']}" for h in history[-5:]
        ])

    prompt = f"""You are a realistic {context} terminal simulator.
The student typed this command: {command}

Previous session history:
{history_str if history_str else "(no history yet)"}

{"Active challenge: " + challenge if challenge else ""}

Rules:
- Generate REALISTIC terminal output exactly as a real {context} environment would show
- Keep output concise — real terminals are concise
- For invalid/wrong commands show realistic error messages
- For docker commands simulate real docker output with fake container IDs, image names
- For git commands show realistic git output
- For linux commands show realistic linux output
- For kubectl commands show realistic kubernetes output
- Never break character — always respond as the terminal, never as an AI
- If there is an active challenge, after the output add a JSON block to evaluate:

Return ONLY this JSON — no markdown, no explanation:
{{
  "output": "the terminal output here\\nwith newlines",
  "success": true or false,
  "challenge_completed": true if command completes the challenge else false,
  "hint": "helpful hint if command was wrong or incomplete, empty string if correct",
  "context_update": "any state change e.g. current directory, running containers"
}}

Topic context: {context}
Command: {command}
"""

    try:
        raw    = ask_ai(prompt, json_mode=True)
        clean  = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)
        return {
            "output":              result.get("output", ""),
            "challenge_completed": result.get("challenge_completed", False),
            "hint":                result.get("hint", ""),
            "context_update":      result.get("context_update", ""),
            "success":             result.get("success", True)
        }
    except json.JSONDecodeError as e:
        print(f"[terminal] JSON error: {e}\nRaw: {raw[:300]}")
        return {"error": "Terminal error. Try again."}, 500
    except Exception as e:
        print(f"[terminal] Error: {e}")
        return {"error": "AI service unavailable."}, 503


@terminal_bp.route("/terminal/challenges", methods=["GET"])
def get_challenges():
    """Returns all challenges organized by topic and day."""
    return {"challenges": CHALLENGES}


# ── CHALLENGE DATA ─────────────────────────────────────────────────────────────
CHALLENGES = {
    "linux": [
        {
            "id": "linux_1",
            "day": 1,
            "title": "Navigate the Filesystem",
            "description": "List all files in the current directory including hidden files",
            "hint": "Try ls with a flag that shows hidden files",
            "solution": "ls -la",
            "context": "linux",
            "difficulty": "easy"
        },
        {
            "id": "linux_2",
            "day": 1,
            "title": "Create a Directory",
            "description": "Create a directory called 'myproject' inside /tmp",
            "hint": "Use mkdir command with the full path",
            "solution": "mkdir /tmp/myproject",
            "context": "linux",
            "difficulty": "easy"
        },
        {
            "id": "linux_3",
            "day": 2,
            "title": "Set File Permissions",
            "description": "Give the owner read+write+execute, group read+execute, others no permissions on file.sh",
            "hint": "Use chmod with octal notation — owner=7, group=5, others=0",
            "solution": "chmod 750 file.sh",
            "context": "linux",
            "difficulty": "medium"
        },
        {
            "id": "linux_4",
            "day": 3,
            "title": "Create a Soft Link",
            "description": "Create a symbolic link named 'latest' pointing to /var/log/app.log",
            "hint": "Use ln -s source destination",
            "solution": "ln -s /var/log/app.log latest",
            "context": "linux",
            "difficulty": "medium"
        },
        {
            "id": "linux_5",
            "day": 5,
            "title": "Find Running Processes",
            "description": "Show all running processes with their PID, CPU and memory usage",
            "hint": "Use ps with aux flags",
            "solution": "ps aux",
            "context": "linux",
            "difficulty": "easy"
        },
        {
            "id": "linux_6",
            "day": 6,
            "title": "Search in Files",
            "description": "Find all lines containing 'ERROR' in /var/log/app.log",
            "hint": "Use grep command",
            "solution": "grep ERROR /var/log/app.log",
            "context": "linux",
            "difficulty": "easy"
        }
    ],
    "git": [
        {
            "id": "git_1",
            "day": 1,
            "title": "Initialize a Repository",
            "description": "Initialize a new git repository in the current directory",
            "hint": "Use git init",
            "solution": "git init",
            "context": "git",
            "difficulty": "easy"
        },
        {
            "id": "git_2",
            "day": 2,
            "title": "Stage and Commit",
            "description": "Stage all changes and commit with message 'initial commit'",
            "hint": "Use git add . then git commit -m",
            "solution": "git add . && git commit -m 'initial commit'",
            "context": "git",
            "difficulty": "easy"
        },
        {
            "id": "git_3",
            "day": 3,
            "title": "Create a Branch",
            "description": "Create and switch to a new branch called 'feature/login'",
            "hint": "Use git checkout -b to create and switch in one command",
            "solution": "git checkout -b feature/login",
            "context": "git",
            "difficulty": "medium"
        },
        {
            "id": "git_4",
            "day": 4,
            "title": "Push to Remote",
            "description": "Push your current branch to origin remote",
            "hint": "Use git push with origin and branch name",
            "solution": "git push origin feature/login",
            "context": "git",
            "difficulty": "medium"
        }
    ],
    "docker": [
        {
            "id": "docker_1",
            "day": 1,
            "title": "Run Your First Container",
            "description": "Run the hello-world Docker container",
            "hint": "Use docker run with the image name",
            "solution": "docker run hello-world",
            "context": "docker",
            "difficulty": "easy"
        },
        {
            "id": "docker_2",
            "day": 2,
            "title": "List Running Containers",
            "description": "Show all running containers including stopped ones",
            "hint": "Use docker ps with a flag to show all containers",
            "solution": "docker ps -a",
            "context": "docker",
            "difficulty": "easy"
        },
        {
            "id": "docker_3",
            "day": 2,
            "title": "Pull an Image",
            "description": "Pull the official nginx image from Docker Hub",
            "hint": "Use docker pull with the image name",
            "solution": "docker pull nginx",
            "context": "docker",
            "difficulty": "easy"
        },
        {
            "id": "docker_4",
            "day": 2,
            "title": "Run nginx Container",
            "description": "Run nginx container in detached mode mapping port 8080 on host to 80 in container",
            "hint": "Use docker run with -d and -p flags",
            "solution": "docker run -d -p 8080:80 nginx",
            "context": "docker",
            "difficulty": "medium"
        },
        {
            "id": "docker_5",
            "day": 3,
            "title": "Build an Image",
            "description": "Build a Docker image tagged as 'myapp:v1' from current directory",
            "hint": "Use docker build with -t flag and . for current directory",
            "solution": "docker build -t myapp:v1 .",
            "context": "docker",
            "difficulty": "medium"
        },
        {
            "id": "docker_6",
            "day": 4,
            "title": "Create a Volume",
            "description": "Create a named Docker volume called 'mydata'",
            "hint": "Use docker volume create",
            "solution": "docker volume create mydata",
            "context": "docker",
            "difficulty": "easy"
        },
        {
            "id": "docker_7",
            "day": 5,
            "title": "Start with Compose",
            "description": "Start all services defined in docker-compose.yml in detached mode",
            "hint": "Use docker-compose up with -d flag",
            "solution": "docker-compose up -d",
            "context": "docker",
            "difficulty": "medium"
        }
    ],
    "kubernetes": [
        {
            "id": "k8s_1",
            "day": 2,
            "title": "Get All Pods",
            "description": "List all pods in all namespaces",
            "hint": "Use kubectl get pods with a flag for all namespaces",
            "solution": "kubectl get pods --all-namespaces",
            "context": "kubernetes",
            "difficulty": "easy"
        },
        {
            "id": "k8s_2",
            "day": 3,
            "title": "Create a Deployment",
            "description": "Create a deployment named 'webapp' using nginx image with 3 replicas",
            "hint": "Use kubectl create deployment with --image and --replicas flags",
            "solution": "kubectl create deployment webapp --image=nginx --replicas=3",
            "context": "kubernetes",
            "difficulty": "medium"
        },
        {
            "id": "k8s_3",
            "day": 4,
            "title": "Expose a Deployment",
            "description": "Expose the 'webapp' deployment as a NodePort service on port 80",
            "hint": "Use kubectl expose deployment with --type=NodePort",
            "solution": "kubectl expose deployment webapp --type=NodePort --port=80",
            "context": "kubernetes",
            "difficulty": "medium"
        },
        {
            "id": "k8s_4",
            "day": 3,
            "title": "Scale a Deployment",
            "description": "Scale the 'webapp' deployment to 5 replicas",
            "hint": "Use kubectl scale deployment with --replicas flag",
            "solution": "kubectl scale deployment webapp --replicas=5",
            "context": "kubernetes",
            "difficulty": "medium"
        },
        {
            "id": "k8s_5",
            "day": 5,
            "title": "Create a ConfigMap",
            "description": "Create a ConfigMap named 'app-config' with key APP_ENV=production",
            "hint": "Use kubectl create configmap with --from-literal",
            "solution": "kubectl create configmap app-config --from-literal=APP_ENV=production",
            "context": "kubernetes",
            "difficulty": "hard"
        }
    ]
}
