import logging
import json
import re
from flask import Blueprint, request
from services.groq_service import ask_ai
from utils.validator import validate_command, validate_text, ValidationError

terminal_bp = Blueprint("terminal", __name__)
logger = logging.getLogger("maitrilearn")

# Same challenge data as before
CHALLENGES = {
    "linux": [
      {"id":"linux_1","day":1,"title":"List Files","description":"List all files including hidden ones","hint":"Try ls with -la flags","solution":"ls -la","context":"linux","difficulty":"easy"},
      {"id":"linux_2","day":1,"title":"Create Directory","description":"Create a directory called myproject in /tmp","hint":"mkdir /tmp/myproject","solution":"mkdir /tmp/myproject","context":"linux","difficulty":"easy"},
      {"id":"linux_3","day":2,"title":"File Permissions","description":"Give owner rwx, group rx, others no permissions on file.sh","hint":"chmod 750 file.sh","solution":"chmod 750 file.sh","context":"linux","difficulty":"medium"},
      {"id":"linux_4","day":3,"title":"Create Soft Link","description":"Create a symbolic link named 'latest' pointing to /var/log/app.log","hint":"ln -s source destination","solution":"ln -s /var/log/app.log latest","context":"linux","difficulty":"medium"},
      {"id":"linux_5","day":5,"title":"Find Running Processes","description":"Show all running processes with PID, CPU and memory","hint":"ps aux","solution":"ps aux","context":"linux","difficulty":"easy"},
      {"id":"linux_6","day":6,"title":"Search in Files","description":"Find all lines containing 'ERROR' in /var/log/app.log","hint":"grep ERROR /var/log/app.log","solution":"grep ERROR /var/log/app.log","context":"linux","difficulty":"easy"},
    ],
    "git": [
      {"id":"git_1","day":1,"title":"Initialize Repo","description":"Initialize a new git repository","hint":"git init","solution":"git init","context":"git","difficulty":"easy"},
      {"id":"git_2","day":2,"title":"Stage & Commit","description":"Stage all changes and commit with message 'initial commit'","hint":"git add . then git commit -m","solution":"git add .","context":"git","difficulty":"easy"},
      {"id":"git_3","day":3,"title":"Create Branch","description":"Create and switch to a new branch called 'feature/login'","hint":"git checkout -b","solution":"git checkout -b feature/login","context":"git","difficulty":"medium"},
      {"id":"git_4","day":4,"title":"Push to Remote","description":"Push your current branch to origin","hint":"git push origin branch-name","solution":"git push origin feature/login","context":"git","difficulty":"medium"},
    ],
    "docker": [
      {"id":"docker_1","day":1,"title":"Hello World","description":"Run the hello-world Docker container","hint":"docker run hello-world","solution":"docker run hello-world","context":"docker","difficulty":"easy"},
      {"id":"docker_2","day":2,"title":"List Containers","description":"Show all containers including stopped ones","hint":"docker ps -a","solution":"docker ps -a","context":"docker","difficulty":"easy"},
      {"id":"docker_3","day":2,"title":"Pull nginx","description":"Pull the official nginx image from Docker Hub","hint":"docker pull nginx","solution":"docker pull nginx","context":"docker","difficulty":"easy"},
      {"id":"docker_4","day":2,"title":"Run nginx","description":"Run nginx in detached mode mapping port 8080 to 80","hint":"docker run -d -p 8080:80 nginx","solution":"docker run -d -p 8080:80 nginx","context":"docker","difficulty":"medium"},
      {"id":"docker_5","day":3,"title":"Build Image","description":"Build a Docker image tagged as 'myapp:v1' from current directory","hint":"docker build -t myapp:v1 .","solution":"docker build -t myapp:v1 .","context":"docker","difficulty":"medium"},
      {"id":"docker_6","day":4,"title":"Create Volume","description":"Create a named Docker volume called 'mydata'","hint":"docker volume create mydata","solution":"docker volume create mydata","context":"docker","difficulty":"easy"},
      {"id":"docker_7","day":5,"title":"Start Compose","description":"Start all services in docker-compose.yml in detached mode","hint":"docker-compose up -d","solution":"docker-compose up -d","context":"docker","difficulty":"medium"},
    ],
    "kubernetes": [
      {"id":"k8s_1","day":2,"title":"Get All Pods","description":"List all pods in all namespaces","hint":"kubectl get pods --all-namespaces","solution":"kubectl get pods --all-namespaces","context":"kubernetes","difficulty":"easy"},
      {"id":"k8s_2","day":3,"title":"Create Deployment","description":"Create deployment 'webapp' using nginx with 3 replicas","hint":"kubectl create deployment with --image and --replicas","solution":"kubectl create deployment webapp --image=nginx --replicas=3","context":"kubernetes","difficulty":"medium"},
      {"id":"k8s_3","day":4,"title":"Expose Deployment","description":"Expose 'webapp' as NodePort service on port 80","hint":"kubectl expose deployment --type=NodePort","solution":"kubectl expose deployment webapp --type=NodePort --port=80","context":"kubernetes","difficulty":"medium"},
      {"id":"k8s_4","day":3,"title":"Scale Deployment","description":"Scale 'webapp' deployment to 5 replicas","hint":"kubectl scale deployment --replicas=5","solution":"kubectl scale deployment webapp --replicas=5","context":"kubernetes","difficulty":"medium"},
      {"id":"k8s_5","day":5,"title":"Create ConfigMap","description":"Create ConfigMap 'app-config' with APP_ENV=production","hint":"kubectl create configmap --from-literal","solution":"kubectl create configmap app-config --from-literal=APP_ENV=production","context":"kubernetes","difficulty":"hard"},
    ]
}


@terminal_bp.route("/terminal/challenges", methods=["GET"])
def get_challenges():
    return {"challenges": CHALLENGES}


@terminal_bp.route("/terminal/run", methods=["POST"])
def run_command():
    data = request.get_json(silent=True) or {}

    try:
        command = validate_command(data.get("command", ""))
    except ValidationError as e:
        return {"error": e.message, "field": e.field}, 400

    context   = data.get("context", "general")[:20]
    challenge = data.get("challenge", "")[:300]
    history   = data.get("history", [])[-5:]

    history_str = "\n".join([f"$ {h['cmd']}\n{h['out']}" for h in history if isinstance(h, dict)])

    prompt = f"""You are a realistic {context} terminal simulator.
Command: {command}
{f"Session history:{chr(10)}{history_str}" if history_str else ""}
{f"Active challenge: {challenge}" if challenge else ""}

Return ONLY valid JSON:
{{
  "output": "realistic terminal output",
  "success": true,
  "challenge_completed": false,
  "hint": "",
  "context_update": ""
}}"""

    try:
        raw    = ask_ai(prompt, json_mode=True)
        clean  = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)
        logger.info(f"[terminal] ctx={context} cmd={command[:30]}")
        return {
            "output":              result.get("output", ""),
            "challenge_completed": result.get("challenge_completed", False),
            "hint":                result.get("hint", ""),
            "success":             result.get("success", True)
        }
    except json.JSONDecodeError:
        logger.warning(f"[terminal] JSON parse failed for cmd={command[:30]}")
        return {"output": raw[:500] if raw else "Terminal error", "success": False, "challenge_completed": False, "hint": ""}, 200
    except Exception as e:
        logger.error(f"[terminal] Error: {e}")
        return {"error": "Terminal unavailable. Please try again."}, 503
