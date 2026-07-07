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


# Simple commands that don't need AI — instant response, no rate limit
STATIC_OUTPUTS = {
    "pwd":                  "/home/student",
    "ls":                   "Desktop  Documents  Downloads  notes.txt  projects",
    "ls -la":               "total 32
drwxr-xr-x 5 student student 4096 Jul  7 10:00 .
drwxr-xr-x 3 root    root    4096 Jul  7 09:00 ..
-rw-r--r-- 1 student student  220 Jul  7 09:00 .bash_logout
-rw-r--r-- 1 student student 3526 Jul  7 09:00 .bashrc
drwxr-xr-x 2 student student 4096 Jul  7 10:00 projects
-rw-r--r-- 1 student student   48 Jul  7 10:00 notes.txt",
    "echo hello":           "hello",
    "echo hello world":     "hello world",
    "whoami":               "student",
    "uname":                "Linux",
    "uname -a":             "Linux maitrilearn 5.15.0-1034-aws #38-Ubuntu SMP Mon May  1 19:45:39 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux",
    "hostname":             "maitrilearn-terminal",
    "cat /etc/hostname":    "maitrilearn-terminal",
    "date":                 "Tue Jul  7 10:00:00 UTC 2026",
    "clear":                "",
    "help":                 "Available: ls, cd, pwd, cat, mkdir, rm, chmod, ps, grep, docker, git, kubectl",
    "git --version":        "git version 2.39.2",
    "docker --version":     "Docker version 24.0.5, build ced0996",
    "kubectl version --client": "Client Version: version.Info{Major:"1", Minor:"27", GitVersion:"v1.27.3"}",
    "python3 --version":    "Python 3.11.4",
    "python --version":     "Python 3.11.4",
    "node --version":       "v18.17.0",
    "pip --version":        "pip 23.2.1 from /usr/lib/python3/dist-packages/pip (python 3.11)",
}


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

    # ── Static shortcut — bypass AI for simple commands ──────────────────────
    cmd_lower = command.strip().lower()
    if cmd_lower in STATIC_OUTPUTS:
        output   = STATIC_OUTPUTS[cmd_lower]
        ch_done  = challenge and any(s in cmd_lower for s in challenge.lower().split()[:3])
        logger.info(f"[terminal] static cmd={command[:30]}")
        return {
            "output":              output,
            "challenge_completed": ch_done,
            "hint":                "",
            "success":             True
        }

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
        # BUG FIX: json_mode=False avoids Groq 400 on complex commands like docker ps
        raw   = ask_ai(prompt, json_mode=False)
        clean = re.sub(r"```json|```", "", raw).strip()

        # Find JSON object in response
        start = clean.find("{")
        end   = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start:end+1]

        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            # If JSON parse still fails, return raw output as terminal text
            logger.warning(f"[terminal] JSON parse failed — returning raw for cmd={command[:30]}")
            return {
                "output":              raw[:800] if raw else "Command executed.",
                "challenge_completed": False,
                "hint":                "",
                "success":             True
            }

        logger.info(f"[terminal] ctx={context} cmd={command[:30]}")
        return {
            "output":              result.get("output", ""),
            "challenge_completed": result.get("challenge_completed", False),
            "hint":                result.get("hint", ""),
            "success":             result.get("success", True)
        }
    except Exception as e:
        logger.error(f"[terminal] Error: {e}")
        return {"error": "Terminal unavailable. Please try again."}, 503
