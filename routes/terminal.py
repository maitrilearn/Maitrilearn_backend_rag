import logging
import json
import re
from flask import Blueprint, request
from services.groq_service import ask_ai
from utils.validator import validate_command, ValidationError

terminal_bp = Blueprint("terminal", __name__)
logger = logging.getLogger("maitrilearn")

# ── Static commands — bypass AI, instant response, zero rate limit ────────────
LS_LA_OUTPUT = (
    "total 32\n"
    "drwxr-xr-x 5 student student 4096 Jul  7 10:00 .\n"
    "drwxr-xr-x 3 root    root    4096 Jul  7 09:00 ..\n"
    "-rw-r--r-- 1 student student  220 Jul  7 09:00 .bash_logout\n"
    "-rw-r--r-- 1 student student 3526 Jul  7 09:00 .bashrc\n"
    "drwxr-xr-x 2 student student 4096 Jul  7 10:00 projects\n"
    "-rw-r--r-- 1 student student   48 Jul  7 10:00 notes.txt"
)

STATIC_OUTPUTS = {
    "pwd":                       "/home/student",
    "ls":                        "Desktop  Documents  Downloads  notes.txt  projects",
    "ls -la":                    LS_LA_OUTPUT,
    "ls -l":                     LS_LA_OUTPUT,
    "ls -a":                     ".  ..  .bash_logout  .bashrc  Desktop  Documents  projects  notes.txt",
    "echo hello":                "hello",
    "echo hello world":          "hello world",
    "whoami":                    "student",
    "uname":                     "Linux",
    "uname -a":                  "Linux maitrilearn 5.15.0-1034-aws #38-Ubuntu SMP x86_64 GNU/Linux",
    "hostname":                  "maitrilearn-terminal",
    "cat /etc/hostname":         "maitrilearn-terminal",
    "date":                      "Tue Jul  7 10:00:00 UTC 2026",
    "clear":                     "",
    "help":                      "Available: ls, cd, pwd, cat, mkdir, rm, chmod, ps, grep, docker, git, kubectl",
    "git --version":             "git version 2.39.2",
    "git version":               "git version 2.39.2",
    "git init":                  "Initialized empty Git repository in /home/student/projects/.git/",
    "git status":                "On branch main\nnothing to commit, working tree clean",
    "git log":                   "commit a1b2c3d (HEAD -> main)\nAuthor: Student <student@example.com>\nDate:   Tue Jul  7 10:00:00 2026\n\n    initial commit",
    "docker --version":          "Docker version 24.0.5, build ced0996",
    "docker version":            "Client: Docker Engine\n Version: 24.0.5\nServer: Docker Engine\n Engine Version: 24.0.5",
    "docker info":               "Containers: 2\n Running: 1\n Paused: 0\n Stopped: 1\nImages: 5\nServer Version: 24.0.5",
    "docker images":             "REPOSITORY   TAG       IMAGE ID       CREATED        SIZE\nnginx        latest    89da1fb6dcb9   2 weeks ago    187MB\nhello-world  latest    9c7a54a9a43c   3 months ago   13.3kB",
    "docker ps":                 "CONTAINER ID   IMAGE     COMMAND                  CREATED        STATUS        PORTS                  NAMES\na1b2c3d4e5f6   nginx     \"/docker-entrypoint.…\"   2 hours ago    Up 2 hours    0.0.0.0:8080->80/tcp   web",
    "docker ps -a":              "CONTAINER ID   IMAGE         COMMAND                  CREATED        STATUS                    PORTS                  NAMES\na1b2c3d4e5f6   nginx         \"/docker-entrypoint.…\"   2 hours ago    Up 2 hours                0.0.0.0:8080->80/tcp   web\nb2c3d4e5f6a7   hello-world   \"/hello\"                 3 hours ago    Exited (0) 3 hours ago                           hello",
    "docker pull nginx":          "Using default tag: latest\nlatest: Pulling from library/nginx\nDigest: sha256:abc123...\nStatus: Image is up to date for nginx:latest",
    "docker run hello-world":     "Hello from Docker!\nThis message shows that your installation appears to be working correctly.",
    "kubectl version --client":   "Client Version: version.Info{Major:\"1\", Minor:\"27\", GitVersion:\"v1.27.3\"}",
    "kubectl version":            "Client Version: v1.27.3\nServer Version: v1.27.3",
    "kubectl get nodes":          "NAME           STATUS   ROLES           AGE   VERSION\nmaitrilearn    Ready    control-plane   1d    v1.27.3",
    "kubectl get pods":           "No resources found in default namespace.",
    "kubectl get pods --all-namespaces": (
        "NAMESPACE     NAME                              READY   STATUS    RESTARTS   AGE\n"
        "kube-system   coredns-5d78c9869d-abc12          1/1     Running   0          1d\n"
        "kube-system   etcd-maitrilearn                  1/1     Running   0          1d\n"
        "kube-system   kube-apiserver-maitrilearn         1/1     Running   0          1d"
    ),
    "kubectl get services":       "NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE\nkubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   1d",
    "kubectl get namespaces":     "NAME              STATUS   AGE\ndefault           Active   1d\nkube-node-lease   Active   1d\nkube-public       Active   1d\nkube-system       Active   1d",
    "python3 --version":          "Python 3.11.4",
    "python --version":           "Python 3.11.4",
    "node --version":             "v18.17.0",
    "pip --version":              "pip 23.2.1 from /usr/lib/python3/dist-packages/pip (python 3.11)",
    "cat /etc/os-release":        "NAME=\"Ubuntu\"\nVERSION=\"22.04.2 LTS (Jammy Jellyfish)\"\nID=ubuntu",
    "ps aux":                     "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\nstudent      1  0.0  0.1  18376  1024 pts/0    Ss   10:00   0:00 bash\nstudent     42  0.0  0.0  34524   768 pts/0    R+   10:05   0:00 ps aux",
    "top":                        "Tasks:   2 total,   1 running\n%Cpu(s):  5.0 us,  2.0 sy\nMiB Mem :   1024.0 total,    512.0 free\n\nPID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND\n  1 student   20   0   18376   1024    896 S   0.0   0.1   0:00.05 bash",
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

    # ── Static shortcut — instant, no AI, no rate limit ──────────────────────
    cmd_clean = command.strip()
    cmd_lower = cmd_clean.lower()

    if cmd_lower in STATIC_OUTPUTS:
        output  = STATIC_OUTPUTS[cmd_lower]
        ch_done = bool(challenge and cmd_lower in challenge.lower())
        logger.info(f"[terminal] static cmd={cmd_clean[:40]}")
        return {
            "output":              output,
            "challenge_completed": ch_done,
            "hint":                "",
            "success":             True
        }

    # ── AI-powered for complex commands ───────────────────────────────────────
    history_str = "\n".join([
        f"$ {h['cmd']}\n{h['out']}"
        for h in history if isinstance(h, dict)
    ])

    prompt = f"""You are a realistic {context} terminal simulator.
Command: {cmd_clean}
{f"Session history:{chr(10)}{history_str}" if history_str else ""}
{f"Active challenge: {challenge}" if challenge else ""}

Return ONLY valid JSON with no markdown:
{{
  "output": "realistic terminal output here",
  "success": true,
  "challenge_completed": false,
  "hint": ""
}}"""

    try:
        raw   = ask_ai(prompt, json_mode=False, route="terminal")
        clean = re.sub(r"```json|```", "", raw).strip()

        start = clean.find("{")
        end   = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start:end+1]

        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            return {
                "output":              raw[:800] if raw else "Command executed.",
                "challenge_completed": False,
                "hint":                "",
                "success":             True
            }

        logger.info(f"[terminal] AI cmd={cmd_clean[:40]}")
        return {
            "output":              result.get("output", ""),
            "challenge_completed": result.get("challenge_completed", False),
            "hint":                result.get("hint", ""),
            "success":             result.get("success", True)
        }
    except Exception as e:
        logger.error(f"[terminal] Error: {e}")
        return {"error": "Terminal unavailable. Please try again."}, 503


# ── CHALLENGES DATA ───────────────────────────────────────────────────────────
CHALLENGES = {
    "linux": [
        {"id":"linux_1","day":1,"title":"List Files","description":"List all files including hidden ones","hint":"Try ls with -la flags","solution":"ls -la","context":"linux","difficulty":"easy"},
        {"id":"linux_2","day":1,"title":"Create Directory","description":"Create a directory called myproject in /tmp","hint":"mkdir /tmp/myproject","solution":"mkdir /tmp/myproject","context":"linux","difficulty":"easy"},
        {"id":"linux_3","day":2,"title":"File Permissions","description":"Give owner rwx, group rx, others no permissions on file.sh","hint":"chmod 750 file.sh","solution":"chmod 750 file.sh","context":"linux","difficulty":"medium"},
        {"id":"linux_4","day":3,"title":"Create Soft Link","description":"Create a symbolic link named 'latest' pointing to /var/log/app.log","hint":"ln -s source destination","solution":"ln -s /var/log/app.log latest","context":"linux","difficulty":"medium"},
        {"id":"linux_5","day":5,"title":"Find Running Processes","description":"Show all running processes with PID, CPU and memory","hint":"ps aux","solution":"ps aux","context":"linux","difficulty":"easy"},
        {"id":"linux_6","day":6,"title":"Search in Files","description":"Find all lines containing ERROR in /var/log/app.log","hint":"grep ERROR /var/log/app.log","solution":"grep ERROR /var/log/app.log","context":"linux","difficulty":"easy"},
    ],
    "git": [
        {"id":"git_1","day":1,"title":"Initialize Repo","description":"Initialize a new git repository","hint":"git init","solution":"git init","context":"git","difficulty":"easy"},
        {"id":"git_2","day":2,"title":"Stage and Commit","description":"Stage all changes and commit with message 'initial commit'","hint":"git add . then git commit -m","solution":"git add .","context":"git","difficulty":"easy"},
        {"id":"git_3","day":3,"title":"Create Branch","description":"Create and switch to a new branch called feature/login","hint":"git checkout -b","solution":"git checkout -b feature/login","context":"git","difficulty":"medium"},
        {"id":"git_4","day":4,"title":"Push to Remote","description":"Push your current branch to origin","hint":"git push origin branch-name","solution":"git push origin feature/login","context":"git","difficulty":"medium"},
    ],
    "docker": [
        {"id":"docker_1","day":1,"title":"Hello World","description":"Run the hello-world Docker container","hint":"docker run hello-world","solution":"docker run hello-world","context":"docker","difficulty":"easy"},
        {"id":"docker_2","day":2,"title":"List Containers","description":"Show all containers including stopped ones","hint":"docker ps -a","solution":"docker ps -a","context":"docker","difficulty":"easy"},
        {"id":"docker_3","day":2,"title":"Pull nginx","description":"Pull the official nginx image from Docker Hub","hint":"docker pull nginx","solution":"docker pull nginx","context":"docker","difficulty":"easy"},
        {"id":"docker_4","day":2,"title":"Run nginx","description":"Run nginx in detached mode mapping port 8080 to 80","hint":"docker run -d -p 8080:80 nginx","solution":"docker run -d -p 8080:80 nginx","context":"docker","difficulty":"medium"},
        {"id":"docker_5","day":3,"title":"Build Image","description":"Build a Docker image tagged as myapp:v1 from current directory","hint":"docker build -t myapp:v1 .","solution":"docker build -t myapp:v1 .","context":"docker","difficulty":"medium"},
        {"id":"docker_6","day":4,"title":"Create Volume","description":"Create a named Docker volume called mydata","hint":"docker volume create mydata","solution":"docker volume create mydata","context":"docker","difficulty":"easy"},
        {"id":"docker_7","day":5,"title":"Start Compose","description":"Start all services in docker-compose.yml in detached mode","hint":"docker-compose up -d","solution":"docker-compose up -d","context":"docker","difficulty":"medium"},
    ],
    "kubernetes": [
        {"id":"k8s_1","day":2,"title":"Get All Pods","description":"List all pods in all namespaces","hint":"kubectl get pods --all-namespaces","solution":"kubectl get pods --all-namespaces","context":"kubernetes","difficulty":"easy"},
        {"id":"k8s_2","day":3,"title":"Create Deployment","description":"Create deployment webapp using nginx with 3 replicas","hint":"kubectl create deployment with --image and --replicas","solution":"kubectl create deployment webapp --image=nginx --replicas=3","context":"kubernetes","difficulty":"medium"},
        {"id":"k8s_3","day":4,"title":"Expose Deployment","description":"Expose webapp as NodePort service on port 80","hint":"kubectl expose deployment --type=NodePort","solution":"kubectl expose deployment webapp --type=NodePort --port=80","context":"kubernetes","difficulty":"medium"},
        {"id":"k8s_4","day":3,"title":"Scale Deployment","description":"Scale webapp deployment to 5 replicas","hint":"kubectl scale deployment --replicas=5","solution":"kubectl scale deployment webapp --replicas=5","context":"kubernetes","difficulty":"medium"},
        {"id":"k8s_5","day":5,"title":"Create ConfigMap","description":"Create ConfigMap app-config with APP_ENV=production","hint":"kubectl create configmap --from-literal","solution":"kubectl create configmap app-config --from-literal=APP_ENV=production","context":"kubernetes","difficulty":"hard"},
    ]
}
