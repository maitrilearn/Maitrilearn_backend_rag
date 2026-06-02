from flask import Flask
from flask_cors import CORS
from routes.ask import ask_bp
from routes.tutor import tutor_bp
from routes.feedback import feedback_bp
from routes.whiteboard import whiteboard_bp
from routes.rag import rag_bp

app = Flask(__name__)

CORS(app, origins=[
    "https://maitrilearn.github.io",
    "https://maitrilearn.com",
    "https://www.maitrilearn.com"
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5500",
    "http://localhost:9000",
    "null"
])

app.register_blueprint(ask_bp)
app.register_blueprint(tutor_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(whiteboard_bp)
app.register_blueprint(rag_bp)

@app.route("/")
def home():
    return {"message": "MaitriLearn Backend Running ✅ — RAG enabled"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
