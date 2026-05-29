from flask import Flask
from flask_cors import CORS
from routes.ask import ask_bp
from routes.tutor import tutor_bp
from routes.feedback import feedback_bp
from routes.whiteboard import whiteboard_bp

app = Flask(__name__)

# Lock CORS to your GitHub Pages domain + localhost for testing
# Replace YOUR_GITHUB_USERNAME with your actual GitHub username
CORS(app, origins=[
    "https://maitrilearn.github.io",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5500",
    "null"
])

app.register_blueprint(ask_bp)
app.register_blueprint(tutor_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(whiteboard_bp)

@app.route("/")
def home():
    return {"message": "MaitriLearn Backend Running ✅"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
