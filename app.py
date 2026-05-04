"""
Web interface for the Multi-Agent System.

Provides a beautiful single-page UI where users can submit queries
and watch the planner → researcher → executor pipeline run with
real-time step updates via Server-Sent Events (SSE).

Usage:
    python app.py
    # Open http://localhost:5000 in your browser
"""

import json
import time
import logging
import os
import sys
import io
import concurrent.futures
import queue
from datetime import datetime

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, redirect, url_for
from flask_compress import Compress
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from pyngrok import ngrok
from agents.planner import plan
from agents.researcher import research
from agents.executor import execute

app = Flask(__name__)
Compress(app)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-1234")
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    preferred_model = db.Column(db.String(100), default="llama-3.3-70b-versatile")
    temperature = db.Column(db.Float, default=0.7)
    histories = db.relationship('QueryHistory', backref='user', lazy=True)

class QueryHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@app.route("/")
def index():
    """Serve the landing page."""
    return render_template("landing.html", current_user=current_user)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Invalid username or password"}), 401
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        if User.query.filter_by(username=username).first():
            return jsonify({"status": "error", "message": "Username already exists"}), 400
        new_user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return jsonify({"status": "success"})
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    """Serve the user history dashboard."""
    history = QueryHistory.query.filter_by(user_id=current_user.id).order_by(QueryHistory.created_at.desc()).all()
    return render_template("dashboard.html", current_user=current_user, history=history)

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    """Serve and update user settings."""
    if request.method == "POST":
        data = request.get_json()
        current_user.preferred_model = data.get("model", current_user.preferred_model)
        current_user.temperature = float(data.get("temperature", current_user.temperature))
        db.session.commit()
        return jsonify({"status": "success"})
    return render_template("settings.html", current_user=current_user)

@app.route("/app")
@login_required
def application():
    """Serve the main UI."""
    return render_template("app.html", current_user=current_user)


@app.route("/api/run", methods=["POST"])
@login_required
def run_pipeline():
    """
    Stream the multi-agent pipeline as Server-Sent Events.

    Each event has a type (step, task, research, result, error)
    and a JSON data payload so the frontend can update in real-time.
    """
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required."}), 400

    model = current_user.preferred_model
    temperature = current_user.temperature
    user_id = current_user.id

    def generate():
        start_time = time.time()

        # ── Step 1: Planning ──
        yield _sse("step", {"step": 1, "name": "Planning", "status": "running"})

        try:
            tasks = plan(query, model=model, temperature=temperature)
        except (ValueError, RuntimeError) as exc:
            logger.exception("Planner error")
            yield _sse("error", {"message": f"Planner failed: {exc}"})
            return

        yield _sse("step", {"step": 1, "name": "Planning", "status": "done"})
        yield _sse("tasks", {
            "count": len(tasks),
            "tasks": tasks,
        })

        # ── Step 2: Researching ──
        yield _sse("step", {"step": 2, "name": "Researching", "status": "running"})
        research_results = []

        # Announce all tasks are running
        for idx, task in enumerate(tasks, start=1):
            yield _sse("research", {
                "task_id": task["task_id"],
                "description": task["description"],
                "index": idx,
                "total": len(tasks),
                "status": "running",
            })

        q = queue.Queue()

        def worker(task, idx):
            desc = task["description"]
            try:
                result = research(desc, model=model, temperature=max(0.1, temperature - 0.2))
                q.put(("done", task, idx, result))
            except Exception as exc:
                q.put(("error", task, idx, exc))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for idx, task in enumerate(tasks, start=1):
                executor.submit(worker, task, idx)
            
            # Wait for all to finish and yield
            for _ in range(len(tasks)):
                status, task, idx, result_or_exc = q.get()
                if status == "done":
                    research_results.append({
                        "task_id": task["task_id"],
                        "description": task["description"],
                        "result": result_or_exc,
                    })
                    yield _sse("research", {
                        "task_id": task["task_id"],
                        "description": task["description"],
                        "index": idx,
                        "total": len(tasks),
                        "status": "done",
                        "result": result_or_exc,
                    })
                else:
                    logger.error("Researcher failed on task %d: %s", task["task_id"], result_or_exc)
                    research_results.append({
                        "task_id": task["task_id"],
                        "description": task["description"],
                        "result": f"[Research failed: {result_or_exc}]",
                    })
                    yield _sse("research", {
                        "task_id": task["task_id"],
                        "description": task["description"],
                        "index": idx,
                        "total": len(tasks),
                        "status": "failed",
                    })

        # Sort results by task_id to maintain order for the executor
        research_results.sort(key=lambda x: x["task_id"])

        yield _sse("step", {"step": 2, "name": "Researching", "status": "done"})

        # ── Step 3: Executing ──
        yield _sse("step", {"step": 3, "name": "Synthesising", "status": "running"})

        try:
            final_answer = execute(query, research_results, model=model, temperature=temperature)
            history = QueryHistory(user_id=user_id, query=query, result=final_answer)
            db.session.add(history)
            db.session.commit()
        except RuntimeError as exc:
            logger.exception("Executor error")
            yield _sse("error", {"message": f"Executor failed: {exc}"})
            return

        elapsed = time.time() - start_time
        yield _sse("step", {"step": 3, "name": "Synthesising", "status": "done"})
        yield _sse("result", {
            "answer": final_answer,
            "elapsed": round(elapsed, 1),
        })
        yield _sse("done", {"elapsed": round(elapsed, 1)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


if __name__ == "__main__":
    print("\n  Multi-Agent System Web UI")
    ngrok_token = os.getenv("NGROK_AUTHTOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        try:
            print("  Starting Ngrok tunnel...")
            public_url = ngrok.connect(5000).public_url
            print(f"  Public URL: {public_url}")
            print("  Open this link on any device!\n")
        except Exception as e:
            print(f"  Failed to start ngrok: {e}")
    else:
        print("  (Ngrok authtoken not found in .env. Running locally only. To share publicly, add NGROK_AUTHTOKEN to .env)\n")
        print("  Open http://localhost:5000 in your browser\n")
        
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
