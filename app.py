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

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from agents.planner import plan
from agents.researcher import research
from agents.executor import execute

app = Flask(__name__)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
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

    def generate():
        start_time = time.time()

        # ── Step 1: Planning ──
        yield _sse("step", {"step": 1, "name": "Planning", "status": "running"})

        try:
            tasks = plan(query)
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
                result = research(desc)
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
            final_answer = execute(query, research_results)
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
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000, threaded=True)
