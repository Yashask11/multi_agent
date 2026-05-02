"""
Multi-Agent Orchestrator — the main entry point.

Pipeline:
    1. Planner  → breaks the user query into sub-tasks
    2. Researcher → executes each sub-task independently
    3. Executor  → synthesises all findings into a final answer

Usage:
    # Interactive (prompts for input)
    python main.py

    # One-shot via CLI argument
    python main.py "Plan a 3-day trip to Goa under ₹15000"

Sample Output:
    ═══════════════════════════════════════════════════════
     MULTI-AGENT SYSTEM — Phase 1
    ═══════════════════════════════════════════════════════
    📝 Query: Plan a 3-day trip to Goa under ₹15000

    ──── Step 1: Planning ────
    ✅ Planner produced 5 task(s):
      1. Research affordable accommodation in Goa
      2. Find budget-friendly transportation options
      ...

    ──── Step 2: Researching ────
    🔍 [1/5] Research affordable accommodation in Goa ... done
    🔍 [2/5] Find budget-friendly transportation ... done
    ...

    ──── Step 3: Executing ────
    📊 Synthesising final answer ...

    ═══════════════════════════════════════════════════════
     FINAL ANSWER
    ═══════════════════════════════════════════════════════
    ## 3-Day Goa Trip Under ₹15,000
    ...
"""

import sys
import os
import io
import logging
import time
import concurrent.futures

# Force UTF-8 output on Windows consoles that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

# Load environment variables from .env file BEFORE any agent imports
# so that OPENAI_API_KEY is available when the LLM module initialises.
from dotenv import load_dotenv
load_dotenv()

from agents.planner import plan
from agents.researcher import research
from agents.executor import execute

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

SEPARATOR = "═" * 55
SUB_SEPARATOR = "─" * 40


def _header(title: str) -> None:
    """Print a prominent section header."""
    print(f"\n{SEPARATOR}")
    print(f" {title}")
    print(SEPARATOR)


def _step(number: int, name: str) -> None:
    """Print a step label."""
    print(f"\n{SUB_SEPARATOR}")
    print(f"  Step {number}: {name}")
    print(SUB_SEPARATOR)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(query: str) -> str:
    """
    Run the full multi-agent pipeline for *query* and return the final
    answer as a string.

    Parameters
    ----------
    query : str
        The end-user's natural-language request.

    Returns
    -------
    str
        The executor's final structured answer.
    """
    start_time = time.time()

    _header("MULTI-AGENT SYSTEM — Phase 1")
    print(f"📝 Query: {query}")

    # ── Step 1: Planning ──────────────────────────────────────────────────
    _step(1, "Planning")
    try:
        tasks = plan(query)
    except (ValueError, RuntimeError) as exc:
        print(f"❌ Planner failed: {exc}")
        logger.exception("Planner error")
        return f"Error during planning: {exc}"

    print(f"✅ Planner produced {len(tasks)} task(s):")
    for t in tasks:
        print(f"   {t['task_id']}. {t['description']}")

    # ── Step 2: Research ──────────────────────────────────────────────────
    _step(2, "Researching")
    research_results: list[dict] = []

    def run_research(task, idx):
        desc = task["description"]
        try:
            result = research(desc)
            return {"task": task, "idx": idx, "result": result, "error": None}
        except Exception as exc:
            return {"task": task, "idx": idx, "result": f"[Research failed: {exc}]", "error": exc}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(run_research, task, idx): task
            for idx, task in enumerate(tasks, start=1)
        }
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            task = res["task"]
            idx = res["idx"]
            
            if res["error"]:
                print(f"🔍 [{idx}/{len(tasks)}] {task['description']} ... FAILED ✗ ({res['error']})")
                logger.error("Researcher failed on task %d: %s", task["task_id"], res["error"])
            else:
                print(f"🔍 [{idx}/{len(tasks)}] {task['description']} ... done ✓")
            
            research_results.append({
                "task_id": task["task_id"],
                "description": task["description"],
                "result": res["result"],
            })

    # Sort results by task_id to maintain order for the executor
    research_results.sort(key=lambda x: x["task_id"])

    # ── Step 3: Execution ─────────────────────────────────────────────────
    _step(3, "Executing")
    print("📊 Synthesising final answer ... ", end="", flush=True)

    try:
        final_answer = execute(query, research_results)
        print("done ✓")
    except RuntimeError as exc:
        print(f"FAILED ✗ ({exc})")
        logger.exception("Executor error")
        return f"Error during execution: {exc}"

    # ── Output ────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    _header("FINAL ANSWER")
    print(final_answer)
    print(f"\n⏱  Completed in {elapsed:.1f}s")

    return final_answer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI args or prompt interactively, then run the pipeline."""
    if len(sys.argv) > 1:
        # One-shot mode: take the query from the command line
        query = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("Multi-Agent System — Phase 1")
        print("Type your query below (or 'quit' to exit).\n")
        query = input(">>> ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            return

    run(query)


if __name__ == "__main__":
    main()
