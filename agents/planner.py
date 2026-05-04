"""
Planner Agent — decomposes a user query into a list of actionable tasks.

The planner asks the LLM to return a strict JSON array so downstream
agents receive well-structured inputs.  If the model's output is not
valid JSON, the agent retries once with an explicit correction prompt
before raising an error.

Example
-------
>>> tasks = plan("Plan a 3-day trip to Goa under ₹15000")
>>> print(tasks)
[
    {"task_id": 1, "description": "Research affordable accommodation in Goa"},
    {"task_id": 2, "description": "Find budget-friendly transportation options"},
    {"task_id": 3, "description": "Create a day-by-day itinerary with activities"},
    {"task_id": 4, "description": "Estimate meals and miscellaneous expenses"},
    {"task_id": 5, "description": "Compile a total budget breakdown under ₹15000"}
]
"""

import logging
from models.llm import call_llm_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — instructs the model to output ONLY a JSON array
# ---------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """\
You are a task-planning AI.

Given a user query, break it down into a concise list of independent,
actionable sub-tasks that — when researched and combined — fully answer
the query.

Rules:
1. Return ONLY a JSON array of objects.
2. Each object MUST have exactly two keys:
   - "task_id"     (int, starting at 1)
   - "description" (str, one clear sentence)
3. Aim for 3-7 tasks.  Fewer is fine if the query is simple.
4. Do NOT include any text outside the JSON array — no commentary,
   no markdown fences, no preamble.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan(query: str, model: str = None, temperature: float = 0.3) -> list[dict]:
    """
    Break *query* into a list of task dicts.

    Parameters
    ----------
    query : str
        The end-user's natural-language request.

    Returns
    -------
    list[dict]
        Each dict has ``task_id`` (int) and ``description`` (str).

    Raises
    ------
    ValueError
        If the LLM output cannot be parsed into a valid task list
        after a retry attempt.
    """
    logger.info("--- PLANNER STARTING ---")
    logger.info(">>> PLANNER AGENT STARTING for query: %s", query)
    logger.info("Planner received query: %s", query)

    prompt = (
        f"User query:\n\"{query}\"\n\n"
        "Break this into sub-tasks. Respond with a JSON array only."
    )

    try:
        result = call_llm_json(
            prompt=prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=temperature,
            model=model,
        )
    except ValueError:
        # First parse failed — retry with an even more explicit nudge
        logger.warning("First planner attempt failed JSON parse; retrying…")
        prompt += (
            "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Return ONLY a raw JSON array like "
            '[{"task_id": 1, "description": "..."}, ...]. '
            "No markdown, no explanation."
        )
        result = call_llm_json(
            prompt=prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=max(0.1, temperature - 0.2), # reduce temp for retry
            model=model,
        )

    # Validate shape — we expect a list of dicts
    tasks = _validate_tasks(result)
    logger.info(">>> PLANNER RESPONSE: %s", tasks)
    logger.info("Planner produced %d task(s).", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_tasks(data: dict | list) -> list[dict]:
    """
    Ensure the parsed JSON is a list of properly shaped task objects.

    Handles two common model quirks:
    - Returning ``{"tasks": [...]}`` instead of a bare list.
    - Missing or mis-typed ``task_id`` / ``description`` fields.
    """
    # Unwrap if the model returned an object with a "tasks" key
    if isinstance(data, dict):
        for key in ("tasks", "sub_tasks", "subtasks", "task_list"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError(
                f"Expected a JSON array of tasks, got object with keys: "
                f"{list(data.keys())}"
            )

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Planner returned an empty or non-list result.")

    validated: list[dict] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Task #{idx} is not a dict: {item!r}")
        description = item.get("description") or item.get("task") or ""
        if not description:
            raise ValueError(f"Task #{idx} is missing a description.")
        validated.append({
            "task_id": item.get("task_id", idx),
            "description": str(description),
        })

    return validated
