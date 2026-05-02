"""
Executor Agent — synthesises all research findings into a final answer.

Receives the original query plus every research output, then produces a
single well-structured response with headings, bullet points, and a
clear conclusion.

Example
-------
>>> final = execute(
...     query="Plan a 3-day trip to Goa under ₹15000",
...     research_results=[
...         {"task_id": 1, "description": "...", "result": "..."},
...         {"task_id": 2, "description": "...", "result": "..."},
...     ],
... )
>>> print(final)   # formatted markdown-style report
"""

import logging
from models.llm import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
EXECUTOR_SYSTEM_PROMPT = """\
You are a report-writing AI that synthesises research findings into a
polished, structured final answer.

Rules:
1. Address the original user query directly.
2. Organise your answer with clear headings (##) and bullet points.
3. Include a brief summary / conclusion at the end.
4. Be concise but comprehensive — aim for 300-600 words.
5. Do NOT fabricate information beyond what the research findings provide.
6. Use the research findings as your ONLY source of truth.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute(query: str, research_results: list[dict]) -> str:
    """
    Combine research findings into a final, structured answer.

    Parameters
    ----------
    query : str
        The original user query.
    research_results : list[dict]
        Each dict has keys ``task_id``, ``description``, and ``result``.

    Returns
    -------
    str
        A formatted report answering the user's query.

    Raises
    ------
    RuntimeError
        Propagated from the LLM wrapper on unrecoverable API errors.
    """
    logger.info("Executor synthesising %d research result(s).", len(research_results))

    # Build a numbered summary of all findings for the prompt
    findings_block = _format_findings(research_results)

    prompt = (
        f"Original user query:\n\"{query}\"\n\n"
        f"Research findings:\n{findings_block}\n\n"
        "Using ONLY the research findings above, produce a well-structured "
        "final answer that fully addresses the user's query. "
        "Use headings (##) and bullet points for clarity."
    )

    result = call_llm(
        prompt=prompt,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        temperature=0.4,   # slightly creative but mostly faithful
        max_tokens=2048,
    )

    logger.info("Executor produced final answer (%d chars).", len(result))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_findings(research_results: list[dict]) -> str:
    """
    Format research results into a numbered text block that the LLM
    can easily reference.
    """
    lines: list[str] = []
    for item in research_results:
        tid = item.get("task_id", "?")
        desc = item.get("description", "N/A")
        result = item.get("result", "No result available.")
        lines.append(
            f"--- Finding #{tid}: {desc} ---\n{result}\n"
        )
    return "\n".join(lines)
