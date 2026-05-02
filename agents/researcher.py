"""
Research Agent — executes a single task and returns a concise finding.

Each call is independent and stateless.  The researcher receives one
task description, queries the LLM, and returns a focused paragraph.

Example
-------
>>> result = research("Research affordable accommodation in Goa")
>>> print(result)
"Budget stays in Goa range from ₹500–₹1500/night. Popular options
include hostels in Anjuna, guesthouses in Palolem, and homestays on
Booking.com / Hostelworld.  A 3-night stay can cost as little as ₹2000
in a dorm bed or ₹4500 for a private room."
"""

import logging
from duckduckgo_search import DDGS
from models.llm import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
RESEARCHER_SYSTEM_PROMPT = """\
You are a research assistant AI with access to live web search results.

Given a specific task and some web search context, provide a clear, concise,
and informative response.

Rules:
1. Synthesize the provided web search context to answer the task.
2. Be factual and specific — include numbers, names, or examples where possible.
3. Keep your response between 100-300 words.
4. Do NOT include preamble like "Sure!" or "Here's what I found". Jump straight into the answer.
5. Use plain text only (no markdown headings or bullet points).
6. If the web context does not contain enough information, state what you know and what is missing.
"""

# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _perform_web_search(query: str) -> str:
    """
    Perform a web search using DuckDuckGo and return formatted top results.
    """
    logger.info("Performing web search for: %s", query)
    try:
        with DDGS() as ddgs:
            # We take top 3 results for context
            results = list(ddgs.text(query, max_results=3))
            
            if not results:
                return "No search results found."
                
            formatted = []
            for idx, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                snippet = r.get("body", "")
                formatted.append(f"[{idx}] {title}\n{snippet}")
                
            return "\n\n".join(formatted)
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return f"Web search failed or is unavailable. Error: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def research(task_description: str) -> str:
    """
    Research a single task using live web search and return a concise finding.

    Parameters
    ----------
    task_description : str
        A one-sentence description of the sub-task to research.

    Returns
    -------
    str
        A paragraph-length research finding.

    Raises
    ------
    RuntimeError
        Propagated from the LLM wrapper on unrecoverable API errors.
    """
    logger.info("Researcher working on: %s", task_description)

    # 1. Gather web context
    # We use the task description directly as the search query.
    # For a more advanced agent, we could have the LLM formulate the query first.
    web_context = _perform_web_search(task_description)

    # 2. Build the prompt with context
    prompt = (
        f"Research task:\n\"{task_description}\"\n\n"
        f"Web Search Context:\n{web_context}\n\n"
        "Provide a concise, informative response based on the task and context."
    )

    # 3. Call LLM
    result = call_llm(
        prompt=prompt,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        temperature=0.4,   # slightly lower temp to stick to search facts
        max_tokens=1024,
    )

    logger.info(
        "Researcher completed task (response length: %d chars).", len(result)
    )
    return result
