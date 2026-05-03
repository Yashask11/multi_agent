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
import math
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from models.llm import call_llm, call_llm_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
RESEARCHER_SYSTEM_PROMPT = """\
You are a research assistant AI with access to advanced tools.
You have been given a task description and the output of a tool that was automatically executed for you.

Given this context, provide a clear, concise, and informative response.

Rules:
1. Synthesize the provided context to answer the task.
2. Be factual and specific — include numbers, names, or examples where possible.
3. Keep your response between 100-300 words.
4. Do NOT include preamble like "Sure!" or "Here's what I found". Jump straight into the answer.
5. Use plain text only (no markdown headings or bullet points).
6. If the context does not contain enough information, state what you know and what is missing.
"""

# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _perform_web_search(query: str) -> str:
    """Perform a web search using DuckDuckGo and return formatted top results."""
    logger.info("Performing web search for: %s", query)
    try:
        with DDGS() as ddgs:
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

def _scrape_website(url: str) -> str:
    """Scrape the text content of a URL."""
    logger.info("Scraping URL: %s", url)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # remove scripts and styles
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
        text = soup.get_text(separator=' ', strip=True)
        return text[:5000] # return first 5000 chars to avoid huge context
    except Exception as exc:
        logger.warning("Scrape failed: %s", exc)
        return f"Scrape failed. Error: {exc}"

def _calculator(expression: str) -> str:
    """Calculate a math expression safely."""
    logger.info("Calculating: %s", expression)
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as exc:
        logger.warning("Calculation failed: %s", exc)
        return f"Calculation failed. Error: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def research(task_description: str, model: str = None, temperature: float = 0.3) -> str:
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

    # 1. Ask LLM to choose a tool
    router_prompt = (
        f"Task: \"{task_description}\"\n\n"
        "You have three tools available:\n"
        "1. search (for general web queries)\n"
        "2. scrape (to read a specific URL, e.g., 'https://example.com')\n"
        "3. calculate (to compute math expressions, e.g., '15000 / 3')\n\n"
        "Choose the best tool and provide the input. Return ONLY a JSON object with 'tool' and 'input' keys. "
        "Example: {\"tool\": \"calculate\", \"input\": \"15000 * 0.18\"}"
    )
    
    try:
        choice = call_llm_json(prompt=router_prompt, system_prompt="You are a routing agent. Respond ONLY in JSON.", model=model, temperature=0.1)
        tool_name = choice.get("tool", "search").lower()
        tool_input = choice.get("input", task_description)
    except ValueError:
        logger.warning("Router failed to return valid JSON. Defaulting to web search.")
        tool_name = "search"
        tool_input = task_description

    # 2. Execute chosen tool
    if tool_name == "calculate":
        context = _calculator(tool_input)
    elif tool_name == "scrape":
        context = _scrape_website(tool_input)
    else:
        context = _perform_web_search(tool_input)

    # 3. Build the prompt with context
    prompt = (
        f"Research task:\n\"{task_description}\"\n\n"
        f"Tool Used: {tool_name}\n"
        f"Tool Context Output:\n{context}\n\n"
        "Provide a concise, informative response based on the task and context."
    )

    # 4. Call LLM
    result = call_llm(
        prompt=prompt,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        temperature=temperature,
        model=model,
        max_tokens=1024,
    )

    logger.info(
        "Researcher completed task (response length: %d chars).", len(result)
    )
    return result
