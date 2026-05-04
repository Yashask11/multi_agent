"""
LLM Wrapper — reusable interface for OpenAI chat completions.

Centralises API calls so every agent goes through a single function.
Handles retries, temperature control, and error logging.
"""

import os
import json
import logging
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client singleton — created once, reused across calls
# ---------------------------------------------------------------------------
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """
    Return (and lazily create) the OpenAI-compatible client.

    Supports any OpenAI-compatible provider (Groq, Google Gemini, etc.)
    by setting LLM_BASE_URL in the environment.  Falls back to standard
    OpenAI if no base URL is configured.
    """
    global _client
    if _client is None:
        # Accept either LLM_API_KEY (generic) or OPENAI_API_KEY (legacy)
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "LLM_API_KEY (or OPENAI_API_KEY) is not set. "
                "Copy .env.example to .env and add your key."
            )
        base_url = os.getenv("LLM_BASE_URL")  # None → default OpenAI
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _default_model() -> str:
    """Read the model name from the environment, with a sensible default."""
    return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Core call function
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant.",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model: str | None = None,
    max_retries: int = 2,
) -> str:
    """
    Send a single user message (with an optional system prompt) to the
    OpenAI chat completions endpoint and return the assistant's reply.

    Parameters
    ----------
    prompt : str
        The user message.
    system_prompt : str
        Instructions that set the assistant's behaviour.
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    max_tokens : int
        Upper bound on response length.
    model : str | None
        Override the default model for this call.
    max_retries : int
        Number of retry attempts on transient errors.

    Returns
    -------
    str
        The text content of the assistant's response.

    Raises
    ------
    RuntimeError
        When all retry attempts are exhausted.
    """
    client = _get_client()
    model = model or _default_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("LLM call attempt %d/%d (model=%s, temp=%.2f)...", attempt, max_retries, model, temperature)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30.0, # 30s timeout to prevent hanging forever
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned an empty response.")
            logger.info("LLM responded successfully.")
            return content.strip()

        except (RateLimitError, APIConnectionError) as exc:
            # Transient — worth retrying
            logger.warning("Transient error on attempt %d: %s", attempt, exc)
            last_error = exc

        except APIError as exc:
            # Non-transient API error — abort immediately
            logger.error("OpenAI API error: %s", exc)
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Convenience: call that expects JSON back
# ---------------------------------------------------------------------------

def call_llm_json(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant that responds in JSON.",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    model: str | None = None,
) -> dict | list:
    """
    Call the LLM and parse the result as JSON.

    Applies a lower default temperature for more deterministic output and
    strips markdown code fences that models sometimes wrap around JSON.

    Returns
    -------
    dict | list
        Parsed JSON object.

    Raises
    ------
    ValueError
        If the response cannot be parsed as valid JSON after cleanup.
    """
    raw = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )

    # Models sometimes wrap JSON in ```json ... ``` — strip that.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (with optional language tag)
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed. Raw response:\n%s", raw)
        raise ValueError(
            f"LLM did not return valid JSON. Parse error: {exc}\n"
            f"Raw response:\n{raw}"
        ) from exc
