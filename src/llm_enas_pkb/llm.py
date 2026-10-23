"""DeepSeek official API client (OpenAI-compatible chat completions).

The api.deepseek.com chat-completions endpoint is OpenAI-compatible.  The client
is intentionally thin: it posts to the chat-completions endpoint, retries on
transient failures with exponential backoff, and returns the assistant text.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)


class DeepSeekError(RuntimeError):
    """Raised for any unrecoverable API or response-parsing failure."""


def _extract_json_object(text: str) -> dict:
    """Best-effort parse of the first JSON object found in ``text``.

    LLMs often wrap their JSON in markdown fences or prepend prose, so this is
    deliberately lenient: it strips fences, tries a direct ``json.loads``, then
    falls back to scanning for the first balanced ``{ ... }`` block.
    """
    text = text.strip()
    # Strip markdown fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced { ... } block.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise DeepSeekError(f"Could not parse a JSON object from response: {text[:200]!r}")


class DeepSeekClient:
    def __init__(self, config):
        self.config = config
        # Read the key from the configured env var, falling back to the standard
        # DEEPSEEK_API_KEY name for compatibility.
        self.api_key = os.environ.get(config.api_key_env) or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise DeepSeekError(
                f"DeepSeek API key not found. Set the {config.api_key_env} environment "
                "variable (see .env.example)."
            )
        self.url = config.api_base.rstrip("/") + "/chat/completions"

    def _payload(self, messages: list[dict], use_response_format: bool = True) -> dict:
        """Build the OpenAI-compatible chat-completions request body."""
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        # Force JSON output when configured and permitted by the caller.
        if self.config.json_mode and use_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def chat(self, messages: list[dict]) -> str:
        """Send a chat request and return the assistant text content.

        Retries transient failures with exponential backoff (capped at 30s):
        timeouts, connection errors, rate limits (429) and 5xx server errors.  A
        400 that rejects ``response_format`` triggers one immediate retry without
        that parameter; any other non-200 status is a hard error.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._payload(messages)
        dropped_response_format = False

        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = requests.post(
                    self.url, headers=headers, json=payload, timeout=self.config.request_timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                # Some reasoning models may reject json_object mode; drop the
                # parameter and retry immediately.
                if resp.status_code == 400 and self.config.json_mode and not dropped_response_format:
                    dropped_response_format = True
                    payload = self._payload(messages, use_response_format=False)
                    logger.warning("response_format rejected by model; retrying without it")
                    continue
                # Rate limit / server error / overload -> retry.
                if resp.status_code in {429, 500, 502, 503, 504}:
                    last_err = DeepSeekError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                else:
                    raise DeepSeekError(
                        f"DeepSeek API error HTTP {resp.status_code}: {resp.text[:500]}"
                    )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_err = exc

            if attempt < self.config.max_retries:
                # Exponential backoff: 1s, 2s, 4s, ... capped at 30s.
                delay = min(2 ** attempt, 30) * 1.0
                logger.warning("DeepSeek request failed (%s); retrying in %.0fs", last_err, delay)
                time.sleep(delay)

        raise DeepSeekError(f"DeepSeek request failed after retries: {last_err}")

    def chat_json(self, messages: list[dict]) -> dict:
        """Chat and parse the response as a JSON object."""
        return _extract_json_object(self.chat(messages))
