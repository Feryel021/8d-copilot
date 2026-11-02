"""
llm_client.py
-------------
One small wrapper around the OpenAI Python SDK.

Why the OpenAI SDK?  Groq and Google Gemini both offer an
"OpenAI-compatible" endpoint, so ONE client works with three providers.
Gemini and Groq have free tiers, which is useful for a student project.

Everything that can go wrong with an API call is handled here, so the
rest of the app only has to deal with one exception type: LLMError.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import openai
from openai import OpenAI

# --------------------------------------------------------------------------
# Supported providers (all speak the OpenAI chat-completions format)
# --------------------------------------------------------------------------
PROVIDERS = {
    "Groq (free tier)": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
        "key_name": "GROQ_API_KEY",
        # gpt-oss is a reasoning model: low effort = faster, fewer tokens
        "reasoning_effort": "low",
    },
    "Google Gemini (free tier)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.5-flash",
        "key_name": "GEMINI_API_KEY",
        # Gemini 3 models "think" before answering. Thinking tokens count
        # against max_tokens, so we ask for low thinking effort.
        "reasoning_effort": "low",
    },
    "OpenAI": {
        "base_url": None,  # SDK default
        "default_model": "gpt-4o-mini",
        "key_name": "OPENAI_API_KEY",
    },
}


class LLMError(Exception):
    """A user-friendly error message that the UI can show directly."""


@dataclass
class GenParams:
    """Generation parameters. Each pipeline step uses its own values."""

    temperature: float = 0.3
    max_tokens: int = 800
    top_p: float = 1.0

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


@dataclass
class CallLog:
    """Kept for the 'Prompt log' tab, so the grader can see every call."""

    step: str
    technique: str
    params: dict
    messages: list
    output: str
    seconds: float
    tokens: int | None = None
    finish_reason: str | None = None  # "length" = cut off by max_tokens


@dataclass
class LLMClient:
    provider: str
    api_key: str
    model: str
    timeout: float = 45.0
    max_retries: int = 3
    logs: list = field(default_factory=list)

    def __post_init__(self):
        if not self.api_key or not self.api_key.strip():
            raise LLMError("No API key. Paste your key in the sidebar.")
        cfg = PROVIDERS[self.provider]
        self._reasoning = cfg.get("reasoning_effort")
        # max_retries=0: we do our own retry loop below so we can control
        # the waiting time and show clear messages.
        self._client = OpenAI(
            api_key=self.api_key.strip(),
            base_url=cfg["base_url"],
            timeout=self.timeout,
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        params: GenParams,
        step: str = "",
        technique: str = "",
    ) -> str:
        """Send messages, return the text answer. Retries on temporary errors."""
        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                extra = {"reasoning_effort": self._reasoning} if self._reasoning else {}
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **params.as_dict(),
                    **extra,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                finish = getattr(choice, "finish_reason", None)
                if not text:
                    if finish == "length":
                        raise LLMError("The model used all max_tokens before writing an answer "
                                       "(thinking models need more). Raise max_tokens.")
                    raise LLMError("The model returned an empty answer. Try again or raise max_tokens.")
                tokens = getattr(resp.usage, "total_tokens", None) if resp.usage else None
                self.logs.append(
                    CallLog(step, technique, params.as_dict(), messages, text,
                            round(time.time() - start, 2), tokens, finish)
                )
                return text

            # --- temporary problems: wait and retry -----------------------
            except openai.RateLimitError:
                last_error = "Rate limit reached (too many requests)."
                wait = 20 * attempt  # 20 s, 40 s: free-tier limits reset per minute
            except openai.APITimeoutError:
                last_error = f"The API did not answer within {self.timeout:.0f} s."
                wait = 2
            except openai.APIConnectionError:
                last_error = "Cannot reach the API. Check your internet connection."
                wait = 2
            except openai.InternalServerError:
                last_error = "The provider had a server error."
                wait = 2 ** attempt

            # --- permanent problems: stop immediately ---------------------
            except openai.AuthenticationError:
                raise LLMError("The API key was rejected. Check that it is correct and matches the provider.")
            except openai.PermissionDeniedError:
                raise LLMError("This key has no access to the selected model.")
            except openai.NotFoundError:
                raise LLMError(f"Model '{self.model}' was not found. Check the model name in the sidebar.")
            except openai.BadRequestError as e:
                # Some models do not accept reasoning_effort: retry once without it.
                if self._reasoning and "reasoning" in str(e.message).lower():
                    self._reasoning = None
                    continue
                raise LLMError(f"The request was refused by the API: {e.message}")
            except openai.APIStatusError as e:
                raise LLMError(f"API error {e.status_code}: {e.message}")

            if attempt < self.max_retries:
                time.sleep(wait)

        raise LLMError(f"{last_error} Gave up after {self.max_retries} attempts. Wait a minute and try again.")

    # ------------------------------------------------------------------
    # JSON helper: models sometimes add text or ``` fences around JSON
    # ------------------------------------------------------------------
    def chat_json(self, messages, params, step="", technique="") -> dict | list:
        """Ask for JSON. If parsing fails, ask the model once to repair it."""
        raw = self.chat(messages, params, step, technique)
        try:
            return parse_json(raw)
        except ValueError:
            # If the answer was cut off by max_tokens, a repair call would be
            # cut off too. Tell the user what to change instead.
            if self.logs and self.logs[-1].finish_reason == "length":
                raise LLMError("The answer was cut off by max_tokens before the JSON was complete. "
                               "Raise max_tokens (sidebar override) and try again.")
            repair = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your answer was not valid JSON. "
                 "Return ONLY the corrected JSON, no explanation, no ``` fences."},
            ]
            raw2 = self.chat(repair, GenParams(0.0, params.max_tokens, 1.0),
                             step + " (JSON repair)", technique)
            try:
                return parse_json(raw2)
            except ValueError:
                raise LLMError("The model did not return valid JSON, even after one repair attempt. "
                               "Try a lower temperature or a stronger model.")


def parse_json(text: str):
    """Extract the first JSON object/array from a model answer."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # fall back: take from the first { or [ to the last } or ]
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found")
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        raise ValueError("no JSON found")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(str(e))
