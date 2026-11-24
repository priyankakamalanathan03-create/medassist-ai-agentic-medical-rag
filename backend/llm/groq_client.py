"""
llm/groq_client.py -- Groq LLM client for MedAssist-AI

Dual-mode LLM client:
  - Primary: Groq Cloud API (qwen/qwen3.8-27b) -- fast, free, high quality
  - Includes automatic retry on rate-limit (429) with back-off
  - Strips <think>...</think> blocks from Qwen reasoning models
"""
import re
import time
import httpx
from config import GROQ_API_KEY, GROQ_MODEL

# Qwen thinking-mode pattern: <think>...</think>
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_RETRY_DELAY_RE = re.compile(
    r"(?:(?P<hours>\d+(?:\.\d+)?)\s*h)?\s*"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)\s*m)?\s*"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)\s*s)?",
    re.IGNORECASE,
)

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds base delay for rate limits


class GroqRateLimitError(RuntimeError):
    """Raised when Groq rejects requests due to a rate or token limit."""


class GroqLLMClient:
    """Wrapper around Groq API providing generate() and generate_stream()."""

    def __init__(self):
        self.model  = GROQ_MODEL
        self.client = None
        self._api_key = GROQ_API_KEY
        self._last_call_time = 0  # Track calls for rate limiting
        self._rate_limited_until = 0.0
        self._daily_quota_exhausted = False

        if self._api_key and self._api_key != "your_groq_api_key_here":
            from groq import Groq

            http_client = httpx.Client(timeout=45.0)
            self.client = Groq(
                api_key=self._api_key,
                http_client=http_client,
                max_retries=0,
            )
            print("[LLM] Groq client ready -- model:", self.model)
        else:
            print("[LLM] WARNING: GROQ_API_KEY not set. Chat will not work.")
            print("[LLM] Get a FREE key at https://console.groq.com")

    def _check_client(self):
        if self.client is None:
            raise EnvironmentError(
                "GROQ_API_KEY not set. "
                "Get a FREE key at https://console.groq.com and add it to backend/.env"
            )

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks from Qwen model output."""
        return _THINK_RE.sub("", text).strip()

    def _wait_for_rate_limit(self):
        """Ensure minimum gap between API calls to stay under OTPM limit."""
        elapsed = time.time() - self._last_call_time
        min_gap = 0.5
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

    def _raise_if_rate_limited(self):
        remaining = self._rate_limited_until - time.monotonic()
        if remaining <= 0:
            self._rate_limited_until = 0.0
            self._daily_quota_exhausted = False
            return
        raise GroqRateLimitError(
            self._format_rate_limit_message(
                remaining,
                daily=self._daily_quota_exhausted,
            )
        )

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        message = str(error)
        retry_match = re.search(
            r"try again in\s+((?:\d+(?:\.\d+)?\s*h)?\s*"
            r"(?:\d+(?:\.\d+)?\s*m)?\s*"
            r"(?:\d+(?:\.\d+)?\s*s)?)",
            message,
            re.IGNORECASE,
        )
        if retry_match:
            duration = retry_match.group(1)
            parts = _RETRY_DELAY_RE.fullmatch(duration)
            if parts and any(parts.groupdict().values()):
                return (
                    float(parts.group("hours") or 0) * 3600
                    + float(parts.group("minutes") or 0) * 60
                    + float(parts.group("seconds") or 0)
                )

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        try:
            return max(0.0, float(headers.get("retry-after")))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_rate_limit_message(remaining: float, *, daily: bool = False) -> str:
        seconds = max(1, round(remaining))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        wait = " ".join(
            part for part in (
                f"{hours}h" if hours else "",
                f"{minutes}m" if minutes else "",
                f"{seconds}s" if seconds or not (hours or minutes) else "",
            ) if part
        )
        if daily:
            return (
                "Groq's daily token quota for this model is exhausted. "
                f"Try again in about {wait}, or configure a model with available quota."
            )
        return (
            "Groq rate limit reached. "
            f"Try again in about {wait} or configure a model with available quota."
        )

    def _record_rate_limit(self, error: Exception) -> GroqRateLimitError:
        error_text = str(error)
        daily = bool(re.search(
            r"tokens per day|\bTPD\b|daily token",
            error_text,
            re.IGNORECASE,
        ))
        retry_after = self._retry_after_seconds(error)
        if retry_after is not None and retry_after > 0:
            self._daily_quota_exhausted = daily
            self._rate_limited_until = max(
                self._rate_limited_until,
                time.monotonic() + retry_after,
            )
            return GroqRateLimitError(
                self._format_rate_limit_message(retry_after, daily=daily)
            )
        if daily:
            return GroqRateLimitError(
                "Groq's daily token quota for this model is exhausted. "
                "Try again after the provider's quota reset or configure a model "
                "with available quota."
            )
        return GroqRateLimitError(
            "Groq rate limit reached. Please wait before trying again or configure "
            "a model with available quota."
        )

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        return (
            getattr(error, "status_code", None) == 429
            or "429" in str(error)
            or "rate_limit" in str(error).lower()
            or "rate limit" in str(error).lower()
        )

    def generate(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 768,
    ) -> str:
        """Send a chat completion request with retry on rate-limit.

        Missing credentials and provider failures are raised explicitly; no
        fabricated offline answer is returned.
        """
        self._check_client()
        self._raise_if_rate_limited()
        self._wait_for_rate_limit()

        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                self._last_call_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Extract raw text from the model response
                raw = response.choices[0].message.content
                cleaned = self._strip_think(raw)
                if not cleaned:
                    cleaned = raw.strip()
                if not cleaned:
                    cleaned = "[No response from model]"
                return cleaned
            except Exception as e:
                last_error = e
                err_str = str(e)
                if self._is_rate_limit_error(e):
                    raise self._record_rate_limit(e) from e
                if "Connection" in err_str:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"[LLM] Connection failed, retrying in {wait_time:.0f}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue
                raise

        raise last_error  # all retries exhausted

    def generate_stream(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 768,
    ):
        """Stream tokens one-by-one. Yields str chunks."""
        self._check_client()
        self._raise_if_rate_limited()
        self._wait_for_rate_limit()

        last_error = None

        for attempt in range(MAX_RETRIES):
            yielded_content = False
            try:
                self._last_call_time = time.time()
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                # Buffer to detect and skip <think>...</think> blocks
                buffer = ""
                in_think = False

                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        buffer += delta
                        if "<think>" in buffer and not in_think:
                            in_think = True
                        if in_think:
                            if "</think>" in buffer:
                                cleaned = _THINK_RE.sub("", buffer)
                                if cleaned:
                                    yielded_content = True
                                    yield cleaned
                                buffer = ""
                                in_think = False
                        else:
                            yielded_content = True
                            yield buffer
                            buffer = ""

                if buffer and not in_think:
                    yielded_content = True
                    yield buffer
                return

            except Exception as e:
                if yielded_content:
                    raise
                last_error = e
                err_str = str(e)
                if self._is_rate_limit_error(e):
                    raise self._record_rate_limit(e) from e
                if "Connection" in err_str:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"[LLM] Connection failed (stream), retrying in {wait_time:.0f}s")
                    time.sleep(wait_time)
                    continue
                raise

        raise last_error

    def get_model_name(self) -> str:
        return self.model

    def is_configured(self) -> bool:
        return self.client is not None

    def get_rate_limit_status(self) -> dict:
        remaining = max(0.0, self._rate_limited_until - time.monotonic())
        return {
            "limited": remaining > 0,
            "retry_after_seconds": round(remaining) if remaining > 0 else None,
        }


# Singleton
llm_client = GroqLLMClient()
