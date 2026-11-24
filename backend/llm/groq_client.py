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
from openai import OpenAI
from config import GROQ_API_KEY, GROQ_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

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
OLLAMA_NUM_CTX = 4096


class GroqRateLimitError(RuntimeError):
    """Raised when Groq rejects requests due to a rate or token limit."""


class GroqLLMClient:
    """Wrapper around Groq API providing generate() and generate_stream()."""

    def __init__(self):
        self.model  = GROQ_MODEL
        self.client = None
        self._api_key = GROQ_API_KEY
        self.local_client = OpenAI(
            api_key="local-ollama",
            base_url=OLLAMA_BASE_URL,
            timeout=45.0,
            max_retries=0,
        )
        self.local_model = OLLAMA_MODEL
        self._active_provider = "groq"
        self._active_model = self.model
        self._last_call_time = 0  # Track calls for rate limiting
        self._rate_limited_until = 0.0
        self._daily_quota_exhausted = False
        self._groq_unavailable = False

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
            self._active_provider = "ollama"
            self._active_model = self.local_model
            print(
                "[LLM] GROQ_API_KEY not set; chat will use local Ollama. "
                "Ensure Ollama is running with the configured model."
            )

    def _check_client(self):
        if self.client is None and not self._has_local_fallback():
            raise EnvironmentError(
                "GROQ_API_KEY not set and local Ollama is not available. "
                "Configure Groq or install and start Ollama for local inference."
            )

    def _has_local_fallback(self) -> bool:
        return getattr(self, "local_client", None) is not None

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

    @staticmethod
    def _is_certificate_verification_error(error: Exception) -> bool:
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            message = str(current).lower()
            if "certificate_verify_failed" in message or "certificate verify failed" in message:
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _is_connection_error(error: Exception) -> bool:
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if "connection" in str(current).lower():
                return True
            current = current.__cause__ or current.__context__
        return False

    def _generate_local(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self._active_provider = "ollama"
        self._active_model = self.local_model
        print(f"[LLM] Groq unavailable; using local Ollama model {self.local_model}")
        try:
            response = self.local_client.chat.completions.create(
                model=self.local_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
            )
        except Exception as error:
            raise RuntimeError(
                f"Groq is unavailable and local Ollama model '{self.local_model}' "
                f"could not respond at {OLLAMA_BASE_URL}: {error}"
            ) from error

        raw = response.choices[0].message.content or ""
        answer = self._strip_think(raw) or raw.strip()
        if not answer:
            raise RuntimeError(
                f"Local Ollama model '{self.local_model}' returned an empty response."
            )
        return answer

    def _generate_local_stream(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
    ):
        self._active_provider = "ollama"
        self._active_model = self.local_model
        print(f"[LLM] Groq unavailable; using local Ollama model {self.local_model}")
        yielded_content = False
        try:
            stream = self.local_client.chat.completions.create(
                model=self.local_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yielded_content = True
                    yield delta
        except Exception as error:
            raise RuntimeError(
                f"Groq is unavailable and local Ollama model '{self.local_model}' "
                f"could not respond at {OLLAMA_BASE_URL}: {error}"
            ) from error

        if not yielded_content:
            raise RuntimeError(
                f"Local Ollama model '{self.local_model}' returned an empty response."
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
        if self.client is None or getattr(self, "_groq_unavailable", False):
            return self._generate_local(messages, temperature, max_tokens)
        try:
            self._raise_if_rate_limited()
        except GroqRateLimitError:
            if self._has_local_fallback():
                return self._generate_local(messages, temperature, max_tokens)
            raise
        self._wait_for_rate_limit()
        self._active_provider = "groq"
        self._active_model = self.model

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
                if self._is_certificate_verification_error(e):
                    self._groq_unavailable = True
                    if self._has_local_fallback():
                        return self._generate_local(messages, temperature, max_tokens)
                    raise RuntimeError(
                        "TLS certificate validation failed while connecting to Groq. "
                        "Check the Windows trust store or network HTTPS inspection; "
                        "do not disable certificate verification."
                    ) from e
                if self._is_rate_limit_error(e):
                    rate_limit_error = self._record_rate_limit(e)
                    if self._has_local_fallback():
                        return self._generate_local(messages, temperature, max_tokens)
                    raise rate_limit_error from e
                if "Connection" in err_str:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"[LLM] Connection failed, retrying in {wait_time:.0f}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue
                raise

        if last_error and self._is_connection_error(last_error):
            self._groq_unavailable = True
            if self._has_local_fallback():
                return self._generate_local(messages, temperature, max_tokens)
        raise last_error  # all retries exhausted

    def generate_stream(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 768,
    ):
        """Stream tokens one-by-one. Yields str chunks."""
        self._check_client()
        if self.client is None or getattr(self, "_groq_unavailable", False):
            yield from self._generate_local_stream(
                messages, temperature, max_tokens
            )
            return
        try:
            self._raise_if_rate_limited()
        except GroqRateLimitError:
            if self._has_local_fallback():
                yield from self._generate_local_stream(
                    messages, temperature, max_tokens
                )
                return
            raise
        self._wait_for_rate_limit()
        self._active_provider = "groq"
        self._active_model = self.model

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
                if self._is_certificate_verification_error(e):
                    self._groq_unavailable = True
                    if self._has_local_fallback():
                        yield from self._generate_local_stream(
                            messages, temperature, max_tokens
                        )
                        return
                    raise RuntimeError(
                        "TLS certificate validation failed while connecting to Groq. "
                        "Check the Windows trust store or network HTTPS inspection; "
                        "do not disable certificate verification."
                    ) from e
                if self._is_rate_limit_error(e):
                    rate_limit_error = self._record_rate_limit(e)
                    if self._has_local_fallback():
                        yield from self._generate_local_stream(
                            messages, temperature, max_tokens
                        )
                        return
                    raise rate_limit_error from e
                if "Connection" in err_str:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"[LLM] Connection failed (stream), retrying in {wait_time:.0f}s")
                    time.sleep(wait_time)
                    continue
                raise

        if last_error and self._is_connection_error(last_error):
            self._groq_unavailable = True
            if self._has_local_fallback():
                yield from self._generate_local_stream(
                    messages, temperature, max_tokens
                )
                return
        raise last_error

    def get_model_name(self) -> str:
        return getattr(self, "_active_model", self.model)

    def is_configured(self) -> bool:
        return self.client is not None or self._has_local_fallback()

    def get_active_provider(self) -> str:
        return getattr(self, "_active_provider", "groq")

    def get_rate_limit_status(self) -> dict:
        remaining = max(0.0, self._rate_limited_until - time.monotonic())
        return {
            "limited": remaining > 0,
            "retry_after_seconds": round(remaining) if remaining > 0 else None,
        }


# Singleton
llm_client = GroqLLMClient()
