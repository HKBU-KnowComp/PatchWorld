"""OpenAI-compatible LLM client used for inducing and refining world-model code."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import hashlib
import math
import os
import random
import sqlite3
import threading
import time

from openai import OpenAI

_MODEL_REQUEST_BUCKETS: Dict[Tuple[str, str], "_ModelRequestBucket"] = {}
_MODEL_REQUEST_BUCKETS_LOCK = threading.Lock()


class _ModelRequestBucket:
    """Model-scoped token bucket for bounding in-flight API calls."""

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self._tokens = self.capacity
        self._cond = threading.Condition()

    def acquire(self, timeout_s: float | None = None) -> None:
        deadline = None if timeout_s is None or timeout_s <= 0.0 else time.monotonic() + timeout_s
        with self._cond:
            while self._tokens <= 0:
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Timed out waiting for an available model request token."
                    )
                self._cond.wait(timeout=remaining)
            self._tokens -= 1

    def release(self) -> None:
        with self._cond:
            self._tokens = min(self.capacity, self._tokens + 1)
            self._cond.notify()


class LLMClient:
    """Minimal OpenAI-compatible client used for world-model induction.

    By default this mirrors the DeepInfra setup used elsewhere in the repo:
    - Reads API key from .deepinfra_api_key if not provided.
    - Uses an OpenAI-compatible HTTP API.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or self._load_api_key()
        self.base_url = (
            base_url
            or os.getenv("PATCHWORLD_LLM_BASE_URL")
            or os.getenv("DEEPINFRA_BASE_URL")
            or os.getenv("MIMO_BASE_URL")
            or "https://api.deepinfra.com/v1/openai"
        ).rstrip("/")
        self.max_concurrent_requests_per_model = max(
            int(os.getenv("PATCHWORLD_LLM_MAX_CONCURRENT_PER_MODEL", "400")), 1
        )
        self.concurrent_request_wait_timeout_s = max(
            float(os.getenv("PATCHWORLD_LLM_CONCURRENT_WAIT_TIMEOUT_S", "180.0")), 1.0
        )
        self.parallel_model_calls = max(
            int(os.getenv("PATCHWORLD_LLM_PARALLEL_MODELS", "7")), 1
        )
        self.cache_enabled = os.getenv("PATCHWORLD_LLM_CACHE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.cache_path = Path(
            os.getenv(
                "PATCHWORLD_LLM_CACHE_PATH",
                "artifacts/patchworld/llm_cache.sqlite3",
            )
        )
        parallel_pressure = max(0.0, math.log2(self.parallel_model_calls))
        self.request_timeout_s = max(
            float(os.getenv("PATCHWORLD_LLM_REQUEST_TIMEOUT_S", "120.0")), 1.0
        )
        self.client = self._make_client()
        self._usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._usage_lock = threading.Lock()
        # Retry settings for transient API failures (429/model busy, timeouts, 5xx).
        self.max_retries = max(int(os.getenv("PATCHWORLD_LLM_MAX_RETRIES", "8")), 0)
        self.retry_base_sleep_s = max(float(os.getenv("PATCHWORLD_LLM_RETRY_BASE_SLEEP_S", "1.5")), 0.0)
        self.retry_max_sleep_s = max(
            float(
                os.getenv(
                    "PATCHWORLD_LLM_RETRY_MAX_SLEEP_S",
                    f"{60.0 + 10.0 * parallel_pressure:.1f}",
                )
            ),
            0.0,
        )
        self.retry_jitter = min(
            max(
                float(
                    os.getenv(
                        "PATCHWORLD_LLM_RETRY_JITTER",
                        f"{min(0.6, 0.2 + 0.1 * parallel_pressure):.2f}",
                    )
                ),
                0.0,
            ),
            1.0,
        )
        self.max_retry_wait_s = max(
            float(
                os.getenv(
                    "PATCHWORLD_LLM_MAX_RETRY_WAIT_S",
                    f"{120.0 + 30.0 * parallel_pressure:.1f}",
                )
            ),
            0.0,
        )
        self.model_busy_min_sleep_s = max(
            float(
                os.getenv(
                    "PATCHWORLD_LLM_MODEL_BUSY_MIN_SLEEP_S",
                    f"{15.0 + 5.0 * parallel_pressure:.1f}",
                )
            ),
            0.0,
        )
        self.model_busy_multiplier = max(
            float(os.getenv("PATCHWORLD_LLM_MODEL_BUSY_MULTIPLIER", "2.0")), 1.0
        )
        if self.cache_enabled:
            self._init_cache()

    def _make_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_s,
            max_retries=0,
        )

    def _recreate_client(self) -> None:
        self.client = self._make_client()

    def _init_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    op_name TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    content TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(llm_cache)").fetchall()
            }
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if name not in columns:
                    conn.execute(
                        f"ALTER TABLE llm_cache ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    )

    def _cache_key(self, *, op_name: str, system: str, prompt: str, temperature: float) -> str:
        payload = "\n".join(
            [
                "v1",
                self.base_url,
                self.model,
                op_name,
                f"{temperature:.4f}",
                system,
                prompt,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Tuple[str, Dict[str, int]]]:
        if not self.cache_enabled:
            return None
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            row = conn.execute(
                """
                SELECT content, prompt_tokens, completion_tokens, total_tokens
                FROM llm_cache
                WHERE cache_key = ? AND model = ? AND base_url = ?
                """,
                (key, self.model, self.base_url),
            ).fetchone()
        if not row:
            return None
        usage = {
            "calls": 1,
            "prompt_tokens": int(row[1] or 0),
            "completion_tokens": int(row[2] or 0),
            "total_tokens": int(row[3] or 0),
        }
        return str(row[0]), usage

    def _cache_put(
        self,
        *,
        key: str,
        op_name: str,
        prompt: str,
        temperature: float,
        content: str,
        usage: Dict[str, int],
    ) -> None:
        if not self.cache_enabled:
            return
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO llm_cache
                (cache_key, model, base_url, op_name, temperature, prompt_sha256, content,
                 prompt_tokens, completion_tokens, total_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    self.model,
                    self.base_url,
                    op_name,
                    float(temperature),
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    content,
                    int(usage.get("prompt_tokens", 0) or 0),
                    int(usage.get("completion_tokens", 0) or 0),
                    int(usage.get("total_tokens", 0) or 0),
                    time.time(),
                ),
            )

    def _request_bucket(self) -> _ModelRequestBucket:
        key = (self.base_url, self.model)
        with _MODEL_REQUEST_BUCKETS_LOCK:
            bucket = _MODEL_REQUEST_BUCKETS.get(key)
            if bucket is None or bucket.capacity != self.max_concurrent_requests_per_model:
                bucket = _ModelRequestBucket(self.max_concurrent_requests_per_model)
                _MODEL_REQUEST_BUCKETS[key] = bucket
            return bucket

    def _accumulate_usage(self, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        with self._usage_lock:
            if usage is None:
                self._usage["calls"] += 1
                return
            self._usage["calls"] += 1
            self._usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            self._usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

    @staticmethod
    def _usage_from_response(resp: Any) -> Dict[str, int]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {
                "calls": 1,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        return {
            "calls": 1,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def _accumulate_usage_dict(self, usage: Dict[str, int]) -> None:
        with self._usage_lock:
            self._usage["calls"] += int(usage.get("calls", 0) or 0)
            self._usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            self._usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            self._usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

    def get_usage(self) -> Dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def reset_usage(self) -> None:
        with self._usage_lock:
            self._usage = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    @staticmethod
    def usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
        keys = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in keys}

    def _load_api_key(self) -> str:
        key = (
            os.getenv("PATCHWORLD_LLM_API_KEY")
            or os.getenv("DEEPINFRA_API_KEY")
            or os.getenv("MIMO_API_KEY")
        )
        if key:
            return key
        for key_file in (Path(".deepinfra_api_key"), Path(".mimo_api_key")):
            if key_file.exists():
                return key_file.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            "World-model induction requires a DeepInfra-style API key. "
            "Set PATCHWORLD_LLM_API_KEY, DEEPINFRA_API_KEY, MIMO_API_KEY, "
            "or create .deepinfra_api_key / .mimo_api_key."
        )

    @staticmethod
    def _status_code_from_error(exc: Exception) -> Optional[int]:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        return int(status) if isinstance(status, int) else None

    def _is_retryable_error(self, exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        if "ratelimit" in name or "timeout" in name or "connection" in name:
            return True

        status = self._status_code_from_error(exc)
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True

        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in (
                "model busy",
                "rate limit",
                "temporarily unavailable",
                "timeout",
                "connection reset",
                "connection error",
                "server disconnected",
            )
        )

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        if "connection" in name:
            return True
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in (
                "connection error",
                "connection reset",
                "server disconnected",
                "remote protocol error",
            )
        )

    @staticmethod
    def _is_model_busy_error(exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        if "ratelimit" in name:
            msg = str(exc).lower()
            return "model busy" in msg

        msg = str(exc).lower()
        return "model busy" in msg

    def _retry_delay(self, retry_idx: int, *, model_busy: bool = False) -> float:
        if self.retry_base_sleep_s <= 0.0:
            delay = 0.0
        else:
            delay = self.retry_base_sleep_s * (2 ** (retry_idx - 1))
            if self.retry_max_sleep_s > 0.0:
                delay = min(delay, self.retry_max_sleep_s)

        if model_busy:
            delay = max(delay * self.model_busy_multiplier, self.model_busy_min_sleep_s)
            if self.retry_max_sleep_s > 0.0:
                delay = min(delay, self.retry_max_sleep_s)

        if self.retry_jitter > 0.0:
            lo = max(0.0, 1.0 - self.retry_jitter)
            hi = 1.0 + self.retry_jitter
            delay *= random.uniform(lo, hi)
        return delay

    def _call_with_retries(self, op_name: str, fn: Callable[[], Any]) -> Any:
        attempts = self.max_retries + 1
        total_wait_s = 0.0
        for attempt in range(1, attempts + 1):
            bucket = self._request_bucket()
            bucket.acquire(timeout_s=self.concurrent_request_wait_timeout_s)
            try:
                try:
                    return fn()
                finally:
                    bucket.release()
            except Exception as exc:
                should_retry = self._is_retryable_error(exc) and attempt < attempts
                if not should_retry:
                    raise
                if self._is_connection_error(exc):
                    self._recreate_client()
                is_model_busy = self._is_model_busy_error(exc)
                delay = self._retry_delay(attempt, model_busy=is_model_busy)
                if self.max_retry_wait_s > 0.0:
                    remaining_wait_s = max(0.0, self.max_retry_wait_s - total_wait_s)
                    if remaining_wait_s <= 0.0:
                        print(
                            f"[patchworld_inducer] {op_name} retry budget exhausted "
                            f"after {total_wait_s:.1f}s of waiting; giving up."
                        )
                        raise
                    delay = min(delay, remaining_wait_s)
                reason = "model-busy cooldown" if is_model_busy else "retry backoff"
                print(
                    f"[patchworld_inducer] {op_name} retry {attempt}/{self.max_retries} "
                    f"after {exc.__class__.__name__}: sleeping {delay:.1f}s ({reason})"
                )
                if delay > 0:
                    time.sleep(delay)
                    total_wait_s += delay
        raise RuntimeError(f"{op_name} failed after retries")

    def generate(self, prompt: str) -> str:
        system = "You are a precise Python code generator."
        temperature = 0.2
        cache_key = self._cache_key(
            op_name="llm.generate",
            system=system,
            prompt=prompt,
            temperature=temperature,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            content, usage = cached
            self._accumulate_usage_dict(usage)
            return content
        resp = self._call_with_retries(
            "llm.generate",
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            ),
        )
        usage = self._usage_from_response(resp)
        self._accumulate_usage_dict(usage)
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned an empty completion content.")
        self._cache_put(
            key=cache_key,
            op_name="llm.generate",
            prompt=prompt,
            temperature=temperature,
            content=content,
            usage=usage,
        )
        return content

    def extract_rules(self, prompt: str) -> str:
        """Call the LLM for rule extraction (plain text, not code)."""
        system = (
            "You are an expert at analyzing agent-environment interactions "
            "and extracting general transition rules."
        )
        temperature = 0.1
        cache_key = self._cache_key(
            op_name="llm.extract_rules",
            system=system,
            prompt=prompt,
            temperature=temperature,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            content, usage = cached
            self._accumulate_usage_dict(usage)
            return content
        resp = self._call_with_retries(
            "llm.extract_rules",
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            ),
        )
        usage = self._usage_from_response(resp)
        self._accumulate_usage_dict(usage)
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned an empty completion content.")
        self._cache_put(
            key=cache_key,
            op_name="llm.extract_rules",
            prompt=prompt,
            temperature=temperature,
            content=content,
            usage=usage,
        )
        return content
