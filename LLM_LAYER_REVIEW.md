# LLM Integration Layer - Code Review Request

**Purpose**: Validate the implementation of an LLM integration layer for an epistemic uncertainty system.

**What was built**: A production-ready LLM client that replaces a stub (`return "{}"`) with actual Claude API calls, including retry logic, rate limiting, and robust JSON parsing.

**Review Focus**:
1. Are there any bugs or logic errors?
2. Is the async/sync handling correct?
3. Are the error handling patterns appropriate?
4. Is the JSON parser robust enough for real LLM outputs?
5. Any security concerns?

---

## File 1: llm/providers/base.py (Abstract Interfaces)

```python
"""
Base classes and interfaces for LLM providers.

This module defines the abstract interface that all LLM providers must implement,
plus the standardized request/response data structures that enable provider-agnostic
code in the rest of the system.

Design Principles:
1. Provider-agnostic: Same interface for Claude, OpenAI, or any future provider
2. Observable: Every request/response carries metadata for logging and debugging
3. Cost-aware: Token counts and cost estimates are first-class citizens
4. Immutable specs: Model specifications are frozen to prevent accidental mutation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass(frozen=True)
class ModelSpec:
    """
    Immutable specification for a model version.

    Frozen to ensure model specs can't be accidentally mutated after creation.
    This is important for reproducibility and debugging.
    """
    provider: ModelProvider
    model_id: str                    # e.g., "claude-sonnet-4-20250514"
    version: str                     # Exact version for reproducibility
    context_window: int              # Max tokens in context
    max_output_tokens: int           # Max tokens in response
    supports_json_mode: bool = False # Native JSON mode support
    supports_system_prompt: bool = True
    supports_streaming: bool = True
    cost_per_1k_input: float = 0.0   # USD per 1000 input tokens
    cost_per_1k_output: float = 0.0  # USD per 1000 output tokens

    def full_id(self) -> str:
        """Full identifier for tracking and logging."""
        return f"{self.provider.value}:{self.model_id}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for given token counts."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost


@dataclass
class LLMRequest:
    """
    Standardized request format across all providers.

    This abstraction allows the same request to be sent to any provider
    with automatic translation to provider-specific formats.
    """
    request_id: str                  # Unique ID for tracking
    messages: List[Dict[str, str]]   # [{"role": "user", "content": "..."}]

    # Optional configuration
    system_prompt: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.0         # Default to deterministic for extraction
    top_p: float = 1.0

    # JSON mode (provider-specific handling)
    json_mode: bool = False          # Request structured JSON output

    # Stop sequences
    stop_sequences: List[str] = field(default_factory=list)

    # Metadata for tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def total_prompt_chars(self) -> int:
        """Estimate total characters in prompt (for rough token estimation)."""
        total = len(self.system_prompt or "")
        for msg in self.messages:
            total += len(msg.get("content", ""))
        return total


@dataclass
class LLMResponse:
    """
    Standardized response format across all providers.

    Contains both the response content and comprehensive metadata
    for cost tracking, debugging, and observability.
    """
    request_id: str                  # Links back to request
    model_spec: ModelSpec            # Which model responded

    # Response content
    content: str                     # The actual response text
    finish_reason: str               # "end_turn", "max_tokens", "stop_sequence"

    # Token usage (critical for cost tracking and calibration)
    input_tokens: int
    output_tokens: int

    # Timing
    latency_ms: float                # Total request latency
    created_at: datetime = field(default_factory=datetime.now)

    # Cost tracking
    estimated_cost_usd: float = 0.0

    # Provider-specific metadata (for debugging)
    raw_response_metadata: Dict[str, Any] = field(default_factory=dict)

    def total_tokens(self) -> int:
        """Total tokens used in this request/response cycle."""
        return self.input_tokens + self.output_tokens

    def was_truncated(self) -> bool:
        """Check if response was truncated due to max_tokens."""
        return self.finish_reason == "max_tokens"

    def tokens_per_second(self) -> float:
        """Output tokens per second (throughput metric)."""
        if self.latency_ms <= 0:
            return 0.0
        return self.output_tokens / (self.latency_ms / 1000)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement this interface to work with the
    UnifiedLLMClient. This ensures consistent behavior regardless
    of which underlying LLM service is used.
    """

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Make a completion request to the LLM."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        ...

    @abstractmethod
    def get_model_spec(self) -> ModelSpec:
        """Get the specification for the current model."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""
        ...

    def validate_request(self, request: LLMRequest) -> List[str]:
        """Validate a request before sending. Returns list of errors."""
        errors = []
        spec = self.get_model_spec()

        estimated_input = self.count_tokens(
            (request.system_prompt or "") +
            "".join(m.get("content", "") for m in request.messages)
        )

        if estimated_input + request.max_tokens > spec.context_window:
            errors.append(
                f"Request may exceed context window: "
                f"~{estimated_input} input + {request.max_tokens} max output "
                f"> {spec.context_window} context window"
            )

        if request.max_tokens > spec.max_output_tokens:
            errors.append(
                f"Requested max_tokens ({request.max_tokens}) exceeds "
                f"model maximum ({spec.max_output_tokens})"
            )

        return errors


# Exception classes for provider errors
class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limited. Should retry with exponential backoff."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """Raised when authentication fails. Should NOT retry."""
    pass


class ContextWindowError(LLMError):
    """Raised when request exceeds context window. Should NOT retry as-is."""
    pass


class ContentFilterError(LLMError):
    """Raised when content is filtered. May or may not retry."""
    pass


class APIError(LLMError):
    """Generic API error. May retry depending on status code."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

    def is_retryable(self) -> bool:
        """Check if this error type is typically retryable."""
        if self.status_code is None:
            return True
        return self.status_code >= 500 or self.status_code == 429
```

---

## File 2: llm/providers/anthropic.py (Claude Implementation)

```python
"""
Anthropic Claude provider implementation.
"""

import os
import time
import logging
from typing import Optional, Dict, Any

from .base import (
    BaseLLMProvider, ModelSpec, ModelProvider, LLMRequest, LLMResponse,
    RateLimitError, AuthenticationError, ContextWindowError, APIError,
)

logger = logging.getLogger(__name__)

CLAUDE_MODELS: Dict[str, ModelSpec] = {
    "claude-sonnet-4-20250514": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        version="2025-05-14",
        context_window=200_000,
        max_output_tokens=16_000,
        supports_json_mode=False,
        supports_system_prompt=True,
        supports_streaming=True,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "claude-opus-4-20250514": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-20250514",
        version="2025-05-14",
        context_window=200_000,
        max_output_tokens=16_000,
        supports_json_mode=False,
        supports_system_prompt=True,
        supports_streaming=True,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
    "claude-3-5-sonnet-20241022": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-5-sonnet-20241022",
        version="2024-10-22",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_json_mode=False,
        supports_system_prompt=True,
        supports_streaming=True,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "claude-3-5-haiku-20241022": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-5-haiku-20241022",
        version="2024-10-22",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_json_mode=False,
        supports_system_prompt=True,
        supports_streaming=True,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
    ),
}

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        max_retries: int = 0,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise AuthenticationError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable."
            )

        self._model_id = model_id
        self._timeout = timeout
        self._max_retries = max_retries

        if model_id not in CLAUDE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        self._model_spec = CLAUDE_MODELS[model_id]
        self._async_client = None

    def _get_async_client(self):
        """Lazy-load the async Anthropic client."""
        if self._async_client is None:
            from anthropic import AsyncAnthropic
            self._async_client = AsyncAnthropic(
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._async_client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion request against Claude."""
        client = self._get_async_client()
        start_time = time.perf_counter()

        # Build messages in Anthropic format
        anthropic_messages = []
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue  # System goes in system parameter
            anthropic_messages.append({"role": role, "content": content})

        # Build system prompt
        system = request.system_prompt or ""
        if request.json_mode:
            system += (
                "\n\nIMPORTANT: You MUST respond with valid JSON only. "
                "Do not include any text before or after the JSON."
            )

        try:
            response = await client.messages.create(
                model=self._model_id,
                max_tokens=request.max_tokens,
                system=system if system else None,
                messages=anthropic_messages,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences if request.stop_sequences else None,
            )
        except Exception as e:
            self._translate_exception(e)

        latency_ms = (time.perf_counter() - start_time) * 1000

        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].text

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        estimated_cost = self._model_spec.estimate_cost(input_tokens, output_tokens)

        return LLMResponse(
            request_id=request.request_id,
            model_spec=self._model_spec,
            content=content,
            finish_reason=self._translate_stop_reason(response.stop_reason),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost,
            raw_response_metadata={
                "id": response.id,
                "model": response.model,
                "stop_reason": response.stop_reason,
            },
        )

    def _translate_stop_reason(self, stop_reason: Optional[str]) -> str:
        if stop_reason is None:
            return "unknown"
        return {"end_turn": "end_turn", "max_tokens": "max_tokens",
                "stop_sequence": "stop_sequence", "tool_use": "tool_use"
               }.get(stop_reason, stop_reason)

    def _translate_exception(self, e: Exception) -> None:
        """Translate Anthropic SDK exceptions to our types."""
        try:
            from anthropic import (
                RateLimitError as AnthropicRateLimitError,
                AuthenticationError as AnthropicAuthError,
                BadRequestError, APIStatusError, APITimeoutError, APIConnectionError,
            )
        except ImportError:
            raise APIError(str(e)) from e

        if isinstance(e, AnthropicRateLimitError):
            retry_after = None
            if hasattr(e, 'response') and e.response:
                retry_after_header = e.response.headers.get('retry-after')
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass
            raise RateLimitError(str(e), retry_after=retry_after) from e
        elif isinstance(e, AnthropicAuthError):
            raise AuthenticationError(str(e)) from e
        elif isinstance(e, BadRequestError):
            if "context" in str(e).lower() or "token" in str(e).lower():
                raise ContextWindowError(str(e)) from e
            raise APIError(str(e), status_code=400) from e
        elif isinstance(e, APITimeoutError):
            raise APIError(f"Request timed out: {e}", status_code=408) from e
        elif isinstance(e, APIConnectionError):
            raise APIError(f"Connection error: {e}", status_code=503) from e
        elif isinstance(e, APIStatusError):
            raise APIError(str(e), status_code=getattr(e, 'status_code', None)) from e
        else:
            raise APIError(f"Unknown error: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token heuristic)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def get_model_spec(self) -> ModelSpec:
        return self._model_spec

    @property
    def provider_name(self) -> str:
        return "Anthropic Claude"
```

---

## File 3: llm/retry.py (Retry Handler)

```python
"""
Retry handler with exponential backoff and jitter.
"""

import asyncio
import random
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

from .providers.base import (
    BaseLLMProvider, LLMRequest, LLMResponse,
    RateLimitError, AuthenticationError, ContextWindowError, ContentFilterError, APIError,
)

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    multiplier: float = 2.0
    jitter_factor: float = 0.25
    max_attempts: int = 3
    max_total_time_seconds: float = 300.0

    def __post_init__(self):
        if self.initial_delay_seconds <= 0:
            raise ValueError("initial_delay_seconds must be positive")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("max_delay_seconds must be >= initial_delay_seconds")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")


@dataclass
class RetryAttempt:
    attempt_number: int
    started_at: datetime
    duration_ms: float
    succeeded: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    delay_before_ms: float = 0.0


@dataclass
class RetryStats:
    total_attempts: int = 0
    successful: bool = False
    total_time_ms: float = 0.0
    total_delay_ms: float = 0.0
    attempts: List[RetryAttempt] = field(default_factory=list)
    final_error: Optional[str] = None

    def add_attempt(self, attempt: RetryAttempt):
        self.attempts.append(attempt)
        self.total_attempts = len(self.attempts)
        if attempt.succeeded:
            self.successful = True
        else:
            self.final_error = attempt.error_message


NON_RETRYABLE_EXCEPTIONS = (
    AuthenticationError,
    ContextWindowError,
    ContentFilterError,
)


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int, rate_limit_delay: Optional[float] = None) -> float:
        if rate_limit_delay is not None and rate_limit_delay > 0:
            jitter = rate_limit_delay * 0.1 * random.random()
            return min(rate_limit_delay + jitter, self.config.max_delay_seconds)

        base_delay = self.config.initial_delay_seconds * (
            self.config.multiplier ** (attempt - 1)
        )
        base_delay = min(base_delay, self.config.max_delay_seconds)

        jitter_range = base_delay * self.config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        return max(0.1, base_delay + jitter)

    def _should_retry(self, error: Exception, attempt: int, elapsed_time: float) -> tuple:
        if attempt >= self.config.max_attempts:
            return False, f"Max attempts ({self.config.max_attempts}) exceeded"
        if elapsed_time >= self.config.max_total_time_seconds:
            return False, f"Time budget exceeded"
        if isinstance(error, NON_RETRYABLE_EXCEPTIONS):
            return False, f"Non-retryable error: {type(error).__name__}"
        if isinstance(error, RateLimitError):
            return True, "Rate limit (retryable)"
        if isinstance(error, APIError):
            if error.is_retryable():
                return True, f"Retryable API error (status {error.status_code})"
            return False, f"Non-retryable API error"
        return True, f"Unknown error (retrying)"

    async def execute(
        self, provider: BaseLLMProvider, request: LLMRequest, max_attempts: Optional[int] = None
    ) -> tuple:
        effective_max = max_attempts or self.config.max_attempts
        stats = RetryStats()
        start_time = time.perf_counter()
        last_error = None
        last_delay = 0

        for attempt in range(1, effective_max + 1):
            attempt_start = datetime.now()
            attempt_record = RetryAttempt(
                attempt_number=attempt,
                started_at=attempt_start,
                duration_ms=0,
                succeeded=False,
                delay_before_ms=last_delay * 1000,
            )

            try:
                response = await provider.complete(request)
                attempt_record.succeeded = True
                attempt_record.duration_ms = (time.perf_counter() - start_time) * 1000 - stats.total_delay_ms
                stats.add_attempt(attempt_record)
                stats.total_time_ms = (time.perf_counter() - start_time) * 1000
                return response, stats

            except Exception as e:
                last_error = e
                elapsed = time.perf_counter() - start_time
                attempt_record.error_type = type(e).__name__
                attempt_record.error_message = str(e)[:200]
                attempt_record.duration_ms = (time.perf_counter() - start_time) * 1000 - stats.total_delay_ms
                stats.add_attempt(attempt_record)

                should_retry, reason = self._should_retry(e, attempt, elapsed)
                if not should_retry:
                    stats.total_time_ms = (time.perf_counter() - start_time) * 1000
                    raise

                rate_limit_delay = None
                if isinstance(e, RateLimitError) and e.retry_after:
                    rate_limit_delay = e.retry_after

                delay = self._calculate_delay(attempt, rate_limit_delay)
                last_delay = delay

                logger.warning(f"Request {request.request_id} failed attempt {attempt}, retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
                stats.total_delay_ms += delay * 1000

        stats.total_time_ms = (time.perf_counter() - start_time) * 1000
        if last_error:
            raise last_error
        raise RuntimeError("Retry loop exited without error or response")
```

---

## File 4: llm/json_parser.py (Robust JSON Parser)

```python
"""
Robust JSON parser for LLM responses.

Handles common LLM JSON issues:
1. Trailing commas
2. Single quotes
3. Unquoted keys
4. Markdown wrappers
5. Truncated JSON
6. Python True/False/None
7. NaN/Infinity
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ParseStatus(Enum):
    SUCCESS = "success"
    RECOVERED = "recovered"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class ParseResult:
    status: ParseStatus
    data: Optional[Union[Dict[str, Any], List[Any]]]
    raw_response: str
    extraction_method: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recovered_issues: List[str] = field(default_factory=list)

    def is_success(self) -> bool:
        return self.status in (ParseStatus.SUCCESS, ParseStatus.RECOVERED)

    def get_field(self, key: str, default: Any = None) -> Any:
        if self.data is None:
            return default
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default


class RobustJSONParser:
    EXTRACTION_PATTERNS = [
        (r'```json\s*([\s\S]*?)\s*```', 'markdown_json'),
        (r'```\s*([\s\S]*?)\s*```', 'markdown_generic'),
        (r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})', 'object_extraction'),
        (r'(\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])', 'array_extraction'),
    ]

    def __init__(self, strict: bool = False, max_repair_depth: int = 10):
        self.strict = strict
        self.max_repair_depth = max_repair_depth

    def parse(
        self, response: str, expected_type: str = "object", expected_fields: Optional[List[str]] = None
    ) -> ParseResult:
        if not response or not response.strip():
            return ParseResult(
                status=ParseStatus.FAILED, data=None, raw_response=response,
                extraction_method="none", errors=["Empty response"]
            )

        errors = []

        # Strategy 1: Direct parse
        try:
            data = json.loads(response)
            return ParseResult(
                status=ParseStatus.SUCCESS, data=data, raw_response=response,
                extraction_method="direct"
            )
        except json.JSONDecodeError as e:
            errors.append(f"Direct parse failed: {e}")

        # Strategy 2: Extract from patterns
        for pattern, pattern_name in self.EXTRACTION_PATTERNS:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                try:
                    data = json.loads(extracted)
                    return ParseResult(
                        status=ParseStatus.RECOVERED, data=data, raw_response=response,
                        extraction_method=f"pattern:{pattern_name}",
                        recovered_issues=[f"Extracted using {pattern_name}"]
                    )
                except json.JSONDecodeError:
                    fixed = self._fix_common_issues(extracted)
                    try:
                        data = json.loads(fixed)
                        return ParseResult(
                            status=ParseStatus.RECOVERED, data=data, raw_response=response,
                            extraction_method=f"pattern:{pattern_name}+fix",
                            recovered_issues=[f"Extracted using {pattern_name}", "Applied fixes"]
                        )
                    except json.JSONDecodeError:
                        pass

        # Strategy 3: Fix common issues on full response
        fixed = self._fix_common_issues(response)
        try:
            data = json.loads(fixed)
            return ParseResult(
                status=ParseStatus.RECOVERED, data=data, raw_response=response,
                extraction_method="fixed_common",
                recovered_issues=["Applied common JSON fixes"]
            )
        except json.JSONDecodeError as e:
            errors.append(f"Common fixes failed: {e}")

        # Strategy 4: Repair truncated JSON
        repaired = self._repair_truncated(response)
        if repaired and repaired != response:
            try:
                data = json.loads(repaired)
                return ParseResult(
                    status=ParseStatus.RECOVERED, data=data, raw_response=response,
                    extraction_method="truncation_repair",
                    recovered_issues=["Repaired truncated JSON"],
                    warnings=["JSON was truncated"]
                )
            except json.JSONDecodeError:
                pass

        # Strategy 5: Partial field extraction
        if expected_fields:
            partial = self._extract_fields(response, expected_fields)
            if partial:
                return ParseResult(
                    status=ParseStatus.PARTIAL, data=partial, raw_response=response,
                    extraction_method="partial_extraction",
                    recovered_issues=[f"Extracted {len(partial)} fields via regex"],
                    errors=errors
                )

        return ParseResult(
            status=ParseStatus.FAILED, data=None, raw_response=response,
            extraction_method="none", errors=errors
        )

    def _fix_common_issues(self, text: str) -> str:
        text = text.strip()

        # Remove markdown fences
        if text.startswith('```'):
            first_newline = text.find('\n')
            if first_newline > 0:
                text = text[first_newline + 1:]
        if text.endswith('```'):
            text = text[:-3].rstrip()

        # Fix trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Fix missing commas between elements
        text = re.sub(r'"\s*\n\s*"', '",\n"', text)

        # Single quotes to double quotes (for string delimiters)
        text = re.sub(r"(?<=[{,:\[\s])'([^']*)'(?=[},:\]\s])", r'"\1"', text)

        # Unquoted keys
        text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

        # Python booleans/None
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)

        # NaN/Infinity
        text = re.sub(r'\bNaN\b', 'null', text)
        text = re.sub(r'\bInfinity\b', 'null', text)
        text = re.sub(r'-Infinity\b', 'null', text)

        return text

    def _repair_truncated(self, text: str) -> Optional[str]:
        text = text.strip()

        # Fix unclosed strings
        if text.count('"') % 2 == 1:
            last_quote_idx = text.rfind('"')
            if last_quote_idx > 0:
                before_quote = text[:last_quote_idx]
                if before_quote.count('"') % 2 == 1:
                    text = text + '"'

        # Count brackets
        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            elif char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1

        if open_braces < 0 or open_brackets < 0:
            return None

        if open_braces == 0 and open_brackets == 0:
            return text

        # Remove trailing comma and close brackets
        text = re.sub(r',\s*$', '', text)
        text = re.sub(r':\s*$', ': null', text)
        text = text.rstrip()
        if text.endswith(','):
            text = text[:-1]

        text += ']' * open_brackets
        text += '}' * open_braces

        return text

    def _extract_fields(self, text: str, fields: List[str]) -> Dict[str, Any]:
        result = {}
        for field_name in fields:
            # String value
            match = re.search(rf'"{field_name}"\s*:\s*"([^"]*)"', text)
            if match:
                result[field_name] = match.group(1)
                continue
            # Number
            match = re.search(rf'"{field_name}"\s*:\s*(-?\d+\.?\d*)', text)
            if match:
                value = match.group(1)
                result[field_name] = float(value) if '.' in value else int(value)
                continue
            # Boolean/null
            match = re.search(rf'"{field_name}"\s*:\s*(true|false|null)', text)
            if match:
                result[field_name] = {'true': True, 'false': False, 'null': None}[match.group(1)]
        return result


def parse_json_response(response: str, expected_fields: Optional[List[str]] = None) -> ParseResult:
    """Convenience function."""
    return RobustJSONParser().parse(response, expected_fields=expected_fields)
```

---

## File 5: llm/rate_limit.py (Rate Limiter) - Key excerpts

```python
"""Token bucket rate limiter with per-minute and per-day limits."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Deque
from collections import deque
from enum import Enum


class RateLimitType(Enum):
    REQUESTS_PER_MINUTE = "rpm"
    TOKENS_PER_MINUTE = "tpm"
    REQUESTS_PER_DAY = "rpd"
    TOKENS_PER_DAY = "tpd"


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 50
    tokens_per_minute: int = 80_000
    requests_per_day: int = 10_000
    tokens_per_day: int = 2_000_000
    max_wait_seconds: float = 60.0
    burst_allowance: float = 1.2


@dataclass
class RateLimitState:
    minute_requests: Deque[float] = field(default_factory=deque)
    minute_tokens: Deque[tuple] = field(default_factory=deque)
    day_requests: int = 0
    day_tokens: int = 0
    day_start: datetime = field(default_factory=datetime.now)


class RateLimitExceeded(Exception):
    def __init__(self, message: str, limit_type: RateLimitType, retry_after: float):
        super().__init__(message)
        self.limit_type = limit_type
        self.retry_after = retry_after


class RateLimiter:
    """Token bucket rate limiter with sliding window."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.state = RateLimitState()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0, block: bool = True) -> float:
        async with self._lock:
            self._cleanup_old_entries()
            self._check_day_rollover()
            wait_time, limit_type = self._calculate_wait_time(estimated_tokens)

            if wait_time > 0:
                if not block:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded: {limit_type.value}",
                        limit_type=limit_type, retry_after=wait_time
                    )
                if wait_time > self.config.max_wait_seconds:
                    raise RateLimitExceeded(
                        f"Wait {wait_time:.1f}s exceeds max {self.config.max_wait_seconds}s",
                        limit_type=limit_type, retry_after=wait_time
                    )
            return wait_time

    async def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        async with self._lock:
            now = time.time()
            total_tokens = input_tokens + output_tokens
            self.state.minute_requests.append(now)
            self.state.minute_tokens.append((now, total_tokens))
            self.state.day_requests += 1
            self.state.day_tokens += total_tokens

    def _cleanup_old_entries(self) -> None:
        now = time.time()
        minute_ago = now - 60
        while self.state.minute_requests and self.state.minute_requests[0] < minute_ago:
            self.state.minute_requests.popleft()
        while self.state.minute_tokens and self.state.minute_tokens[0][0] < minute_ago:
            self.state.minute_tokens.popleft()

    # ... additional methods for wait time calculation
```

---

## File 6: llm/client.py (Unified Client) - Key excerpts

```python
"""Unified LLM Client - Main entry point."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

from .providers.base import BaseLLMProvider, LLMRequest, LLMResponse, ModelSpec, LLMError
from .providers.anthropic import AnthropicProvider, DEFAULT_MODEL
from .retry import RetryHandler, RetryConfig, RetryStats
from .rate_limit import RateLimiter, RateLimitConfig, RateLimitExceeded
from .json_parser import RobustJSONParser, ParseResult, ParseStatus

logger = logging.getLogger(__name__)


@dataclass
class LLMClientConfig:
    provider: str = "anthropic"
    model: str = DEFAULT_MODEL
    anthropic_api_key: Optional[str] = None
    default_max_tokens: int = 4096
    default_temperature: float = 0.0
    request_timeout: float = 300.0
    retry_max_attempts: int = 3
    rate_limit_rpm: int = 50
    rate_limit_tpm: int = 80_000
    rate_limit_enabled: bool = True
    json_strict_mode: bool = False
    track_costs: bool = True


@dataclass
class CompletionResult:
    success: bool
    content: str
    data: Optional[Dict[str, Any]] = None
    parse_result: Optional[ParseResult] = None
    request_id: str = ""
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    retry_stats: Optional[RetryStats] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class UnifiedLLMClient:
    """High-level client combining provider, retry, rate limiting, JSON parsing."""

    def __init__(self, config: Optional[LLMClientConfig] = None):
        self.config = config or LLMClientConfig()
        self._provider = self._create_provider()
        self._retry_handler = RetryHandler(RetryConfig(
            max_attempts=self.config.retry_max_attempts,
        ))
        self._rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=self.config.rate_limit_rpm,
            tokens_per_minute=self.config.rate_limit_tpm,
        )) if self.config.rate_limit_enabled else None
        self._json_parser = RobustJSONParser(strict=self.config.json_strict_mode)

    def _create_provider(self) -> BaseLLMProvider:
        if self.config.provider == "anthropic":
            return AnthropicProvider(
                api_key=self.config.anthropic_api_key,
                model_id=self.config.model,
                timeout=self.config.request_timeout,
            )
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    async def complete(self, prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> CompletionResult:
        """Make a completion request."""
        request_id = str(uuid.uuid4())[:8]
        request = LLMRequest(
            request_id=request_id,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature if temperature is not None else self.config.default_temperature,
        )

        # Rate limit check
        if self._rate_limiter:
            wait_time = await self._rate_limiter.acquire(
                estimated_tokens=self._provider.count_tokens(prompt)
            )
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Execute with retry
        response, retry_stats = await self._retry_handler.execute(self._provider, request)

        if self._rate_limiter:
            await self._rate_limiter.record_usage(response.input_tokens, response.output_tokens)

        return CompletionResult(
            success=True,
            content=response.content,
            request_id=request_id,
            model_used=response.model_spec.model_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost_usd=response.estimated_cost_usd,
            latency_ms=response.latency_ms,
            retry_stats=retry_stats,
        )

    async def complete_json(self, prompt: str, system_prompt: Optional[str] = None,
                           expected_fields: Optional[List[str]] = None, **kwargs) -> CompletionResult:
        """Make a completion expecting JSON response."""
        json_system = (system_prompt or "") + (
            "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no text before/after."
        )

        result = await self.complete(prompt=prompt, system_prompt=json_system, **kwargs)
        if not result.success:
            return result

        parse_result = self._json_parser.parse(result.content, expected_fields=expected_fields)
        result.parse_result = parse_result
        result.data = parse_result.data

        if not parse_result.is_success():
            result.success = False
            result.error = f"JSON parse failed: {parse_result.errors}"
            result.error_type = "JSONParseError"

        return result
```

---

## File 7: Modified core/pattern_extractor.py - Key changes

```python
# At top of file, added:
import asyncio
import logging
from llm import UnifiedLLMClient, LLMClientConfig, CompletionResult
LLM_AVAILABLE = True  # Or False if import fails

# Constructor now accepts:
def __init__(
    self,
    event_store: EventStore,
    llm_client: Optional["UnifiedLLMClient"] = None,
    llm_config: Optional["LLMClientConfig"] = None,
    calibration_data_path: Optional[str] = None
):
    # Initialize LLM client
    if llm_client:
        self.llm_client = llm_client
    elif llm_config and LLM_AVAILABLE:
        self.llm_client = UnifiedLLMClient(llm_config)
    elif LLM_AVAILABLE:
        try:
            self.llm_client = UnifiedLLMClient()
        except Exception:
            self.llm_client = None
    else:
        self.llm_client = None

# New async extraction method:
async def extract_patterns_async(self, events, pattern_type=None, auto_validate_threshold=0.95):
    prompt = self.build_extraction_prompt(events, pattern_type)
    if self.llm_client:
        response = await self._call_llm_async(prompt)
        patterns = self._parse_llm_response(response, events)
        if not patterns:
            patterns = self._demo_extraction(events)  # Fallback
    else:
        patterns = self._demo_extraction(events)
    # ... apply calibration, return patterns

# New LLM call method (replaces the stub):
async def _call_llm_async(self, prompt: ExtractionPrompt) -> str:
    if not self.llm_client:
        return "{}"

    user_prompt = f"EVENTS:\n{prompt.event_context}\n\nINSTRUCTIONS:\n{prompt.extraction_instructions}\n\nFORMAT:\n{prompt.output_format}"

    result = await self.llm_client.complete_json(
        prompt=user_prompt,
        system_prompt=prompt.system_prompt,
        expected_fields=["patterns", "no_patterns_found"],
        max_tokens=4096,
        temperature=0.0,
    )

    if result.success and result.data:
        return result.content
    return "{}"
```

---

## Validation Questions

1. **Async/Sync Handling**: Is the pattern of wrapping async calls in `asyncio.run()` for sync compatibility correct? Any edge cases with nested event loops?

2. **Error Handling**: Are all the exception types properly defined and translated? Missing any edge cases?

3. **JSON Parser**: Are the regex patterns for fixing common issues correct? Any patterns that might break valid JSON?

4. **Rate Limiter**: Is the sliding window implementation correct? Any race conditions with the async lock?

5. **Security**: Any concerns with the API key handling or logging?

6. **Memory**: Any concerns with the deque-based sliding windows for high-volume usage?

---

## Test Results (Non-API)

```
============================================================
LLM Integration Layer - Quick Validation
============================================================

1. Testing imports...
   ✓ All imports successful

2. Testing JSON parser...
   ✓ Valid JSON: PASS
   ✓ Markdown wrapper: PASS
   ✓ Trailing comma: PASS
   ✓ Python boolean: PASS
   ✓ Empty string: PASS

3. Testing rate limiter...
   ✓ Rate limiter tracking: PASS

4. Testing model specifications...
   ✓ claude-sonnet-4-20250514: 200,000 tokens
   ✓ claude-opus-4-20250514: 200,000 tokens
   ✓ claude-3-5-sonnet-20241022: 200,000 tokens
   ✓ claude-3-5-haiku-20241022: 200,000 tokens

5. Testing pattern extractor integration...
   ✓ LLM_AVAILABLE = True
   ✓ PatternType enum has 7 types

All validation checks PASSED!
============================================================
```

Please review and flag any issues.
