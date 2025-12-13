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

    Implementation Requirements:
    1. complete() must be async and handle provider-specific API calls
    2. count_tokens() should provide accurate estimates for the provider
    3. Providers should handle their own authentication
    4. Errors should be raised as specific exception types for retry logic
    """

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Make a completion request to the LLM.

        Args:
            request: Standardized request object

        Returns:
            Standardized response object

        Raises:
            RateLimitError: If rate limited (should retry with backoff)
            AuthenticationError: If auth fails (should not retry)
            TimeoutError: If request times out (may retry)
            APIError: For other API errors
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Used for:
        - Pre-flight context window checks
        - Cost estimation before requests
        - Rate limiting (tokens per minute)

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        ...

    @abstractmethod
    def get_model_spec(self) -> ModelSpec:
        """
        Get the specification for the current model.

        Returns:
            Immutable model specification
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""
        ...

    def validate_request(self, request: LLMRequest) -> List[str]:
        """
        Validate a request before sending.

        Returns list of validation errors (empty if valid).
        Override in subclasses for provider-specific validation.
        """
        errors = []

        spec = self.get_model_spec()

        # Check context window
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

        # Check max output tokens
        if request.max_tokens > spec.max_output_tokens:
            errors.append(
                f"Requested max_tokens ({request.max_tokens}) exceeds "
                f"model maximum ({spec.max_output_tokens})"
            )

        return errors


# Exception classes for provider errors
# These allow the retry handler to make informed decisions

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limited. Should retry with exponential backoff."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after  # Seconds to wait before retry


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
            return True  # Unknown, try anyway
        # 5xx errors are typically retryable, 4xx are not
        return self.status_code >= 500 or self.status_code == 429
