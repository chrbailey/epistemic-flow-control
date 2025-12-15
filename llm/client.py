"""
Unified LLM Client - The main entry point for all LLM interactions.

This module provides a high-level interface that combines:
- Provider abstraction (currently Anthropic, extensible to others)
- Automatic retry with exponential backoff
- Rate limiting to stay within API quotas
- Robust JSON parsing for structured outputs
- Comprehensive observability (logging, metrics, cost tracking)

Design Philosophy:
- Single entry point: All LLM calls go through UnifiedLLMClient
- Fail gracefully: Prefer returning partial results over raising exceptions
- Observable: Every request is logged with full context for debugging
- Cost-aware: Track and report costs at every level

Usage:
    client = UnifiedLLMClient(LLMClientConfig(
        anthropic_api_key="sk-ant-...",
        model="claude-sonnet-4-20250514",
    ))

    result = await client.complete_json(
        prompt="Extract the key facts from this text...",
        system_prompt="You are a fact extractor...",
        expected_fields=["facts", "confidence"],
    )

    if result.success:
        facts = result.data["facts"]
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

from .providers.base import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ModelSpec,
    LLMError,
    RateLimitError,
    AuthenticationError,
    ContextWindowError,
)
from .providers.anthropic import AnthropicProvider, CLAUDE_MODELS, DEFAULT_MODEL
from .retry import RetryHandler, RetryConfig, RetryStats
from .rate_limit import RateLimiter, RateLimitConfig, RateLimitStatus, RateLimitExceeded
from .json_parser import RobustJSONParser, ParseResult, ParseStatus

logger = logging.getLogger(__name__)


@dataclass
class LLMClientConfig:
    """
    Configuration for the unified LLM client.

    Sensible defaults are provided for all parameters, but you'll
    need to provide API keys either here or via environment variables.
    """
    # Provider selection
    provider: str = "anthropic"  # Currently only anthropic supported
    model: str = DEFAULT_MODEL

    # API authentication (falls back to environment variables)
    anthropic_api_key: Optional[str] = None

    # Request defaults
    default_max_tokens: int = 4096
    default_temperature: float = 0.0  # Deterministic for extraction tasks
    request_timeout: float = 300.0  # 5 minutes

    # Retry configuration
    retry_max_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 60.0

    # Rate limiting
    rate_limit_rpm: int = 50  # Requests per minute
    rate_limit_tpm: int = 80_000  # Tokens per minute
    rate_limit_enabled: bool = True

    # JSON parsing
    json_strict_mode: bool = False  # If True, only accept SUCCESS status

    # Observability
    log_requests: bool = True
    log_responses: bool = True
    track_costs: bool = True

    def __repr__(self) -> str:
        """Safe repr that redacts API keys."""
        key_display = "***REDACTED***" if self.anthropic_api_key else "None"
        return (
            f"LLMClientConfig(provider={self.provider!r}, model={self.model!r}, "
            f"anthropic_api_key={key_display}, ...)"
        )


@dataclass
class CompletionResult:
    """
    Result of a completion request with full context.

    This wraps the raw LLM response with additional metadata about
    retries, rate limiting, and parsing.
    """
    success: bool
    content: str  # Raw response content

    # For JSON requests
    data: Optional[Dict[str, Any]] = None  # Parsed JSON data
    parse_result: Optional[ParseResult] = None

    # Request/Response details
    request_id: str = ""
    model_used: str = ""

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost tracking
    estimated_cost_usd: float = 0.0

    # Timing
    latency_ms: float = 0.0
    total_time_ms: float = 0.0  # Including retries and rate limit waits

    # Retry information
    retry_stats: Optional[RetryStats] = None
    attempts: int = 1

    # Rate limit information
    rate_limit_wait_ms: float = 0.0

    # Error information (if failed)
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Raw response for debugging
    raw_response: Optional[LLMResponse] = None


@dataclass
class ClientMetrics:
    """Aggregated metrics for the client session."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    total_cost_usd: float = 0.0

    total_latency_ms: float = 0.0
    total_retry_time_ms: float = 0.0
    total_rate_limit_wait_ms: float = 0.0

    requests_retried: int = 0
    requests_rate_limited: int = 0

    json_parse_successes: int = 0
    json_parse_recovered: int = 0
    json_parse_failures: int = 0

    session_start: datetime = field(default_factory=datetime.now)

    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    def avg_latency_ms(self) -> float:
        """Calculate average latency per successful request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    def tokens_per_dollar(self) -> float:
        """Calculate token efficiency."""
        if self.total_cost_usd == 0:
            return float('inf')
        return (self.total_input_tokens + self.total_output_tokens) / self.total_cost_usd


class UnifiedLLMClient:
    """
    High-level client for LLM interactions.

    This is the main entry point for all LLM calls in the system.
    It handles provider selection, retries, rate limiting, and JSON parsing
    automatically, presenting a simple interface to callers.

    Thread Safety:
    - The client is async-safe and can be used from multiple coroutines
    - Rate limiting uses locks internally for thread safety
    - Metrics updates are not locked (acceptable for monitoring purposes)
    """

    def __init__(self, config: Optional[LLMClientConfig] = None):
        """
        Initialize the unified client.

        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self.config = config or LLMClientConfig()
        self._metrics = ClientMetrics()

        # Initialize provider
        self._provider = self._create_provider()

        # Initialize retry handler
        self._retry_handler = RetryHandler(RetryConfig(
            max_attempts=self.config.retry_max_attempts,
            initial_delay_seconds=self.config.retry_initial_delay,
            max_delay_seconds=self.config.retry_max_delay,
        ))

        # Initialize rate limiter
        self._rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=self.config.rate_limit_rpm,
            tokens_per_minute=self.config.rate_limit_tpm,
        )) if self.config.rate_limit_enabled else None

        # Initialize JSON parser
        self._json_parser = RobustJSONParser(
            strict=self.config.json_strict_mode,
        )

        logger.info(
            f"UnifiedLLMClient initialized: provider={self.config.provider}, "
            f"model={self.config.model}"
        )

    def _create_provider(self) -> BaseLLMProvider:
        """Create the appropriate provider based on config."""
        if self.config.provider == "anthropic":
            return AnthropicProvider(
                api_key=self.config.anthropic_api_key,
                model_id=self.config.model,
                timeout=self.config.request_timeout,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompletionResult:
        """
        Make a completion request and get raw text response.

        This is the basic completion method. For structured JSON output,
        use complete_json() instead.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt.
                SECURITY NOTE: The caller is responsible for sanitizing the system_prompt.
                User-controlled content should not be directly passed as system_prompt
                without validation, as it could alter the LLM's behavior (prompt injection).
            max_tokens: Max response tokens (uses config default if not set)
            temperature: Sampling temperature (uses config default if not set)
            metadata: Optional metadata to attach to request

        Returns:
            CompletionResult with response content and metadata
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        # Build the request
        request = LLMRequest(
            request_id=request_id,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature if temperature is not None else self.config.default_temperature,
            metadata=metadata or {},
        )

        if self.config.log_requests:
            logger.debug(
                f"Request {request_id}: {len(prompt)} chars, "
                f"max_tokens={request.max_tokens}"
            )

        # Check rate limits and acquire reservation
        rate_limit_wait = 0.0
        reservation_id: Optional[str] = None
        if self._rate_limiter:
            try:
                estimated_tokens = self._provider.count_tokens(prompt)
                wait_time, reservation_id = await self._rate_limiter.acquire(
                    estimated_tokens=estimated_tokens,
                    block=True,
                )
                if wait_time > 0:
                    rate_limit_wait = wait_time * 1000
                    self._metrics.requests_rate_limited += 1
                    logger.info(f"Request {request_id}: rate limited, waited {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            except RateLimitExceeded as e:
                return self._create_error_result(
                    request_id=request_id,
                    error=str(e),
                    error_type="RateLimitExceeded",
                    start_time=start_time,
                )

        # Execute with retry
        try:
            response, retry_stats = await self._retry_handler.execute(
                self._provider,
                request,
            )

            # Record usage and complete reservation
            if self._rate_limiter:
                await self._rate_limiter.record_usage(
                    response.input_tokens,
                    response.output_tokens,
                    reservation_id=reservation_id,
                )

            # Update metrics
            self._update_metrics_success(response, retry_stats, rate_limit_wait)

            # Build result
            total_time = (datetime.now() - start_time).total_seconds() * 1000

            result = CompletionResult(
                success=True,
                content=response.content,
                request_id=request_id,
                model_used=response.model_spec.model_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens(),
                estimated_cost_usd=response.estimated_cost_usd,
                latency_ms=response.latency_ms,
                total_time_ms=total_time,
                retry_stats=retry_stats,
                attempts=retry_stats.total_attempts,
                rate_limit_wait_ms=rate_limit_wait,
                raw_response=response,
            )

            if self.config.log_responses:
                logger.debug(
                    f"Response {request_id}: {response.output_tokens} tokens, "
                    f"{response.latency_ms:.0f}ms, ${response.estimated_cost_usd:.6f}"
                )

            return result

        except AuthenticationError as e:
            # Cancel reservation on failure (no tokens consumed)
            if self._rate_limiter and reservation_id:
                await self._rate_limiter.cancel_reservation(reservation_id)
            return self._create_error_result(
                request_id=request_id,
                error=str(e),
                error_type="AuthenticationError",
                start_time=start_time,
            )
        except ContextWindowError as e:
            if self._rate_limiter and reservation_id:
                await self._rate_limiter.cancel_reservation(reservation_id)
            return self._create_error_result(
                request_id=request_id,
                error=str(e),
                error_type="ContextWindowError",
                start_time=start_time,
            )
        except LLMError as e:
            if self._rate_limiter and reservation_id:
                await self._rate_limiter.cancel_reservation(reservation_id)
            return self._create_error_result(
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
                start_time=start_time,
            )
        except Exception as e:
            if self._rate_limiter and reservation_id:
                await self._rate_limiter.cancel_reservation(reservation_id)
            logger.exception(f"Unexpected error in request {request_id}")
            return self._create_error_result(
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
                start_time=start_time,
            )

    async def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        expected_fields: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompletionResult:
        """
        Make a completion request expecting JSON response.

        This method automatically:
        - Adds JSON formatting instructions to the system prompt
        - Parses the response with the robust JSON parser
        - Handles malformed JSON with multiple recovery strategies

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt (JSON instructions will be appended).
                SECURITY NOTE: The caller is responsible for sanitizing the system_prompt.
                User-controlled content should not be directly passed as system_prompt
                without validation, as it could alter the LLM's behavior (prompt injection).
            expected_fields: Fields to attempt extraction for if parsing fails
            max_tokens: Max response tokens
            temperature: Sampling temperature (recommend 0.0 for JSON)
            metadata: Optional metadata

        Returns:
            CompletionResult with parsed data in .data field
        """
        # Build JSON-aware system prompt
        json_system = system_prompt or ""
        json_system += (
            "\n\nIMPORTANT: You MUST respond with valid JSON only. "
            "Do not include any text before or after the JSON object. "
            "Do not wrap the JSON in markdown code blocks. "
            "Ensure all strings are properly escaped."
        )

        # Make the request
        result = await self.complete(
            prompt=prompt,
            system_prompt=json_system,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            metadata=metadata,
        )

        if not result.success:
            return result

        # Parse the JSON response
        parse_result = self._json_parser.parse(
            result.content,
            expected_fields=expected_fields,
        )

        # Update result with parse information
        result.parse_result = parse_result
        result.data = parse_result.data

        # Update metrics
        if parse_result.status == ParseStatus.SUCCESS:
            self._metrics.json_parse_successes += 1
        elif parse_result.status == ParseStatus.RECOVERED:
            self._metrics.json_parse_recovered += 1
            logger.debug(
                f"JSON recovered for {result.request_id}: "
                f"{parse_result.recovered_issues}"
            )
        elif parse_result.status == ParseStatus.PARTIAL:
            self._metrics.json_parse_recovered += 1
            logger.warning(
                f"JSON partial parse for {result.request_id}: "
                f"only got fields {list(parse_result.data.keys()) if parse_result.data else []}"
            )
        else:
            self._metrics.json_parse_failures += 1
            result.success = False
            result.error = f"JSON parse failed: {parse_result.errors}"
            result.error_type = "JSONParseError"
            logger.warning(
                f"JSON parse failed for {result.request_id}: {parse_result.errors}"
            )

        return result

    def _create_error_result(
        self,
        request_id: str,
        error: str,
        error_type: str,
        start_time: datetime,
    ) -> CompletionResult:
        """Create a failed CompletionResult."""
        self._metrics.total_requests += 1
        self._metrics.failed_requests += 1

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.warning(f"Request {request_id} failed: {error_type}: {error}")

        return CompletionResult(
            success=False,
            content="",
            request_id=request_id,
            total_time_ms=total_time,
            error=error,
            error_type=error_type,
        )

    def _update_metrics_success(
        self,
        response: LLMResponse,
        retry_stats: RetryStats,
        rate_limit_wait: float,
    ) -> None:
        """Update metrics after a successful request."""
        self._metrics.total_requests += 1
        self._metrics.successful_requests += 1

        self._metrics.total_input_tokens += response.input_tokens
        self._metrics.total_output_tokens += response.output_tokens

        if self.config.track_costs:
            self._metrics.total_cost_usd += response.estimated_cost_usd

        self._metrics.total_latency_ms += response.latency_ms
        self._metrics.total_retry_time_ms += retry_stats.total_delay_ms
        self._metrics.total_rate_limit_wait_ms += rate_limit_wait

        if retry_stats.total_attempts > 1:
            self._metrics.requests_retried += 1

    def get_metrics(self) -> ClientMetrics:
        """Get current session metrics."""
        return self._metrics

    async def get_rate_limit_status(self) -> Optional[RateLimitStatus]:
        """Get current rate limit status (async for thread safety)."""
        if self._rate_limiter:
            return await self._rate_limiter.get_status()
        return None

    def get_model_spec(self) -> ModelSpec:
        """Get the model specification for the current model."""
        return self._provider.get_model_spec()

    async def reset_rate_limits(self) -> None:
        """Reset rate limit state (useful for testing)."""
        if self._rate_limiter:
            await self._rate_limiter.reset()

    def reset_metrics(self) -> None:
        """Reset session metrics."""
        self._metrics = ClientMetrics()


# Convenience function for quick client creation
def create_client(
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> UnifiedLLMClient:
    """
    Create a client with common defaults.

    Args:
        model: Model ID to use
        api_key: API key (or use ANTHROPIC_API_KEY env var)

    Returns:
        Configured UnifiedLLMClient
    """
    return UnifiedLLMClient(LLMClientConfig(
        model=model,
        anthropic_api_key=api_key,
    ))
