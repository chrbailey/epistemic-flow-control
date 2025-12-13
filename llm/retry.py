"""
Retry handler with exponential backoff and jitter.

This module implements robust retry logic for LLM API calls, handling
transient failures gracefully while avoiding retry storms.

The algorithm is based on the "decorrelated jitter" approach from
AWS Architecture Blog, which provides better retry distribution than
simple exponential backoff.

Key Features:
- Exponential backoff with configurable parameters
- Jitter to prevent thundering herd
- Per-exception-type retry decisions
- Retry budget tracking (max retries, max total time)
- Detailed retry statistics for observability
"""

import asyncio
import random
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, TypeVar, Awaitable, Tuple, List
from datetime import datetime

from .providers.base import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    LLMError,
    RateLimitError,
    AuthenticationError,
    ContextWindowError,
    ContentFilterError,
    APIError,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    These defaults are tuned for LLM API calls:
    - Initial delay of 1s is reasonable for rate limits
    - Max delay of 60s prevents indefinite waiting
    - 3 attempts is enough for transient errors without wasting time on persistent ones
    - Jitter of 0.25 provides good distribution without too much variance
    """
    initial_delay_seconds: float = 1.0      # Starting delay
    max_delay_seconds: float = 60.0         # Cap on delay
    multiplier: float = 2.0                 # Exponential factor
    jitter_factor: float = 0.25             # +/- this fraction of delay
    max_attempts: int = 3                   # Total attempts (including first)
    max_total_time_seconds: float = 300.0   # Total time budget for all retries

    def __post_init__(self):
        """Validate configuration."""
        if self.initial_delay_seconds <= 0:
            raise ValueError("initial_delay_seconds must be positive")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("max_delay_seconds must be >= initial_delay_seconds")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""
    attempt_number: int
    started_at: datetime
    duration_ms: float
    succeeded: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    delay_before_ms: float = 0.0  # How long we waited before this attempt


@dataclass
class RetryStats:
    """Statistics from a retry sequence."""
    total_attempts: int = 0
    successful: bool = False
    total_time_ms: float = 0.0
    total_delay_ms: float = 0.0  # Time spent waiting between retries
    attempts: List[RetryAttempt] = field(default_factory=list)
    final_error: Optional[str] = None

    def add_attempt(self, attempt: RetryAttempt):
        """Add an attempt to the statistics."""
        self.attempts.append(attempt)
        self.total_attempts = len(self.attempts)
        if attempt.succeeded:
            self.successful = True
        else:
            self.final_error = attempt.error_message


# Exceptions that should NOT be retried
NON_RETRYABLE_EXCEPTIONS = (
    AuthenticationError,     # Won't succeed on retry
    ContextWindowError,      # Won't succeed without changing request
    ContentFilterError,      # Content issue, not transient
)


class RetryHandler:
    """
    Handles retry logic with exponential backoff and jitter.

    This class wraps API calls and automatically retries transient failures
    while respecting rate limits and avoiding retry storms.

    Usage:
        handler = RetryHandler()
        response = await handler.execute(provider, request)

    The handler will:
    1. Make the initial request
    2. If it fails with a retryable error, wait with exponential backoff
    3. Retry up to max_attempts times
    4. Return the successful response or raise the final error
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration. Uses defaults if not provided.
        """
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int, rate_limit_delay: Optional[float] = None) -> float:
        """
        Calculate delay before next retry.

        Uses decorrelated jitter algorithm:
        delay = min(cap, random_between(base, previous_delay * 3))

        This provides better distribution than simple exponential backoff
        and helps prevent retry storms when multiple clients are retrying.

        Args:
            attempt: Which attempt this is (1-indexed, so attempt 1 means first retry)
            rate_limit_delay: If set, use this instead (from Retry-After header)

        Returns:
            Delay in seconds
        """
        # If we have a rate limit delay from the server, respect it
        if rate_limit_delay is not None and rate_limit_delay > 0:
            # Add small jitter even to server-provided delays
            jitter = rate_limit_delay * 0.1 * random.random()
            return min(rate_limit_delay + jitter, self.config.max_delay_seconds)

        # Calculate base exponential delay
        base_delay = self.config.initial_delay_seconds * (
            self.config.multiplier ** (attempt - 1)
        )

        # Apply cap
        base_delay = min(base_delay, self.config.max_delay_seconds)

        # Apply jitter
        jitter_range = base_delay * self.config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        delay = base_delay + jitter

        # Ensure minimum delay
        return max(0.1, delay)

    def _should_retry(self, error: Exception, attempt: int, elapsed_time: float) -> Tuple[bool, str]:
        """
        Determine if we should retry after an error.

        Args:
            error: The exception that was raised
            attempt: Which attempt just failed (1-indexed)
            elapsed_time: Total time elapsed since first attempt

        Returns:
            (should_retry, reason) tuple
        """
        # Check if we've exhausted attempts
        if attempt >= self.config.max_attempts:
            return False, f"Max attempts ({self.config.max_attempts}) exceeded"

        # Check if we've exhausted time budget
        if elapsed_time >= self.config.max_total_time_seconds:
            return False, f"Time budget ({self.config.max_total_time_seconds}s) exceeded"

        # Check if exception type is retryable
        if isinstance(error, NON_RETRYABLE_EXCEPTIONS):
            return False, f"Non-retryable error type: {type(error).__name__}"

        # Rate limit errors are always retryable (with appropriate delay)
        if isinstance(error, RateLimitError):
            return True, "Rate limit error (retryable)"

        # Generic API errors: check if status suggests retry
        if isinstance(error, APIError):
            if error.is_retryable():
                return True, f"Retryable API error (status {error.status_code})"
            return False, f"Non-retryable API error (status {error.status_code})"

        # For unknown errors, retry cautiously
        return True, f"Unknown error type {type(error).__name__} (retrying)"

    async def execute(
        self,
        provider: BaseLLMProvider,
        request: LLMRequest,
        max_attempts: Optional[int] = None,
    ) -> Tuple[LLMResponse, RetryStats]:
        """
        Execute a request with retry logic.

        Args:
            provider: LLM provider to use
            request: Request to execute
            max_attempts: Override max attempts (optional)

        Returns:
            (response, stats) tuple

        Raises:
            The final error if all retries fail
        """
        effective_max_attempts = max_attempts or self.config.max_attempts
        stats = RetryStats()
        start_time = time.perf_counter()
        last_error: Optional[Exception] = None
        last_delay: float = 0

        for attempt in range(1, effective_max_attempts + 1):
            attempt_start = datetime.now()
            attempt_record = RetryAttempt(
                attempt_number=attempt,
                started_at=attempt_start,
                duration_ms=0,
                succeeded=False,
                delay_before_ms=last_delay * 1000,
            )

            try:
                # Make the request
                response = await provider.complete(request)

                # Success!
                attempt_record.succeeded = True
                attempt_record.duration_ms = (time.perf_counter() - start_time) * 1000 - stats.total_delay_ms
                stats.add_attempt(attempt_record)
                stats.total_time_ms = (time.perf_counter() - start_time) * 1000

                if attempt > 1:
                    logger.info(
                        f"Request {request.request_id} succeeded on attempt {attempt} "
                        f"after {stats.total_time_ms:.0f}ms total"
                    )

                return response, stats

            except Exception as e:
                last_error = e
                elapsed = time.perf_counter() - start_time

                # Record the failed attempt
                attempt_record.error_type = type(e).__name__
                attempt_record.error_message = str(e)[:200]  # Truncate long messages
                attempt_record.duration_ms = (time.perf_counter() - start_time) * 1000 - stats.total_delay_ms
                stats.add_attempt(attempt_record)

                # Check if we should retry
                should_retry, reason = self._should_retry(e, attempt, elapsed)

                if not should_retry:
                    logger.warning(
                        f"Request {request.request_id} failed on attempt {attempt}, "
                        f"not retrying: {reason}"
                    )
                    stats.total_time_ms = (time.perf_counter() - start_time) * 1000
                    raise

                # Calculate delay for next attempt
                rate_limit_delay = None
                if isinstance(e, RateLimitError) and e.retry_after:
                    rate_limit_delay = e.retry_after

                delay = self._calculate_delay(attempt, rate_limit_delay)
                last_delay = delay

                logger.warning(
                    f"Request {request.request_id} failed on attempt {attempt}/{effective_max_attempts} "
                    f"with {type(e).__name__}: {str(e)[:100]}. "
                    f"Retrying in {delay:.2f}s"
                )

                # Wait before retry
                await asyncio.sleep(delay)
                stats.total_delay_ms += delay * 1000

        # If we get here, we've exhausted all retries
        stats.total_time_ms = (time.perf_counter() - start_time) * 1000

        if last_error:
            raise last_error
        else:
            raise RuntimeError("Retry loop exited without error or response")


# Convenience function for simple retry wrapping
async def with_retry(
    func: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None,
    max_attempts: Optional[int] = None,
) -> T:
    """
    Execute an async function with retry logic.

    This is a simpler interface for when you don't need the full
    RetryHandler capabilities.

    Usage:
        result = await with_retry(
            lambda: client.complete(request),
            max_attempts=3
        )

    Args:
        func: Async function to execute
        config: Retry configuration
        max_attempts: Override max attempts

    Returns:
        Result of the function

    Raises:
        The final error if all retries fail
    """
    cfg = config or RetryConfig()
    effective_max = max_attempts or cfg.max_attempts

    last_error = None

    for attempt in range(1, effective_max + 1):
        try:
            return await func()
        except NON_RETRYABLE_EXCEPTIONS:
            raise
        except Exception as e:
            last_error = e
            if attempt >= effective_max:
                raise

            delay = cfg.initial_delay_seconds * (cfg.multiplier ** (attempt - 1))
            delay = min(delay, cfg.max_delay_seconds)
            delay *= (1 + cfg.jitter_factor * (random.random() - 0.5))

            await asyncio.sleep(delay)

    raise last_error or RuntimeError("Retry loop exited unexpectedly")
