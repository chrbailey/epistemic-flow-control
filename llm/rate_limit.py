"""
Rate limiter for LLM API calls.

This module implements token bucket rate limiting with support for
both request-based and token-based limits, matching the rate limit
models used by Anthropic and OpenAI.

Rate Limiting Strategy:
- Requests per minute (RPM): Prevent too many API calls
- Tokens per minute (TPM): Prevent exceeding token quotas
- Daily limits: Budget constraints for cost control

The token bucket algorithm provides smooth rate limiting without
hard cutoffs, allowing bursts when capacity is available while
preventing sustained overuse.

Usage:
    limiter = RateLimiter(RateLimitConfig(
        requests_per_minute=60,
        tokens_per_minute=100_000,
    ))

    # Before making a request
    wait_time = await limiter.acquire(estimated_tokens=1000)
    if wait_time > 0:
        await asyncio.sleep(wait_time)

    # After request completes
    await limiter.record_usage(input_tokens=500, output_tokens=200)
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Deque
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types of rate limits that can be exceeded."""
    REQUESTS_PER_MINUTE = "rpm"
    TOKENS_PER_MINUTE = "tpm"
    REQUESTS_PER_DAY = "rpd"
    TOKENS_PER_DAY = "tpd"


@dataclass
class RateLimitConfig:
    """
    Rate limiting configuration.

    These defaults are conservative and work within free tier limits.
    Adjust based on your API tier.

    Anthropic limits vary by tier:
    - Free: ~5 RPM, ~20K TPM
    - Tier 1: ~60 RPM, ~80K TPM
    - Tier 2+: Higher limits

    OpenAI limits also vary by tier and model.
    """
    # Per-minute limits
    requests_per_minute: int = 50
    tokens_per_minute: int = 80_000

    # Per-day limits (for budget control)
    requests_per_day: int = 10_000
    tokens_per_day: int = 2_000_000

    # Behavior
    max_wait_seconds: float = 60.0      # Max time to wait for capacity
    burst_allowance: float = 1.2        # Allow 20% burst above sustained rate


@dataclass
class RateLimitState:
    """Current state of rate limiting."""
    # Sliding window for per-minute tracking
    minute_requests: Deque[float] = field(default_factory=deque)
    minute_tokens: Deque[tuple] = field(default_factory=deque)  # (timestamp, count)

    # Daily counters
    day_requests: int = 0
    day_tokens: int = 0
    day_start: datetime = field(default_factory=datetime.now)

    # Tracking for observability
    total_wait_time_ms: float = 0.0
    total_requests_limited: int = 0


@dataclass
class RateLimitStatus:
    """Current rate limit status for reporting."""
    requests_this_minute: int
    tokens_this_minute: int
    requests_today: int
    tokens_today: int

    rpm_remaining: int
    tpm_remaining: int
    rpd_remaining: int
    tpd_remaining: int

    is_limited: bool
    limiting_factor: Optional[RateLimitType]
    estimated_wait_seconds: float


class RateLimitExceeded(Exception):
    """Raised when rate limit would be exceeded and max_wait is exceeded."""
    def __init__(
        self,
        message: str,
        limit_type: RateLimitType,
        retry_after: float,
    ):
        super().__init__(message)
        self.limit_type = limit_type
        self.retry_after = retry_after


class RateLimiter:
    """
    Token bucket rate limiter with per-minute and per-day limits.

    This class tracks API usage and provides wait times to stay
    within configured limits. It's async-safe and supports
    concurrent callers.

    The implementation uses a sliding window for per-minute limits
    (more accurate than fixed windows) and simple counters for
    daily limits.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration. Uses defaults if not provided.
        """
        self.config = config or RateLimitConfig()
        self.state = RateLimitState()
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        estimated_tokens: int = 0,
        block: bool = True,
    ) -> float:
        """
        Acquire permission to make a request.

        This should be called before making an API request. It checks
        all rate limits and returns the time to wait (if any).

        Args:
            estimated_tokens: Estimated tokens for this request
            block: If True, wait for capacity. If False, raise if limited.

        Returns:
            Seconds to wait before making request (0 if immediate)

        Raises:
            RateLimitExceeded: If block=False and limits would be exceeded
        """
        async with self._lock:
            self._cleanup_old_entries()
            self._check_day_rollover()

            wait_time, limit_type = self._calculate_wait_time(estimated_tokens)

            if wait_time > 0:
                if not block:
                    raise RateLimitExceeded(
                        f"Rate limit would be exceeded: {limit_type.value}",
                        limit_type=limit_type,
                        retry_after=wait_time,
                    )

                if wait_time > self.config.max_wait_seconds:
                    raise RateLimitExceeded(
                        f"Would need to wait {wait_time:.1f}s, exceeds max {self.config.max_wait_seconds}s",
                        limit_type=limit_type,
                        retry_after=wait_time,
                    )

                self.state.total_wait_time_ms += wait_time * 1000
                self.state.total_requests_limited += 1

                logger.debug(
                    f"Rate limited ({limit_type.value}), waiting {wait_time:.2f}s"
                )

            return wait_time

    async def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """
        Record actual token usage after request completes.

        This updates our tracking with the actual tokens used,
        which may differ from the estimate provided to acquire().

        Args:
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens used
        """
        async with self._lock:
            now = time.time()
            total_tokens = input_tokens + output_tokens

            # Record in sliding windows
            self.state.minute_requests.append(now)
            self.state.minute_tokens.append((now, total_tokens))

            # Update daily counters
            self.state.day_requests += 1
            self.state.day_tokens += total_tokens

    def get_status(self) -> RateLimitStatus:
        """
        Get current rate limit status.

        Useful for monitoring and debugging.
        """
        self._cleanup_old_entries()
        self._check_day_rollover()

        requests_this_minute = len(self.state.minute_requests)
        tokens_this_minute = sum(t[1] for t in self.state.minute_tokens)

        wait_time, limit_type = self._calculate_wait_time(0)

        return RateLimitStatus(
            requests_this_minute=requests_this_minute,
            tokens_this_minute=tokens_this_minute,
            requests_today=self.state.day_requests,
            tokens_today=self.state.day_tokens,
            rpm_remaining=max(0, self.config.requests_per_minute - requests_this_minute),
            tpm_remaining=max(0, self.config.tokens_per_minute - tokens_this_minute),
            rpd_remaining=max(0, self.config.requests_per_day - self.state.day_requests),
            tpd_remaining=max(0, self.config.tokens_per_day - self.state.day_tokens),
            is_limited=wait_time > 0,
            limiting_factor=limit_type if wait_time > 0 else None,
            estimated_wait_seconds=wait_time,
        )

    def _cleanup_old_entries(self) -> None:
        """Remove entries older than 1 minute from sliding windows."""
        now = time.time()
        minute_ago = now - 60

        # Clean request timestamps
        while self.state.minute_requests and self.state.minute_requests[0] < minute_ago:
            self.state.minute_requests.popleft()

        # Clean token records
        while self.state.minute_tokens and self.state.minute_tokens[0][0] < minute_ago:
            self.state.minute_tokens.popleft()

    def _check_day_rollover(self) -> None:
        """Reset daily counters if day has changed."""
        now = datetime.now()
        if now.date() != self.state.day_start.date():
            logger.info(
                f"Day rollover: {self.state.day_requests} requests, "
                f"{self.state.day_tokens} tokens yesterday"
            )
            self.state.day_requests = 0
            self.state.day_tokens = 0
            self.state.day_start = now

    def _calculate_wait_time(
        self,
        estimated_tokens: int,
    ) -> tuple[float, Optional[RateLimitType]]:
        """
        Calculate how long to wait before making a request.

        Returns (wait_seconds, limiting_factor).
        """
        wait_time = 0.0
        limit_type = None

        # Check requests per minute
        current_rpm = len(self.state.minute_requests)
        effective_rpm_limit = int(self.config.requests_per_minute * self.config.burst_allowance)

        if current_rpm >= effective_rpm_limit:
            # Need to wait for oldest request to fall out of window
            oldest = self.state.minute_requests[0]
            rpm_wait = 60 - (time.time() - oldest) + 0.1  # Small buffer
            if rpm_wait > wait_time:
                wait_time = rpm_wait
                limit_type = RateLimitType.REQUESTS_PER_MINUTE

        # Check tokens per minute
        current_tpm = sum(t[1] for t in self.state.minute_tokens)
        effective_tpm_limit = int(self.config.tokens_per_minute * self.config.burst_allowance)

        if current_tpm + estimated_tokens > effective_tpm_limit:
            # Estimate when enough tokens will fall out of window
            tokens_needed = current_tpm + estimated_tokens - effective_tpm_limit

            # Find when enough tokens will expire
            cumulative = 0
            for ts, tokens in self.state.minute_tokens:
                cumulative += tokens
                if cumulative >= tokens_needed:
                    tpm_wait = 60 - (time.time() - ts) + 0.1
                    if tpm_wait > wait_time:
                        wait_time = tpm_wait
                        limit_type = RateLimitType.TOKENS_PER_MINUTE
                    break

        # Check daily limits (hard limits, no burst)
        if self.state.day_requests >= self.config.requests_per_day:
            # Would need to wait until tomorrow
            now = datetime.now()
            tomorrow = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            day_wait = (tomorrow - now).total_seconds()
            if day_wait > wait_time:
                wait_time = day_wait
                limit_type = RateLimitType.REQUESTS_PER_DAY

        if self.state.day_tokens + estimated_tokens > self.config.tokens_per_day:
            now = datetime.now()
            tomorrow = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            day_wait = (tomorrow - now).total_seconds()
            if day_wait > wait_time:
                wait_time = day_wait
                limit_type = RateLimitType.TOKENS_PER_DAY

        return max(0.0, wait_time), limit_type

    async def reset(self) -> None:
        """Reset all rate limit state. Useful for testing."""
        async with self._lock:
            self.state = RateLimitState()


class MultiProviderRateLimiter:
    """
    Rate limiter that tracks limits per provider.

    Use this when you have multiple LLM providers with different
    rate limits.
    """

    def __init__(self, provider_configs: Optional[Dict[str, RateLimitConfig]] = None):
        """
        Initialize with per-provider configs.

        Args:
            provider_configs: Dict of provider_name -> RateLimitConfig
        """
        self.provider_configs = provider_configs or {}
        self._limiters: Dict[str, RateLimiter] = {}

    def get_limiter(self, provider: str) -> RateLimiter:
        """Get or create limiter for a provider."""
        if provider not in self._limiters:
            config = self.provider_configs.get(provider, RateLimitConfig())
            self._limiters[provider] = RateLimiter(config)
        return self._limiters[provider]

    async def acquire(
        self,
        provider: str,
        estimated_tokens: int = 0,
        block: bool = True,
    ) -> float:
        """Acquire permission for a specific provider."""
        limiter = self.get_limiter(provider)
        return await limiter.acquire(estimated_tokens, block)

    async def record_usage(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record usage for a specific provider."""
        limiter = self.get_limiter(provider)
        await limiter.record_usage(input_tokens, output_tokens)
