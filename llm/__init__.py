"""
LLM Integration Layer for Epistemic Flow Control

This module provides a robust, production-ready interface to LLM providers
(Claude, OpenAI) with:
- Automatic retry with exponential backoff
- Rate limiting
- Robust JSON parsing with error recovery
- Cost tracking
- Comprehensive logging

Usage:
    from llm import UnifiedLLMClient, LLMClientConfig

    client = UnifiedLLMClient(LLMClientConfig(
        anthropic_api_key="sk-...",
    ))

    result = await client.complete_json(
        prompt="Extract patterns from: ...",
        system_prompt="You are a pattern extractor...",
    )

    if result.success:
        patterns = result.data
"""

from .client import (
    UnifiedLLMClient,
    LLMClientConfig,
    CompletionResult,
    ClientMetrics,
    create_client,
)
from .providers.base import (
    LLMRequest,
    LLMResponse,
    ModelSpec,
    ModelProvider,
    LLMError,
    RateLimitError,
    AuthenticationError,
    ContextWindowError,
    ContentFilterError,
    APIError,
)
from .json_parser import (
    RobustJSONParser,
    ParseResult,
    ParseStatus,
    parse_json_response,
)
from .retry import (
    RetryHandler,
    RetryConfig,
    RetryStats,
    with_retry,
)
from .rate_limit import (
    RateLimiter,
    RateLimitConfig,
    RateLimitStatus,
    RateLimitExceeded,
)

__all__ = [
    # Main client
    "UnifiedLLMClient",
    "LLMClientConfig",
    "CompletionResult",
    "ClientMetrics",
    "create_client",
    # Provider interfaces
    "LLMRequest",
    "LLMResponse",
    "ModelSpec",
    "ModelProvider",
    # Exceptions
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ContextWindowError",
    "ContentFilterError",
    "APIError",
    # JSON parsing
    "RobustJSONParser",
    "ParseResult",
    "ParseStatus",
    "parse_json_response",
    # Retry
    "RetryHandler",
    "RetryConfig",
    "RetryStats",
    "with_retry",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStatus",
    "RateLimitExceeded",
]
