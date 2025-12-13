"""
Anthropic Claude provider implementation.

This module implements the BaseLLMProvider interface for Anthropic's Claude models.
It handles all the Claude-specific API details while presenting a standardized
interface to the rest of the system.

Key Features:
- Async API calls using the official anthropic SDK
- Automatic token counting estimation
- Cost calculation based on current pricing
- Proper error translation to our exception types
- Support for Claude's prompt caching (when available)

Usage:
    provider = AnthropicProvider(
        api_key="sk-ant-...",
        model_id="claude-sonnet-4-20250514"
    )
    response = await provider.complete(request)
"""

import os
import time
import asyncio
import logging
from typing import Optional, Dict, Any

from .base import (
    BaseLLMProvider,
    ModelSpec,
    ModelProvider,
    LLMRequest,
    LLMResponse,
    RateLimitError,
    AuthenticationError,
    ContextWindowError,
    ContentFilterError,
    APIError,
)

logger = logging.getLogger(__name__)

# Model specifications for known Claude models
# These are frozen/immutable - update this dict when new models release
CLAUDE_MODELS: Dict[str, ModelSpec] = {
    "claude-sonnet-4-20250514": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        version="2025-05-14",
        context_window=200_000,
        max_output_tokens=16_000,
        supports_json_mode=False,  # Claude uses prompting, not mode flag
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

# Default model if not specified
DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider implementation.

    This class handles all communication with Anthropic's API, translating
    between our standardized request/response format and Claude's API format.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        max_retries: int = 0,  # We handle retries at a higher level
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            model_id: Model to use. Must be a key in CLAUDE_MODELS.
            timeout: Request timeout in seconds.
            max_retries: SDK-level retries (we typically handle retries ourselves).
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise AuthenticationError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._model_id = model_id
        self._timeout = timeout
        self._max_retries = max_retries

        # Validate model
        if model_id not in CLAUDE_MODELS:
            available = ", ".join(CLAUDE_MODELS.keys())
            raise ValueError(
                f"Unknown model: {model_id}. Available models: {available}"
            )

        self._model_spec = CLAUDE_MODELS[model_id]

        # Lazy-load the client (avoids import errors if anthropic not installed)
        self._client = None
        self._async_client = None

    def _get_async_client(self):
        """Lazy-load the async Anthropic client."""
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Run: pip install anthropic"
                )

            self._async_client = AsyncAnthropic(
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )

        return self._async_client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Execute a completion request against Claude.

        Translates our standardized request to Claude's format, makes the API call,
        and translates the response back to our standardized format.

        Args:
            request: Standardized LLMRequest

        Returns:
            Standardized LLMResponse

        Raises:
            RateLimitError: If rate limited (429)
            AuthenticationError: If auth fails (401)
            ContextWindowError: If request too large
            ContentFilterError: If content filtered
            APIError: For other API errors
        """
        client = self._get_async_client()
        start_time = time.perf_counter()

        # Build messages in Anthropic format
        anthropic_messages = []
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Anthropic only accepts "user" and "assistant" roles in messages
            if role == "system":
                # System messages should go in system parameter, not messages
                continue

            anthropic_messages.append({
                "role": role,
                "content": content,
            })

        # Build system prompt
        system = request.system_prompt or ""
        if request.json_mode:
            # Claude doesn't have a JSON mode flag, so we enforce via prompt
            system += (
                "\n\nIMPORTANT: You MUST respond with valid JSON only. "
                "Do not include any text before or after the JSON. "
                "Do not wrap the JSON in markdown code blocks."
            )

        # Make the API call
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
            # Translate Anthropic exceptions to our types
            self._translate_exception(e)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract response content
        content = ""
        if response.content and len(response.content) > 0:
            # Claude returns a list of content blocks
            content = response.content[0].text

        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        estimated_cost = self._model_spec.estimate_cost(input_tokens, output_tokens)

        # Build standardized response
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
                "stop_sequence": response.stop_sequence,
            },
        )

    def _translate_stop_reason(self, stop_reason: Optional[str]) -> str:
        """Translate Claude's stop_reason to our standardized format."""
        if stop_reason is None:
            return "unknown"

        translation = {
            "end_turn": "end_turn",
            "max_tokens": "max_tokens",
            "stop_sequence": "stop_sequence",
            "tool_use": "tool_use",
        }

        return translation.get(stop_reason, stop_reason)

    def _translate_exception(self, e: Exception) -> None:
        """
        Translate Anthropic SDK exceptions to our exception types.

        This allows the retry handler to make informed decisions about
        whether to retry and how long to wait.
        """
        try:
            from anthropic import (
                RateLimitError as AnthropicRateLimitError,
                AuthenticationError as AnthropicAuthError,
                BadRequestError,
                APIStatusError,
                APITimeoutError,
                APIConnectionError,
            )
        except ImportError:
            # If we can't import, just re-raise
            raise APIError(str(e)) from e

        if isinstance(e, AnthropicRateLimitError):
            # Extract retry-after if available
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
            error_msg = str(e).lower()
            if "context" in error_msg or "token" in error_msg:
                raise ContextWindowError(str(e)) from e
            raise APIError(str(e), status_code=400) from e

        elif isinstance(e, APITimeoutError):
            raise APIError(f"Request timed out: {e}", status_code=408) from e

        elif isinstance(e, APIConnectionError):
            raise APIError(f"Connection error: {e}", status_code=503) from e

        elif isinstance(e, APIStatusError):
            status_code = getattr(e, 'status_code', None)
            raise APIError(str(e), status_code=status_code) from e

        else:
            # Unknown exception type
            raise APIError(f"Unknown error: {e}") from e

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Claude uses a BPE tokenizer similar to GPT. We use a simple heuristic
        that's reasonably accurate for English text: ~4 characters per token.

        For production use with exact counts, you'd use the anthropic tokenizer,
        but that requires the full SDK and is slower.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Heuristic: ~4 characters per token for English
        # This is conservative (slightly overestimates)
        char_count = len(text)
        estimated = max(1, char_count // 4)

        # Adjust for whitespace-heavy content (fewer tokens)
        whitespace_ratio = text.count(' ') / max(1, char_count)
        if whitespace_ratio > 0.2:
            estimated = int(estimated * 0.9)

        return estimated

    def get_model_spec(self) -> ModelSpec:
        """Get the specification for the current model."""
        return self._model_spec

    @property
    def provider_name(self) -> str:
        """Human-readable provider name."""
        return "Anthropic Claude"

    def validate_request(self, request: LLMRequest) -> list[str]:
        """
        Validate a request before sending.

        Adds Claude-specific validation on top of base validation.
        """
        errors = super().validate_request(request)

        # Claude requires at least one message
        if not request.messages:
            errors.append("Claude requires at least one message")

        # Claude doesn't allow empty content
        for i, msg in enumerate(request.messages):
            if not msg.get("content"):
                errors.append(f"Message {i} has empty content")

        # Validate message roles
        valid_roles = {"user", "assistant", "system"}
        for i, msg in enumerate(request.messages):
            role = msg.get("role", "")
            if role not in valid_roles:
                errors.append(
                    f"Message {i} has invalid role '{role}'. "
                    f"Valid roles: {valid_roles}"
                )

        return errors


# Convenience function for quick provider creation
def create_anthropic_provider(
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> AnthropicProvider:
    """
    Create an Anthropic provider with common defaults.

    Args:
        model: Model ID to use
        api_key: API key (or use ANTHROPIC_API_KEY env var)

    Returns:
        Configured AnthropicProvider
    """
    return AnthropicProvider(api_key=api_key, model_id=model)
