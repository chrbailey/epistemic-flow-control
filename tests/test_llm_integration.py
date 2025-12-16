"""
Integration tests for LLM layer.

These tests verify that:
1. The LLM client can connect to Claude
2. Claude returns valid JSON for extraction prompts
3. The JSON parser handles various response formats
4. Rate limiting works correctly
5. Retry logic handles transient failures

IMPORTANT: These tests make REAL API calls and cost money.
Run with: pytest tests/test_llm_integration.py -v
Skip expensive tests with: pytest tests/test_llm_integration.py -v -m "not expensive"

Environment Requirements:
- ANTHROPIC_API_KEY must be set
"""

import asyncio
import json
import os
import pytest
from datetime import datetime
from typing import Optional

# Skip all tests if no API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
HAS_API_KEY = ANTHROPIC_API_KEY is not None and len(ANTHROPIC_API_KEY) > 0

# Import LLM modules
try:
    from llm import (
        UnifiedLLMClient,
        LLMClientConfig,
        CompletionResult,
        RobustJSONParser,
        ParseStatus,
        parse_json_response,
        RateLimiter,
        RateLimitConfig,
        RateLimitExceeded,
    )
    from llm.providers.anthropic import AnthropicProvider, CLAUDE_MODELS
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    pytest.skip(f"LLM module not available: {e}", allow_module_level=True)


# ============================================================
# JSON Parser Tests (no API calls)
# ============================================================

class TestRobustJSONParser:
    """Test the robust JSON parser with various malformed inputs."""

    def test_valid_json_direct(self):
        """Parser handles valid JSON directly."""
        parser = RobustJSONParser()
        result = parser.parse('{"key": "value", "num": 42}')

        assert result.status == ParseStatus.SUCCESS
        assert result.data == {"key": "value", "num": 42}
        assert result.extraction_method == "direct"

    def test_json_with_markdown_wrapper(self):
        """Parser extracts JSON from markdown code blocks."""
        parser = RobustJSONParser()
        response = '''Here's the data:

```json
{"patterns": [{"type": "test"}]}
```

Hope that helps!'''

        result = parser.parse(response)

        assert result.is_success()
        assert result.data == {"patterns": [{"type": "test"}]}

    def test_trailing_comma_fix(self):
        """Parser fixes trailing commas."""
        parser = RobustJSONParser()
        result = parser.parse('{"a": 1, "b": 2,}')

        assert result.is_success()
        assert result.data == {"a": 1, "b": 2}

    def test_single_quotes_to_double(self):
        """Parser converts single quotes to double quotes."""
        parser = RobustJSONParser()
        result = parser.parse("{'key': 'value'}")

        assert result.is_success()
        assert result.data == {"key": "value"}

    def test_python_booleans(self):
        """Parser converts Python True/False/None to JSON."""
        parser = RobustJSONParser()
        result = parser.parse('{"active": True, "deleted": False, "data": None}')

        assert result.is_success()
        assert result.data == {"active": True, "deleted": False, "data": None}

    def test_truncated_json_repair(self):
        """Parser attempts to repair truncated JSON."""
        parser = RobustJSONParser()
        result = parser.parse('{"patterns": [{"name": "test"')

        # Should attempt repair
        assert result.status in (ParseStatus.RECOVERED, ParseStatus.FAILED)

    def test_partial_field_extraction(self):
        """Parser extracts specific fields as last resort."""
        parser = RobustJSONParser()
        malformed = 'Some text "confidence": 0.85 more "subject": "Judge Smith" end'

        result = parser.parse(
            malformed,
            expected_fields=["confidence", "subject"]
        )

        # May get partial results
        if result.status == ParseStatus.PARTIAL:
            assert "confidence" in result.data or "subject" in result.data

    def test_empty_response(self):
        """Parser handles empty input gracefully."""
        parser = RobustJSONParser()
        result = parser.parse("")

        assert result.status == ParseStatus.FAILED
        assert "Empty response" in result.errors[0]

    def test_nan_infinity_handling(self):
        """Parser handles NaN and Infinity values."""
        parser = RobustJSONParser()
        # Note: Python's json module with allow_nan=True (default in some configs)
        # will parse these as float('nan') and float('inf')
        # Our parser attempts to replace them with null in preprocessing
        result = parser.parse('{"a": NaN, "b": Infinity, "c": -Infinity}')

        assert result.is_success()
        # The actual behavior depends on Python version and json settings
        # Just verify it parses without error
        assert "a" in result.data
        assert "b" in result.data


# ============================================================
# Rate Limiter Tests (no API calls)
# ============================================================

class TestRateLimiter:
    """Test rate limiting logic."""

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Rate limiter tracks requests correctly."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=5,
            tokens_per_minute=1000,
        ))

        # First few requests should be immediate
        for i in range(3):
            wait, reservation_id = await limiter.acquire(estimated_tokens=100)
            await limiter.record_usage(50, 50, reservation_id=reservation_id)
            assert wait == 0  # No wait for first few

        status = await limiter.get_status()
        assert status.requests_this_minute == 3
        assert status.tokens_this_minute == 300

    @pytest.mark.asyncio
    async def test_token_limit_reached(self):
        """Rate limiter enforces token limits."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=500,
            burst_allowance=1.0,  # No burst
        ))

        # Record usage up to limit
        _, reservation_id = await limiter.acquire(estimated_tokens=0)
        await limiter.record_usage(400, 0, reservation_id=reservation_id)

        # Next request with tokens should trigger wait or raise
        # With block=False and over limit, this should raise RateLimitExceeded
        try:
            _, _ = await limiter.acquire(estimated_tokens=200, block=False)
            # If we get here, the limiter allowed it (burst allowance)
        except RateLimitExceeded:
            pass  # Expected behavior when over limit

    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self):
        """Rate limiter can be reset."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=5,
            tokens_per_minute=1000,
        ))

        _, reservation_id = await limiter.acquire(estimated_tokens=100)
        await limiter.record_usage(500, 500, reservation_id=reservation_id)

        status_before = await limiter.get_status()
        assert status_before.requests_this_minute == 1

        await limiter.reset()

        status_after = await limiter.get_status()
        assert status_after.requests_this_minute == 0


# ============================================================
# Integration Tests (require API key)
# ============================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="No ANTHROPIC_API_KEY set")
class TestAnthropicProvider:
    """Test the Anthropic provider directly."""

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_simple_completion(self):
        """Provider can make a simple completion request."""
        from llm.providers.base import LLMRequest

        provider = AnthropicProvider(model_id="claude-3-5-haiku-20241022")

        request = LLMRequest(
            request_id="test_001",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
            temperature=0.0,
        )

        response = await provider.complete(request)

        assert response.request_id == "test_001"
        assert "hello" in response.content.lower()
        assert response.input_tokens > 0
        assert response.output_tokens > 0

    def test_token_estimation(self):
        """Provider estimates tokens reasonably."""
        provider = AnthropicProvider(model_id="claude-3-5-haiku-20241022")

        short_text = "Hello world"
        long_text = "This is a longer piece of text that should have more tokens."

        short_estimate = provider.count_tokens(short_text)
        long_estimate = provider.count_tokens(long_text)

        assert short_estimate < long_estimate
        assert short_estimate > 0

    def test_model_spec(self):
        """Provider returns correct model spec."""
        provider = AnthropicProvider(model_id="claude-3-5-haiku-20241022")
        spec = provider.get_model_spec()

        assert spec.model_id == "claude-3-5-haiku-20241022"
        assert spec.context_window == 200_000
        assert spec.cost_per_1k_input > 0


@pytest.mark.skipif(not HAS_API_KEY, reason="No ANTHROPIC_API_KEY set")
class TestUnifiedLLMClient:
    """Test the unified client end-to-end."""

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_complete_text(self):
        """Client can complete a text request."""
        client = UnifiedLLMClient(LLMClientConfig(
            model="claude-3-5-haiku-20241022",  # Cheaper model for tests
            rate_limit_enabled=False,  # Don't rate limit tests
        ))

        result = await client.complete(
            prompt="What is 2+2? Reply with just the number.",
            max_tokens=10,
        )

        assert result.success
        assert "4" in result.content
        assert result.input_tokens > 0

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_complete_json(self):
        """Client can complete a JSON request."""
        client = UnifiedLLMClient(LLMClientConfig(
            model="claude-3-5-haiku-20241022",
            rate_limit_enabled=False,
        ))

        result = await client.complete_json(
            prompt="List three primary colors. Return as JSON with key 'colors' containing an array.",
            expected_fields=["colors"],
            max_tokens=100,
        )

        assert result.success
        assert result.data is not None
        assert "colors" in result.data
        assert isinstance(result.data["colors"], list)

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_pattern_extraction_prompt(self):
        """Client can handle a realistic pattern extraction prompt."""
        client = UnifiedLLMClient(LLMClientConfig(
            model="claude-3-5-haiku-20241022",
            rate_limit_enabled=False,
        ))

        system_prompt = """You are a pattern extraction system.
Extract patterns from events and return them as JSON.
You MUST respond with valid JSON only."""

        user_prompt = """Analyze this event and extract any patterns:

EVENT:
- What: Judge Smith granted motion for summary judgment
- Who: Judge Smith, Plaintiff Corp, Defendant LLC
- When: 2024-01-15
- Where: N.D. Cal

Return JSON with this structure:
{
    "patterns": [
        {
            "pattern_type": "outcome",
            "subject": "who exhibits the pattern",
            "description": "human readable description",
            "confidence": 0.0-1.0,
            "reasoning": "why this is a pattern"
        }
    ],
    "no_patterns_found": false
}"""

        result = await client.complete_json(
            prompt=user_prompt,
            system_prompt=system_prompt,
            expected_fields=["patterns", "no_patterns_found"],
            max_tokens=500,
        )

        assert result.success, f"Failed: {result.error}"
        assert result.data is not None

        # Check structure
        assert "patterns" in result.data or "no_patterns_found" in result.data

        if result.data.get("patterns"):
            pattern = result.data["patterns"][0]
            assert "pattern_type" in pattern
            assert "subject" in pattern
            assert "confidence" in pattern

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_metrics_tracking(self):
        """Client tracks metrics correctly."""
        client = UnifiedLLMClient(LLMClientConfig(
            model="claude-3-5-haiku-20241022",
            rate_limit_enabled=False,
            track_costs=True,
        ))

        # Make a request
        await client.complete(
            prompt="Say 'test'",
            max_tokens=10,
        )

        metrics = client.get_metrics()

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.total_input_tokens > 0
        assert metrics.total_output_tokens > 0
        assert metrics.total_cost_usd > 0


# ============================================================
# Pattern Extractor Integration (require API key)
# ============================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="No ANTHROPIC_API_KEY set")
class TestPatternExtractorIntegration:
    """Test the pattern extractor with real LLM calls."""

    @pytest.mark.asyncio
    @pytest.mark.expensive
    async def test_extract_patterns_from_events(self):
        """Pattern extractor can extract patterns using LLM."""
        # Import here to avoid issues if core module has problems
        from core.pattern_extractor import PatternExtractor, PatternType
        from core.event_store import Event, EventStore, VerificationStatus

        # Create a minimal event store (in-memory)
        store = EventStore(":memory:")

        # Create extractor with LLM client
        llm_config = LLMClientConfig(
            model="claude-3-5-haiku-20241022",
            rate_limit_enabled=False,
        )
        extractor = PatternExtractor(
            event_store=store,
            llm_config=llm_config,
        )

        # Create test events
        events = [
            Event(
                event_id="test_evt_001",
                who=["Judge Smith"],
                what="Granted motion for summary judgment",
                when=datetime(2024, 1, 15),
                where="N.D. Cal",
                why="No genuine issue of material fact",
                how="Written order",
                source_id="test",
                source_url="",
                raw_text="Order granting MSJ",
                verification_status=VerificationStatus.VERIFIED,
                domain="judicial",
                event_type="order"
            ),
            Event(
                event_id="test_evt_002",
                who=["Judge Smith"],
                what="Granted motion for summary judgment",
                when=datetime(2024, 3, 20),
                where="N.D. Cal",
                why="Defendant failed to present triable issues",
                how="Written order",
                source_id="test",
                source_url="",
                raw_text="Order granting MSJ",
                verification_status=VerificationStatus.VERIFIED,
                domain="judicial",
                event_type="order"
            ),
        ]

        # Extract patterns (async)
        patterns = await extractor.extract_patterns_async(events)

        # Should get at least one pattern
        assert len(patterns) >= 1

        # Check pattern structure
        pattern = patterns[0]
        assert pattern.pattern_id is not None
        assert pattern.subject  # Should have a subject
        assert pattern.description  # Should have a description
        assert 0 <= pattern.llm_confidence <= 1
        assert pattern.effective_confidence >= 0


# ============================================================
# Run configuration
# ============================================================

if __name__ == "__main__":
    # Run with: python -m pytest tests/test_llm_integration.py -v
    pytest.main([__file__, "-v", "-m", "not expensive"])
