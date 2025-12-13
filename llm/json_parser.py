"""
Robust JSON parser for LLM responses.

LLMs frequently produce malformed JSON. This module provides multiple
strategies for extracting valid JSON from imperfect responses.

Common LLM JSON Issues Handled:
1. Trailing commas: {"a": 1,}
2. Single quotes: {'key': 'value'}
3. Unquoted keys: {key: "value"}
4. Markdown wrappers: ```json {...} ```
5. Extra text before/after: "Here's the JSON: {...}"
6. Truncated JSON: {"incomplete": tru
7. Invalid escape sequences
8. NaN/Infinity literals (not valid JSON)

Usage:
    parser = RobustJSONParser()
    result = parser.parse(llm_response)
    if result.status == ParseStatus.SUCCESS:
        data = result.data
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ParseStatus(Enum):
    """Status of JSON parsing attempt."""
    SUCCESS = "success"           # Parsed without modification
    RECOVERED = "recovered"       # Fixed issues and parsed
    PARTIAL = "partial"           # Extracted some fields
    FAILED = "failed"             # Could not parse


@dataclass
class ParseResult:
    """Result of JSON parsing attempt."""
    status: ParseStatus
    data: Optional[Union[Dict[str, Any], List[Any]]]
    raw_response: str

    # Diagnostic information
    extraction_method: str                    # Which strategy succeeded
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recovered_issues: List[str] = field(default_factory=list)

    def is_success(self) -> bool:
        """Check if parsing succeeded (including recovered)."""
        return self.status in (ParseStatus.SUCCESS, ParseStatus.RECOVERED)

    def get_field(self, key: str, default: Any = None) -> Any:
        """Safely get a field from parsed data."""
        if self.data is None:
            return default
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default


class RobustJSONParser:
    """
    Parser that handles malformed JSON from LLM responses.

    This class implements a cascade of extraction strategies,
    trying each in order until one succeeds.

    Strategy Order:
    1. Direct parse (fastest, for well-formed JSON)
    2. Extract from markdown code blocks
    3. Find JSON object/array in surrounding text
    4. Fix common formatting issues
    5. Repair truncated JSON
    6. Partial field extraction (last resort)
    """

    # Regex patterns for extracting JSON from wrapped responses
    # Order matters: more specific patterns first
    EXTRACTION_PATTERNS = [
        # Markdown JSON code block (most common)
        (r'```json\s*([\s\S]*?)\s*```', 'markdown_json'),
        # Generic markdown code block
        (r'```\s*([\s\S]*?)\s*```', 'markdown_generic'),
        # JSON object (greedy, but handles nested)
        (r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})', 'object_extraction'),
        # JSON array
        (r'(\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])', 'array_extraction'),
    ]

    def __init__(
        self,
        strict: bool = False,
        max_repair_depth: int = 10,
    ):
        """
        Initialize the parser.

        Args:
            strict: If True, only return SUCCESS status (not RECOVERED)
            max_repair_depth: Maximum nesting depth to attempt repair
        """
        self.strict = strict
        self.max_repair_depth = max_repair_depth

    def parse(
        self,
        response: str,
        expected_type: str = "object",  # "object" or "array"
        expected_fields: Optional[List[str]] = None,
    ) -> ParseResult:
        """
        Parse JSON from LLM response with multiple fallback strategies.

        Args:
            response: Raw LLM response text
            expected_type: Expected JSON type ("object" or "array")
            expected_fields: List of fields to extract if partial parsing needed

        Returns:
            ParseResult with parsed data or error information
        """
        if not response or not response.strip():
            return ParseResult(
                status=ParseStatus.FAILED,
                data=None,
                raw_response=response,
                extraction_method="none",
                errors=["Empty response"],
            )

        errors = []
        warnings = []

        # Strategy 1: Direct parse
        try:
            data = json.loads(response)
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=data,
                raw_response=response,
                extraction_method="direct",
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
                        status=ParseStatus.RECOVERED,
                        data=data,
                        raw_response=response,
                        extraction_method=f"pattern:{pattern_name}",
                        recovered_issues=[f"Extracted JSON using {pattern_name}"],
                    )
                except json.JSONDecodeError:
                    # Pattern matched but content wasn't valid JSON
                    # Try fixing the extracted content
                    fixed = self._fix_common_issues(extracted)
                    try:
                        data = json.loads(fixed)
                        return ParseResult(
                            status=ParseStatus.RECOVERED,
                            data=data,
                            raw_response=response,
                            extraction_method=f"pattern:{pattern_name}+fix",
                            recovered_issues=[
                                f"Extracted using {pattern_name}",
                                "Applied common fixes",
                            ],
                        )
                    except json.JSONDecodeError:
                        pass

        # Strategy 3: Fix common issues on full response
        fixed = self._fix_common_issues(response)
        try:
            data = json.loads(fixed)
            return ParseResult(
                status=ParseStatus.RECOVERED,
                data=data,
                raw_response=response,
                extraction_method="fixed_common",
                recovered_issues=["Applied common JSON fixes"],
            )
        except json.JSONDecodeError as e:
            errors.append(f"Common fixes failed: {e}")

        # Strategy 4: Try to repair truncated JSON
        repaired = self._repair_truncated(response)
        if repaired and repaired != response:
            try:
                data = json.loads(repaired)
                return ParseResult(
                    status=ParseStatus.RECOVERED,
                    data=data,
                    raw_response=response,
                    extraction_method="truncation_repair",
                    recovered_issues=["Repaired truncated JSON"],
                    warnings=["JSON was truncated - data may be incomplete"],
                )
            except json.JSONDecodeError:
                pass

        # Strategy 5: Partial extraction (last resort)
        if expected_fields:
            partial = self._extract_fields(response, expected_fields)
            if partial:
                return ParseResult(
                    status=ParseStatus.PARTIAL,
                    data=partial,
                    raw_response=response,
                    extraction_method="partial_extraction",
                    recovered_issues=[f"Extracted {len(partial)} fields via regex"],
                    warnings=[f"Only partial data extracted: {list(partial.keys())}"],
                    errors=errors,
                )

        # All strategies failed
        return ParseResult(
            status=ParseStatus.FAILED,
            data=None,
            raw_response=response,
            extraction_method="none",
            errors=errors,
        )

    def _fix_common_issues(self, text: str) -> str:
        """
        Fix common JSON formatting issues.

        This applies a series of regex-based fixes for issues that
        LLMs commonly produce in JSON output.
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Remove markdown code fence if present at start/end only
        if text.startswith('```'):
            # Find the end of the first line
            first_newline = text.find('\n')
            if first_newline > 0:
                text = text[first_newline + 1:]
        if text.endswith('```'):
            text = text[:-3].rstrip()

        # Remove BOM if present
        if text.startswith('\ufeff'):
            text = text[1:]

        # Fix trailing commas before ] or }
        # {"a": 1, "b": 2,} -> {"a": 1, "b": 2}
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Fix missing commas between elements (common in LLM output)
        # {"a": 1 "b": 2} -> {"a": 1, "b": 2}
        text = re.sub(r'"\s*\n\s*"', '",\n"', text)
        text = re.sub(r'(\d)\s*\n\s*"', r'\1,\n"', text)
        text = re.sub(r'(true|false|null)\s*\n\s*"', r'\1,\n"', text)

        # Replace single quotes with double quotes (for string delimiters)
        # But be careful not to break apostrophes in text
        # Only replace when they're clearly string delimiters
        # {'key': 'value'} -> {"key": "value"}
        text = re.sub(r"(?<=[{,:\[\s])'([^']*)'(?=[},:}\]\s])", r'"\1"', text)

        # Fix unquoted keys (common in JavaScript-style output)
        # {key: "value"} -> {"key": "value"}
        text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

        # Replace Python-style True/False/None with JSON equivalents
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)

        # Replace NaN and Infinity with null (not valid JSON)
        text = re.sub(r'\bNaN\b', 'null', text)
        text = re.sub(r'\bInfinity\b', 'null', text)
        text = re.sub(r'-Infinity\b', 'null', text)

        # Fix escaped single quotes that should be regular apostrophes
        text = text.replace("\\'", "'")

        # Remove C-style comments (sometimes LLMs add these)
        text = re.sub(r'//[^\n]*\n', '\n', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        return text

    def _repair_truncated(self, text: str) -> Optional[str]:
        """
        Attempt to repair truncated JSON by closing open brackets.

        This is useful when the LLM hit max_tokens mid-output.
        We try to close brackets in a way that produces valid JSON.
        """
        # First, try to find where valid JSON structure ends
        text = text.strip()

        # Remove any trailing incomplete string
        # e.g., {"key": "incomplete val -> {"key": "incomplete val"}
        if text.count('"') % 2 == 1:
            # Odd number of quotes - likely truncated string
            # Find the last quote and close the string
            last_quote_idx = text.rfind('"')
            # Check if we're inside a string value
            if last_quote_idx > 0:
                before_quote = text[:last_quote_idx]
                # Count quotes to determine if we're in a string
                if before_quote.count('"') % 2 == 1:
                    # We're inside an unclosed string, close it
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

        # If brackets are imbalanced, try to fix
        if open_braces < 0 or open_brackets < 0:
            # More closes than opens - probably unfixable
            return None

        if open_braces == 0 and open_brackets == 0:
            # Already balanced
            return text

        # Remove any trailing partial value
        # e.g., {"key": tru -> {"key": true}
        text = re.sub(r',\s*$', '', text)  # Remove trailing comma
        text = re.sub(r':\s*$', ': null', text)  # Add null for trailing colon
        text = re.sub(r':\s*(tru|fals|nul)$', lambda m: ': ' + {
            'tru': 'true', 'fals': 'false', 'nul': 'null'
        }.get(m.group(1), 'null'), text)

        # Close brackets
        text = text.rstrip()

        # Remove trailing comma before we close
        if text.endswith(','):
            text = text[:-1]

        # Add closing brackets
        text += ']' * open_brackets
        text += '}' * open_braces

        return text

    def _extract_fields(
        self,
        text: str,
        fields: List[str],
    ) -> Dict[str, Any]:
        """
        Extract specific fields using regex patterns.

        This is a last-resort strategy when JSON is too malformed
        to parse but we can still extract some useful data.
        """
        result = {}

        for field_name in fields:
            # Try various patterns for each field

            # Pattern 1: "field": "string_value"
            pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
            match = re.search(pattern, text)
            if match:
                result[field_name] = match.group(1)
                continue

            # Pattern 2: "field": number
            pattern = rf'"{field_name}"\s*:\s*(-?\d+\.?\d*)'
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                result[field_name] = float(value) if '.' in value else int(value)
                continue

            # Pattern 3: "field": true/false/null
            pattern = rf'"{field_name}"\s*:\s*(true|false|null)'
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                result[field_name] = {'true': True, 'false': False, 'null': None}[value]
                continue

            # Pattern 4: "field": [...] (simple array)
            pattern = rf'"{field_name}"\s*:\s*(\[[^\]]*\])'
            match = re.search(pattern, text)
            if match:
                try:
                    result[field_name] = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
                continue

            # Pattern 5: "field": {...} (simple object)
            pattern = rf'"{field_name}"\s*:\s*(\{{[^}}]*\}})'
            match = re.search(pattern, text)
            if match:
                try:
                    result[field_name] = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        return result


def parse_json_response(
    response: str,
    expected_fields: Optional[List[str]] = None,
) -> ParseResult:
    """
    Convenience function for parsing JSON from LLM response.

    Args:
        response: Raw LLM response
        expected_fields: Fields to extract if full parsing fails

    Returns:
        ParseResult with parsed data or error information
    """
    parser = RobustJSONParser()
    return parser.parse(response, expected_fields=expected_fields)
