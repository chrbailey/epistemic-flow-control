"""
Judge Name Normalizer

Extracts and normalizes judge names from various messy input formats:
- CourtListener URLs (e.g., "/person/john-g-roberts-jr/")
- Direct names with inconsistent casing
- Names with honorifics and suffixes

This is critical for pattern matching across data sources where
the same judge may appear in different formats.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SourceType(Enum):
    """How the judge name was extracted."""
    COURTLISTENER_URL = "courtlistener_url"
    DIRECT_NAME = "direct_name"
    PACER_ID = "pacer_id"
    UNKNOWN = "unknown"


@dataclass
class NormalizedJudge:
    """Result of judge name normalization."""
    raw_input: str
    normalized_name: str
    source_type: SourceType
    confidence: float  # 0.0 to 1.0

    # Extracted components (when available)
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    suffix: Optional[str] = None

    def __str__(self) -> str:
        return self.normalized_name


class JudgeNormalizer:
    """
    Normalizes judge names from various messy input formats.

    Example usage:
        normalizer = JudgeNormalizer()
        result = normalizer.normalize("https://courtlistener.com/person/john-g-roberts-jr/")
        print(result.normalized_name)  # "John G. Roberts Jr."
    """

    # Common suffixes to preserve
    SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

    # Honorifics to strip (we normalize to just the name)
    HONORIFICS = {"judge", "justice", "hon", "honorable", "chief", "associate"}

    # CourtListener URL pattern
    COURTLISTENER_PATTERN = re.compile(
        r"(?:https?://)?(?:www\.)?courtlistener\.com/person/([a-z0-9-]+)/?",
        re.IGNORECASE
    )

    # Alternative: just the slug part
    SLUG_PATTERN = re.compile(r"^[a-z]+(?:-[a-z]+)*(?:-(?:jr|sr|ii|iii|iv|v))?$", re.IGNORECASE)

    def normalize(self, raw_input: str) -> NormalizedJudge:
        """
        Normalize a judge name from any supported format.

        Args:
            raw_input: The raw judge identifier (URL, name, etc.)

        Returns:
            NormalizedJudge with the cleaned name and metadata
        """
        if not raw_input or not raw_input.strip():
            return NormalizedJudge(
                raw_input=raw_input or "",
                normalized_name="",
                source_type=SourceType.UNKNOWN,
                confidence=0.0
            )

        raw_input = raw_input.strip()

        # Try CourtListener URL first
        cl_match = self.COURTLISTENER_PATTERN.search(raw_input)
        if cl_match:
            return self._from_courtlistener_slug(raw_input, cl_match.group(1))

        # Check if it looks like a slug (hyphenated lowercase)
        if self.SLUG_PATTERN.match(raw_input):
            return self._from_slug(raw_input)

        # Treat as direct name
        return self._from_direct_name(raw_input)

    def _from_courtlistener_slug(self, raw_input: str, slug: str) -> NormalizedJudge:
        """Extract judge name from CourtListener URL slug."""
        parts = slug.lower().split("-")

        # Handle suffix
        suffix = None
        if parts and parts[-1] in self.SUFFIXES:
            suffix = self._format_suffix(parts.pop())

        if not parts:
            return NormalizedJudge(
                raw_input=raw_input,
                normalized_name="",
                source_type=SourceType.COURTLISTENER_URL,
                confidence=0.0
            )

        # First name
        first_name = parts[0].title() if parts else None

        # Last name (last remaining part)
        last_name = parts[-1].title() if len(parts) > 1 else None

        # Middle name(s)
        middle_parts = parts[1:-1] if len(parts) > 2 else []
        middle_name = " ".join(self._format_middle(p) for p in middle_parts) if middle_parts else None

        # Build normalized name
        name_parts = [first_name]
        if middle_name:
            name_parts.append(middle_name)
        if last_name:
            name_parts.append(last_name)
        elif first_name:
            # Single name case - treat as last name
            last_name = first_name
            first_name = None
            name_parts = [last_name]
        if suffix:
            name_parts.append(suffix)

        normalized_name = " ".join(p for p in name_parts if p)

        return NormalizedJudge(
            raw_input=raw_input,
            normalized_name=normalized_name,
            source_type=SourceType.COURTLISTENER_URL,
            confidence=0.95,  # High confidence from structured URL
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            suffix=suffix
        )

    def _from_slug(self, raw_input: str) -> NormalizedJudge:
        """Handle hyphenated slug format without full URL."""
        result = self._from_courtlistener_slug(raw_input, raw_input)
        return NormalizedJudge(
            raw_input=result.raw_input,
            normalized_name=result.normalized_name,
            source_type=SourceType.DIRECT_NAME,  # Not from URL
            confidence=0.85,  # Slightly lower - might be coincidental format
            first_name=result.first_name,
            middle_name=result.middle_name,
            last_name=result.last_name,
            suffix=result.suffix
        )

    def _from_direct_name(self, raw_input: str) -> NormalizedJudge:
        """Normalize a direct name string."""
        # Remove common honorifics
        cleaned = raw_input
        for honorific in self.HONORIFICS:
            pattern = re.compile(rf"\b{honorific}\.?\b", re.IGNORECASE)
            cleaned = pattern.sub("", cleaned)

        # Clean up whitespace and punctuation
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[,.\s]+|[,.\s]+$", "", cleaned)

        if not cleaned:
            return NormalizedJudge(
                raw_input=raw_input,
                normalized_name="",
                source_type=SourceType.DIRECT_NAME,
                confidence=0.0
            )

        # Split into parts
        parts = cleaned.split()

        # Handle suffix
        suffix = None
        if parts and parts[-1].lower().rstrip(".") in self.SUFFIXES:
            suffix = self._format_suffix(parts.pop())

        if not parts:
            return NormalizedJudge(
                raw_input=raw_input,
                normalized_name="",
                source_type=SourceType.DIRECT_NAME,
                confidence=0.0
            )

        # Normalize casing
        normalized_parts = []
        for i, part in enumerate(parts):
            # Handle initials (single letter or letter with period)
            if len(part.rstrip(".")) == 1:
                normalized_parts.append(part.upper().rstrip(".") + ".")
            else:
                normalized_parts.append(part.title())

        if suffix:
            normalized_parts.append(suffix)

        normalized_name = " ".join(normalized_parts)

        # Extract components
        first_name = normalized_parts[0] if normalized_parts else None
        last_name = normalized_parts[-1] if len(normalized_parts) > 1 else None

        # Handle middle name(s)
        middle_idx_end = -1 if suffix is None else -2
        middle_parts = normalized_parts[1:middle_idx_end] if len(normalized_parts) > 2 else []
        middle_name = " ".join(middle_parts) if middle_parts else None

        return NormalizedJudge(
            raw_input=raw_input,
            normalized_name=normalized_name,
            source_type=SourceType.DIRECT_NAME,
            confidence=0.80,  # Lower confidence for unstructured input
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            suffix=suffix
        )

    def _format_suffix(self, suffix: str) -> str:
        """Format a suffix properly."""
        s = suffix.lower().rstrip(".")
        if s in {"jr", "sr"}:
            return s.title() + "."
        elif s in {"ii", "iii", "iv", "v"}:
            return s.upper()
        return suffix.title()

    def _format_middle(self, middle: str) -> str:
        """Format a middle name or initial."""
        if len(middle) == 1:
            return middle.upper() + "."
        return middle.title()

    def batch_normalize(self, inputs: list[str]) -> list[NormalizedJudge]:
        """Normalize multiple judge names."""
        return [self.normalize(inp) for inp in inputs]
