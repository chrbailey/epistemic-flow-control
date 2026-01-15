"""
Lawyer Entity Validator and Normalizer

Filters out invalid entities that commonly appear in lawyer fields:
- Geographic locations (cities, states, countries)
- Organizations and institutions
- Court personnel (clerks, staff)
- Pro se indicators

Legal data often contains messy lawyer fields where non-lawyer entities
appear due to data entry errors or system limitations.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RejectionReason(Enum):
    """Why a lawyer entity was rejected."""
    GEOGRAPHIC = "geographic"           # City, state, country name
    ORGANIZATION = "organization"       # Company, institution, agency
    COURT_PERSONNEL = "court_personnel" # Clerk, staff, court
    PRO_SE = "pro_se"                   # Self-represented indicator
    TOO_SHORT = "too_short"             # Not enough characters
    NUMERIC = "numeric"                 # Contains only numbers
    INVALID_PATTERN = "invalid_pattern" # Matches known bad pattern


@dataclass
class LawyerValidation:
    """Result of lawyer entity validation."""
    raw_input: str
    normalized_name: Optional[str]  # None if invalid
    is_valid: bool
    rejection_reason: Optional[RejectionReason]
    confidence: float  # Confidence in the validation decision

    def __str__(self) -> str:
        return self.normalized_name or f"[INVALID: {self.rejection_reason}]"


class LawyerNormalizer:
    """
    Validates and normalizes lawyer entity names.

    Example usage:
        normalizer = LawyerNormalizer()

        # Valid lawyer
        result = normalizer.validate("John Smith")
        print(result.is_valid)  # True

        # Invalid - geographic
        result = normalizer.validate("San Francisco")
        print(result.is_valid)  # False
        print(result.rejection_reason)  # RejectionReason.GEOGRAPHIC
    """

    # Common US cities that appear in lawyer fields
    CITIES = {
        # California
        "san francisco", "los angeles", "san diego", "san jose", "oakland",
        "sacramento", "fresno", "long beach", "santa ana", "anaheim",
        "bakersfield", "riverside", "stockton", "irvine", "fremont",
        "palo alto", "menlo park", "mountain view", "sunnyvale", "santa clara",
        "redwood city", "berkeley", "alameda", "san mateo", "burlingame",
        # Major US cities
        "new york", "chicago", "houston", "phoenix", "philadelphia",
        "dallas", "austin", "san antonio", "jacksonville", "fort worth",
        "columbus", "charlotte", "indianapolis", "seattle", "denver",
        "washington", "boston", "nashville", "baltimore", "oklahoma city",
        "portland", "las vegas", "milwaukee", "albuquerque", "tucson",
        "atlanta", "miami", "detroit", "minneapolis", "cleveland",
        # With common misspellings/variants
        "ny", "nyc", "la", "sf", "dc", "philly", "chi-town",
    }

    # US States and territories
    STATES = {
        "alabama", "alaska", "arizona", "arkansas", "california",
        "colorado", "connecticut", "delaware", "florida", "georgia",
        "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
        "kentucky", "louisiana", "maine", "maryland", "massachusetts",
        "michigan", "minnesota", "mississippi", "missouri", "montana",
        "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
        "new york", "north carolina", "north dakota", "ohio", "oklahoma",
        "oregon", "pennsylvania", "rhode island", "south carolina",
        "south dakota", "tennessee", "texas", "utah", "vermont",
        "virginia", "washington", "west virginia", "wisconsin", "wyoming",
        "district of columbia", "puerto rico", "guam",
        # Abbreviations
        "ca", "ny", "tx", "fl", "il", "pa", "oh", "ga", "nc", "mi",
        "nj", "va", "wa", "az", "ma", "tn", "in", "mo", "md", "wi",
    }

    # Countries
    COUNTRIES = {
        "united states", "usa", "us", "america", "canada", "mexico",
        "united kingdom", "uk", "england", "france", "germany", "japan",
        "china", "india", "australia", "brazil", "italy", "spain",
    }

    # Organization indicators
    ORGANIZATION_PATTERNS = [
        r"\b(llp|llc|inc|corp|ltd|pllc|pc|pa|plc)\b",
        r"\b(law\s*firm|law\s*office|legal\s*group|associates)\b",
        r"\b(department|agency|bureau|commission|authority)\b",
        r"\b(university|college|institute|school|academy)\b",
        r"\b(hospital|medical|clinic|health)\b",
        r"\b(bank|financial|insurance|trust)\b",
        r"\b(government|federal|state|county|city|municipal)\b",
    ]

    # Court personnel indicators
    COURT_PERSONNEL_PATTERNS = [
        r"\bclerk\b",
        r"\bdeputy\s*clerk\b",
        r"\bcourt\s*staff\b",
        r"\bcourt\s*reporter\b",
        r"\bbailiff\b",
        r"\bmarshal\b",
        r"\bcourt\s*administrator\b",
        r"\bjudicial\s*assistant\b",
    ]

    # Pro se indicators
    PRO_SE_PATTERNS = [
        r"\bpro\s*se\b",
        r"\bself[\s-]*represent",
        r"\bin\s*propria\s*persona\b",
        r"\bpro\s*per\b",
        r"\bwithout\s*attorney\b",
        r"\bunrepresented\b",
    ]

    def __init__(self):
        """Initialize with compiled patterns for efficiency."""
        self._org_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.ORGANIZATION_PATTERNS
        ]
        self._court_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.COURT_PERSONNEL_PATTERNS
        ]
        self._prose_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PRO_SE_PATTERNS
        ]

    def validate(self, raw_input: str) -> LawyerValidation:
        """
        Validate and normalize a lawyer entity name.

        Args:
            raw_input: The raw lawyer name/entity

        Returns:
            LawyerValidation with validity status and normalized name if valid
        """
        if not raw_input or not raw_input.strip():
            return LawyerValidation(
                raw_input=raw_input or "",
                normalized_name=None,
                is_valid=False,
                rejection_reason=RejectionReason.TOO_SHORT,
                confidence=1.0
            )

        cleaned = raw_input.strip()

        # Check minimum length
        if len(cleaned) < 3:
            return LawyerValidation(
                raw_input=raw_input,
                normalized_name=None,
                is_valid=False,
                rejection_reason=RejectionReason.TOO_SHORT,
                confidence=1.0
            )

        # Check for numeric-only
        if re.match(r"^[\d\s\-\.]+$", cleaned):
            return LawyerValidation(
                raw_input=raw_input,
                normalized_name=None,
                is_valid=False,
                rejection_reason=RejectionReason.NUMERIC,
                confidence=1.0
            )

        lower = cleaned.lower()

        # Check geographic entities
        if self._is_geographic(lower):
            return LawyerValidation(
                raw_input=raw_input,
                normalized_name=None,
                is_valid=False,
                rejection_reason=RejectionReason.GEOGRAPHIC,
                confidence=0.95
            )

        # Check organizations
        if self._is_organization(cleaned):
            return LawyerValidation(
                raw_input=raw_input,
                normalized_name=None,
                is_valid=False,
                rejection_reason=RejectionReason.ORGANIZATION,
                confidence=0.90
            )

        # Check court personnel
        if self._is_court_personnel(cleaned):
            return LawyerValidation(
                raw_input=raw_input,
                normalized_name=None,
                is_valid=False,
                rejection_reason=RejectionReason.COURT_PERSONNEL,
                confidence=0.95
            )

        # Check pro se
        if self._is_pro_se(cleaned):
            return LawyerValidation(
                raw_input=raw_input,
                normalized_name=None,
                is_valid=False,
                rejection_reason=RejectionReason.PRO_SE,
                confidence=0.99
            )

        # Valid - normalize the name
        normalized = self._normalize_name(cleaned)

        return LawyerValidation(
            raw_input=raw_input,
            normalized_name=normalized,
            is_valid=True,
            rejection_reason=None,
            confidence=0.85
        )

    def _is_geographic(self, lower: str) -> bool:
        """Check if the input is a geographic entity."""
        # Exact match with cities/states/countries
        if lower in self.CITIES or lower in self.STATES or lower in self.COUNTRIES:
            return True

        # Check for "City, State" pattern
        if re.match(r"^[a-z\s]+,\s*[a-z]{2}$", lower):
            return True

        return False

    def _is_organization(self, text: str) -> bool:
        """Check if the input is an organization."""
        return any(p.search(text) for p in self._org_patterns)

    def _is_court_personnel(self, text: str) -> bool:
        """Check if the input is court personnel."""
        return any(p.search(text) for p in self._court_patterns)

    def _is_pro_se(self, text: str) -> bool:
        """Check if the input indicates pro se representation."""
        return any(p.search(text) for p in self._prose_patterns)

    def _normalize_name(self, name: str) -> str:
        """Normalize a valid lawyer name."""
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", name).strip()

        # Normalize casing (title case, but preserve certain patterns)
        parts = normalized.split()
        normalized_parts = []

        for part in parts:
            # Preserve common suffixes
            lower = part.lower().rstrip(".,")
            if lower in {"jr", "sr", "ii", "iii", "iv", "esq"}:
                if lower == "esq":
                    normalized_parts.append("Esq.")
                elif lower in {"jr", "sr"}:
                    normalized_parts.append(lower.title() + ".")
                else:
                    normalized_parts.append(lower.upper())
            # Handle initials
            elif len(part.rstrip(".")) == 1:
                normalized_parts.append(part.upper().rstrip(".") + ".")
            # Handle hyphenated names
            elif "-" in part:
                normalized_parts.append("-".join(p.title() for p in part.split("-")))
            # Handle names with apostrophes (O'Brien, etc.)
            elif "'" in part:
                idx = part.index("'")
                normalized_parts.append(
                    part[:idx].title() + "'" + part[idx+1:].title()
                )
            else:
                normalized_parts.append(part.title())

        return " ".join(normalized_parts)

    def batch_validate(self, inputs: list[str]) -> list[LawyerValidation]:
        """Validate multiple lawyer entities."""
        return [self.validate(inp) for inp in inputs]

    def filter_valid(self, inputs: list[str]) -> list[str]:
        """Return only valid, normalized lawyer names."""
        results = self.batch_validate(inputs)
        return [r.normalized_name for r in results if r.is_valid and r.normalized_name]
