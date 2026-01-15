"""
Base Jurisdictional Context

Abstract base class for court and judge-specific contexts.
Each jurisdiction can define its own:
- Format requirements (page limits, fonts, margins)
- Procedural rules (filing deadlines, motion types)
- Behavioral patterns (typical grant rates, timing)
- Areas of expertise or preference
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MotionType(Enum):
    """Common motion types in federal litigation."""
    SUMMARY_JUDGMENT = "summary_judgment"
    MOTION_TO_DISMISS = "motion_to_dismiss"
    MOTION_IN_LIMINE = "motion_in_limine"
    DISCOVERY_MOTION = "discovery_motion"
    PRELIMINARY_INJUNCTION = "preliminary_injunction"
    CLASS_CERTIFICATION = "class_certification"
    SANCTIONS = "sanctions"
    EXTENSION = "extension"
    ADMINISTRATIVE = "administrative"


@dataclass
class FormatRequirement:
    """Document formatting requirement."""
    name: str
    value: str
    is_mandatory: bool = True
    notes: str = ""

    def __str__(self) -> str:
        status = "REQUIRED" if self.is_mandatory else "recommended"
        return f"{self.name}: {self.value} ({status})"


@dataclass
class ProceduralRule:
    """A procedural rule or preference."""
    rule_id: str
    title: str
    description: str
    source: str  # Local rule number, standing order, etc.
    is_mandatory: bool = True
    applies_to: list[MotionType] = field(default_factory=list)

    def applies_to_motion(self, motion_type: MotionType) -> bool:
        """Check if rule applies to a specific motion type."""
        if not self.applies_to:
            return True  # Empty means applies to all
        return motion_type in self.applies_to


@dataclass
class ExpertiseArea:
    """An area of judicial expertise or interest."""
    area: str
    description: str
    experience_level: str  # "extensive", "moderate", "limited"
    notable_cases: list[str] = field(default_factory=list)


class JurisdictionalContext(ABC):
    """
    Abstract base class for jurisdictional contexts.

    Subclasses provide court-specific or judge-specific information
    that affects how patterns should be interpreted and applied.
    """

    @property
    @abstractmethod
    def court_name(self) -> str:
        """Full court name."""
        pass

    @property
    @abstractmethod
    def court_code(self) -> str:
        """Short court identifier (e.g., 'nd_cal', 'ed_tex')."""
        pass

    @property
    def judge_name(self) -> Optional[str]:
        """Judge name if this is a judge-specific context."""
        return None

    @abstractmethod
    def get_format_requirements(self) -> list[FormatRequirement]:
        """Get all format requirements for this context."""
        pass

    @abstractmethod
    def get_procedural_rules(self) -> list[ProceduralRule]:
        """Get all procedural rules for this context."""
        pass

    def get_rules_for_motion(self, motion_type: MotionType) -> list[ProceduralRule]:
        """Get procedural rules applicable to a specific motion type."""
        return [
            rule for rule in self.get_procedural_rules()
            if rule.applies_to_motion(motion_type)
        ]

    def get_expertise_areas(self) -> list[ExpertiseArea]:
        """Get areas of expertise (primarily for judge contexts)."""
        return []

    def get_pattern_adjustments(self) -> dict:
        """
        Get jurisdiction-specific adjustments to pattern weights.

        Returns a dictionary of pattern_type -> adjustment_factor.
        For example, if a judge is known to grant summary judgment
        less often than average, return {"summary_judgment_grant_rate": 0.8}
        """
        return {}

    def get_context_summary(self) -> dict:
        """Get a summary of this context for display."""
        return {
            "court_name": self.court_name,
            "court_code": self.court_code,
            "judge_name": self.judge_name,
            "format_requirements": len(self.get_format_requirements()),
            "procedural_rules": len(self.get_procedural_rules()),
            "expertise_areas": [e.area for e in self.get_expertise_areas()],
        }

    def validate_document(self, document_info: dict) -> list[str]:
        """
        Validate a document against format requirements.

        Args:
            document_info: Dict with keys like 'page_count', 'font', 'font_size'

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        requirements = self.get_format_requirements()

        for req in requirements:
            if not req.is_mandatory:
                continue

            if req.name == "page_limit":
                page_count = document_info.get("page_count", 0)
                limit = int(req.value.split()[0])  # "25 pages" -> 25
                if page_count > limit:
                    errors.append(f"Document exceeds page limit: {page_count} > {limit}")

            elif req.name == "font":
                font = document_info.get("font", "")
                if font and req.value.lower() not in font.lower():
                    errors.append(f"Font mismatch: expected {req.value}, got {font}")

            elif req.name == "font_size":
                size = document_info.get("font_size", 0)
                required_size = int(req.value.replace("pt", ""))
                if size and size < required_size:
                    errors.append(f"Font size too small: {size}pt < {required_size}pt")

        return errors
