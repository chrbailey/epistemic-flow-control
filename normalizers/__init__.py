"""
Entity Normalization Module

Provides tools for cleaning and normalizing messy court data:
- JudgeNormalizer: Extract and normalize judge names from various formats
- LawyerNormalizer: Validate and clean lawyer entity names

These normalizers are essential for reliable pattern matching across
data sources with inconsistent formatting (CourtListener, PACER, etc.)
"""

from .judge_normalizer import JudgeNormalizer, NormalizedJudge
from .lawyer_normalizer import LawyerNormalizer, LawyerValidation

__all__ = [
    "JudgeNormalizer",
    "NormalizedJudge",
    "LawyerNormalizer",
    "LawyerValidation",
]
