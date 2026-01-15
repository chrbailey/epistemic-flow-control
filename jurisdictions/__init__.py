"""
Jurisdictional Context Module

Provides court-specific and judge-specific context for pattern analysis:
- JurisdictionalContext: Base class for court contexts
- NDCalContext: Northern District of California specifics
- AlsupContext: Judge William Alsup preferences

Different courts and judges have different procedural requirements,
formatting preferences, and behavioral patterns. This module makes
that context available to the pattern analysis system.
"""

from .base import JurisdictionalContext, FormatRequirement, ProceduralRule
from .nd_cal import NDCalContext
from .alsup import AlsupContext

__all__ = [
    "JurisdictionalContext",
    "FormatRequirement",
    "ProceduralRule",
    "NDCalContext",
    "AlsupContext",
]
