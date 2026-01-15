"""
Concentration Analysis Module

Tools for detecting unhealthy concentration patterns (Single Point of Failure risk):
- HHICalculator: Compute Herfindahl-Hirschman Index for market concentration
- SPOFDetector: Identify when single entities handle disproportionate workload

The HHI is widely used in antitrust analysis and is perfectly suited for
detecting when legal work is overly concentrated with specific judges,
lawyers, or law firms.
"""

from .hhi_calculator import HHICalculator, ConcentrationLevel, HHIResult
from .spof_detector import SPOFDetector, SPOFRisk, RiskAssessment

__all__ = [
    "HHICalculator",
    "ConcentrationLevel",
    "HHIResult",
    "SPOFDetector",
    "SPOFRisk",
    "RiskAssessment",
]
