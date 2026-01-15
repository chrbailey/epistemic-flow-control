"""
Pattern Drift Detection Module

Tools for monitoring changes in judicial behavior patterns over time:
- EmbeddingTracker: Generate and store pattern embeddings
- DriftDetector: Detect statistically significant pattern drift

Judicial patterns can change for many reasons:
- New case law precedents
- Judge retirement/replacement
- Policy changes at the court
- Evolving legal standards

Early drift detection allows proactive pattern recalibration.
"""

from .embedding_tracker import EmbeddingTracker, PatternEmbedding
from .drift_detector import DriftDetector, DriftEvent, DriftType

__all__ = [
    "EmbeddingTracker",
    "PatternEmbedding",
    "DriftDetector",
    "DriftEvent",
    "DriftType",
]
