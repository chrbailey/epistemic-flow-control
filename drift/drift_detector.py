"""
Pattern Drift Detector

Monitors pattern embeddings over time to detect statistically significant
changes in judicial behavior. Drift can indicate:

- Gradual drift: Slow evolution over many decisions (natural pattern evolution)
- Sudden drift: Abrupt change (new precedent, policy change, personnel change)
- Seasonal drift: Cyclical patterns (end-of-term effects, holiday patterns)
- Reverting drift: Temporary deviation that returns to baseline

Early detection enables proactive recalibration of confidence estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from .embedding_tracker import EmbeddingTracker, PatternEmbedding


class DriftType(Enum):
    """Classification of detected drift."""
    NONE = "none"                 # No significant drift
    GRADUAL = "gradual"           # Slow evolution over time
    SUDDEN = "sudden"             # Abrupt change
    SEASONAL = "seasonal"         # Cyclical pattern
    REVERTING = "reverting"       # Temporary deviation


class DriftSeverity(Enum):
    """Severity of detected drift."""
    NEGLIGIBLE = "negligible"   # < 5% change
    MINOR = "minor"             # 5-15% change
    MODERATE = "moderate"       # 15-30% change
    SIGNIFICANT = "significant" # 30-50% change
    SEVERE = "severe"           # > 50% change


@dataclass
class DriftEvent:
    """A detected drift event."""
    entity_id: str
    pattern_type: str
    drift_type: DriftType
    severity: DriftSeverity

    # Metrics
    baseline_similarity: float    # Cosine similarity to baseline
    drift_magnitude: float        # Euclidean distance from baseline
    confidence_impact: float      # Estimated impact on confidence

    # Context
    detected_at: datetime
    baseline_date: datetime
    samples_since_baseline: int

    # Dimension analysis
    top_changed_dimensions: list[tuple[int, float]] = field(default_factory=list)

    # Recommendations
    recommendation: str = ""

    @property
    def requires_recalibration(self) -> bool:
        """Check if drift requires pattern recalibration."""
        return self.severity in {DriftSeverity.SIGNIFICANT, DriftSeverity.SEVERE}

    @property
    def drift_percentage(self) -> float:
        """Express drift as a percentage change."""
        # Convert cosine similarity to percentage drift
        # similarity of 1.0 = 0% drift, 0.5 = 50% drift
        return (1.0 - self.baseline_similarity) * 100


@dataclass
class Baseline:
    """Stored baseline for drift comparison."""
    entity_id: str
    pattern_type: str
    embedding: PatternEmbedding
    created_at: datetime
    sample_count: int
    is_active: bool = True


class DriftDetector:
    """
    Detects drift in judicial behavior patterns over time.

    Example usage:
        detector = DriftDetector()

        # Set baseline
        detector.set_baseline(baseline_embedding)

        # Check for drift
        event = detector.detect_drift(
            current_embedding,
            entity_id="judge_alsup",
            pattern_type="summary_judgment"
        )

        if event.requires_recalibration:
            print(f"DRIFT DETECTED: {event.recommendation}")
    """

    # Thresholds for drift detection
    THRESHOLD_NEGLIGIBLE = 0.95  # Similarity above this = negligible drift
    THRESHOLD_MINOR = 0.85       # Similarity above this = minor drift
    THRESHOLD_MODERATE = 0.70    # Similarity above this = moderate drift
    THRESHOLD_SIGNIFICANT = 0.50 # Similarity above this = significant drift
    # Below 0.50 = severe drift

    def __init__(self):
        """Initialize the drift detector."""
        self.tracker = EmbeddingTracker()
        self.baselines: dict[str, Baseline] = {}  # key: entity_id:pattern_type

    def _baseline_key(self, entity_id: str, pattern_type: str) -> str:
        """Generate key for baseline storage."""
        return f"{entity_id}:{pattern_type}"

    def set_baseline(
        self,
        embedding: PatternEmbedding,
        entity_id: Optional[str] = None,
        pattern_type: Optional[str] = None
    ) -> Baseline:
        """
        Set or update the baseline for drift comparison.

        Args:
            embedding: The baseline embedding
            entity_id: Override entity ID (defaults to embedding's)
            pattern_type: Override pattern type (defaults to embedding's)

        Returns:
            The created Baseline object
        """
        eid = entity_id or embedding.entity_id
        ptype = pattern_type or embedding.pattern_type
        key = self._baseline_key(eid, ptype)

        baseline = Baseline(
            entity_id=eid,
            pattern_type=ptype,
            embedding=embedding,
            created_at=datetime.now(),
            sample_count=embedding.sample_count,
            is_active=True
        )

        self.baselines[key] = baseline
        return baseline

    def get_baseline(
        self,
        entity_id: str,
        pattern_type: str
    ) -> Optional[Baseline]:
        """Get the current baseline for an entity/pattern."""
        key = self._baseline_key(entity_id, pattern_type)
        return self.baselines.get(key)

    def detect_drift(
        self,
        current: PatternEmbedding,
        entity_id: Optional[str] = None,
        pattern_type: Optional[str] = None
    ) -> DriftEvent:
        """
        Compare current embedding against baseline to detect drift.

        Args:
            current: Current pattern embedding
            entity_id: Override entity ID
            pattern_type: Override pattern type

        Returns:
            DriftEvent describing any detected drift
        """
        eid = entity_id or current.entity_id
        ptype = pattern_type or current.pattern_type

        baseline = self.get_baseline(eid, ptype)

        if baseline is None:
            # No baseline - set this as baseline and return no drift
            self.set_baseline(current, eid, ptype)
            return DriftEvent(
                entity_id=eid,
                pattern_type=ptype,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NEGLIGIBLE,
                baseline_similarity=1.0,
                drift_magnitude=0.0,
                confidence_impact=0.0,
                detected_at=datetime.now(),
                baseline_date=datetime.now(),
                samples_since_baseline=0,
                recommendation="Initial baseline established. No drift comparison possible yet."
            )

        # Calculate similarity metrics
        similarity = self.tracker.cosine_similarity(baseline.embedding, current)
        magnitude = self.tracker.euclidean_distance(baseline.embedding, current)

        # Determine severity
        severity = self._assess_severity(similarity)

        # Determine drift type
        drift_type = self._classify_drift_type(
            similarity,
            magnitude,
            baseline.created_at,
            current.sample_count - baseline.sample_count
        )

        # Find most changed dimensions
        top_changed = self._find_top_changed_dimensions(
            baseline.embedding.embedding,
            current.embedding,
            top_n=5
        )

        # Estimate confidence impact
        confidence_impact = self._estimate_confidence_impact(similarity, severity)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            eid, ptype, drift_type, severity, top_changed
        )

        return DriftEvent(
            entity_id=eid,
            pattern_type=ptype,
            drift_type=drift_type,
            severity=severity,
            baseline_similarity=round(similarity, 4),
            drift_magnitude=round(magnitude, 4),
            confidence_impact=round(confidence_impact, 4),
            detected_at=datetime.now(),
            baseline_date=baseline.created_at,
            samples_since_baseline=current.sample_count - baseline.sample_count,
            top_changed_dimensions=top_changed,
            recommendation=recommendation
        )

    def _assess_severity(self, similarity: float) -> DriftSeverity:
        """Assess drift severity from similarity score."""
        if similarity >= self.THRESHOLD_NEGLIGIBLE:
            return DriftSeverity.NEGLIGIBLE
        elif similarity >= self.THRESHOLD_MINOR:
            return DriftSeverity.MINOR
        elif similarity >= self.THRESHOLD_MODERATE:
            return DriftSeverity.MODERATE
        elif similarity >= self.THRESHOLD_SIGNIFICANT:
            return DriftSeverity.SIGNIFICANT
        else:
            return DriftSeverity.SEVERE

    def _classify_drift_type(
        self,
        similarity: float,
        magnitude: float,
        baseline_date: datetime,
        samples_diff: int
    ) -> DriftType:
        """Classify the type of drift detected."""
        if similarity >= self.THRESHOLD_NEGLIGIBLE:
            return DriftType.NONE

        days_since_baseline = (datetime.now() - baseline_date).days

        # Sudden drift: large change with few samples
        if samples_diff < 10 and similarity < self.THRESHOLD_MODERATE:
            return DriftType.SUDDEN

        # Gradual drift: change over many samples/time
        if days_since_baseline > 90 and samples_diff > 50:
            return DriftType.GRADUAL

        # Default to gradual for moderate changes
        return DriftType.GRADUAL

    def _find_top_changed_dimensions(
        self,
        baseline: list[float],
        current: list[float],
        top_n: int = 5
    ) -> list[tuple[int, float]]:
        """Find the dimensions with the largest changes."""
        changes = [
            (i, abs(current[i] - baseline[i]))
            for i in range(len(baseline))
        ]
        changes.sort(key=lambda x: x[1], reverse=True)
        return [(dim, round(delta, 4)) for dim, delta in changes[:top_n]]

    def _estimate_confidence_impact(
        self,
        similarity: float,
        severity: DriftSeverity
    ) -> float:
        """Estimate how much drift should reduce confidence."""
        # Map severity to confidence reduction
        impact_map = {
            DriftSeverity.NEGLIGIBLE: 0.0,
            DriftSeverity.MINOR: 0.05,
            DriftSeverity.MODERATE: 0.15,
            DriftSeverity.SIGNIFICANT: 0.30,
            DriftSeverity.SEVERE: 0.50,
        }
        return impact_map[severity]

    def _generate_recommendation(
        self,
        entity_id: str,
        pattern_type: str,
        drift_type: DriftType,
        severity: DriftSeverity,
        top_changed: list[tuple[int, float]]
    ) -> str:
        """Generate actionable recommendation for detected drift."""
        if drift_type == DriftType.NONE:
            return f"Pattern stable for {entity_id}. No action required."

        dimension_names = {
            0: "grant_rate", 1: "denial_rate", 2: "partial_grant_rate",
            16: "avg_days_to_decision", 17: "median_days",
            32: "patent_share", 33: "employment_share",
            48: "motion_grant_rate", 49: "discovery_limit_rate"
        }

        changed_names = [
            dimension_names.get(dim, f"dim_{dim}")
            for dim, _ in top_changed[:3]
        ]

        if severity == DriftSeverity.SEVERE:
            return (
                f"SEVERE DRIFT for {entity_id} ({pattern_type}). "
                f"Primary changes in: {', '.join(changed_names)}. "
                f"IMMEDIATE recalibration required. Consider: "
                f"(1) Verify data quality, (2) Check for external events, "
                f"(3) Reset baseline if change is permanent."
            )
        elif severity == DriftSeverity.SIGNIFICANT:
            return (
                f"SIGNIFICANT DRIFT for {entity_id} ({pattern_type}). "
                f"Changed dimensions: {', '.join(changed_names)}. "
                f"Schedule recalibration and investigate root cause."
            )
        elif severity == DriftSeverity.MODERATE:
            return (
                f"MODERATE DRIFT for {entity_id} ({pattern_type}). "
                f"Monitor closely and consider recalibration if trend continues."
            )
        else:
            return (
                f"MINOR DRIFT for {entity_id} ({pattern_type}). "
                f"Normal variation within expected bounds."
            )

    def batch_detect(
        self,
        embeddings: list[PatternEmbedding]
    ) -> list[DriftEvent]:
        """Detect drift for multiple embeddings."""
        return [self.detect_drift(emb) for emb in embeddings]

    def should_recalibrate(
        self,
        entity_id: str,
        pattern_type: str,
        days_threshold: int = 90,
        samples_threshold: int = 100
    ) -> tuple[bool, str]:
        """
        Check if a baseline should be recalibrated based on age/samples.

        Args:
            entity_id: Entity to check
            pattern_type: Pattern type to check
            days_threshold: Max days before recalibration recommended
            samples_threshold: Max samples before recalibration recommended

        Returns:
            Tuple of (should_recalibrate, reason)
        """
        baseline = self.get_baseline(entity_id, pattern_type)

        if baseline is None:
            return True, "No baseline exists"

        days_old = (datetime.now() - baseline.created_at).days

        if days_old > days_threshold:
            return True, f"Baseline is {days_old} days old (threshold: {days_threshold})"

        return False, "Baseline is current"
