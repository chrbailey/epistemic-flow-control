"""
Pattern Embedding Tracker

Generates and manages numerical representations of judicial behavior patterns.
These embeddings capture multi-dimensional pattern characteristics:
- Grant/denial rates at various confidence thresholds
- Temporal patterns (time-to-decision, seasonal variation)
- Case type distributions
- Procedural preferences

The embeddings enable mathematical comparison of patterns over time
using cosine similarity and other distance metrics.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib
import json


@dataclass
class PatternEmbedding:
    """A numerical representation of a judicial behavior pattern."""
    entity_id: str
    pattern_type: str
    dimensions: int
    embedding: list[float]
    sample_count: int
    created_at: datetime
    metadata: dict = field(default_factory=dict)

    @property
    def embedding_hash(self) -> str:
        """Generate a hash for quick comparison."""
        content = json.dumps(self.embedding, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def magnitude(self) -> float:
        """Calculate the L2 norm of the embedding."""
        return math.sqrt(sum(x ** 2 for x in self.embedding))

    def normalize(self) -> "PatternEmbedding":
        """Return a unit-normalized copy of this embedding."""
        mag = self.magnitude()
        if mag == 0:
            return self
        return PatternEmbedding(
            entity_id=self.entity_id,
            pattern_type=self.pattern_type,
            dimensions=self.dimensions,
            embedding=[x / mag for x in self.embedding],
            sample_count=self.sample_count,
            created_at=self.created_at,
            metadata=self.metadata
        )


class EmbeddingTracker:
    """
    Generates and tracks pattern embeddings for drift detection.

    The default embedding is 64-dimensional, capturing:
    - Dimensions 0-15: Rate metrics at different thresholds
    - Dimensions 16-31: Temporal characteristics
    - Dimensions 32-47: Case type distribution
    - Dimensions 48-63: Procedural preferences

    Example usage:
        tracker = EmbeddingTracker()

        # Generate embedding from pattern data
        embedding = tracker.generate(
            entity_id="judge_alsup",
            pattern_type="summary_judgment",
            metrics={
                "grant_rate": 0.45,
                "avg_days_to_decision": 120,
                "case_types": {"patent": 0.6, "employment": 0.3, "other": 0.1}
            }
        )

        # Compare two embeddings
        similarity = tracker.cosine_similarity(embedding1, embedding2)
    """

    DEFAULT_DIMENSIONS = 64

    # Metric mappings to embedding dimensions
    RATE_METRICS = [
        "grant_rate", "denial_rate", "partial_grant_rate",
        "dismissal_rate", "settlement_rate", "default_rate",
        "reversal_rate", "affirmance_rate",
        "rate_q1", "rate_q2", "rate_q3", "rate_q4",  # Quarterly rates
        "trend_slope", "variance", "wilson_lower", "wilson_upper"
    ]

    TEMPORAL_METRICS = [
        "avg_days_to_decision", "median_days", "std_days",
        "morning_rate", "afternoon_rate", "monday_rate",
        "friday_rate", "month_end_rate", "quarter_end_rate",
        "holiday_proximity_rate", "recess_rate", "term_start_rate",
        "seasonal_amplitude", "weekly_variance", "monthly_variance",
        "year_over_year_change"
    ]

    CASE_TYPE_METRICS = [
        "patent_share", "employment_share", "civil_rights_share",
        "contract_share", "tort_share", "securities_share",
        "antitrust_share", "environmental_share", "immigration_share",
        "criminal_share", "bankruptcy_share", "tax_share",
        "ip_general_share", "admin_share", "constitutional_share",
        "other_share"
    ]

    PROCEDURAL_METRICS = [
        "motion_grant_rate", "discovery_limit_rate", "extension_grant_rate",
        "oral_argument_rate", "bench_trial_rate", "jury_trial_rate",
        "pretrial_settlement_rate", "summary_judgment_rate",
        "class_cert_rate", "injunction_rate", "sanction_rate",
        "remand_rate", "transfer_rate", "consolidation_rate",
        "bifurcation_rate", "stay_rate"
    ]

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS):
        """
        Initialize the embedding tracker.

        Args:
            dimensions: Number of dimensions in embeddings (default 64)
        """
        self.dimensions = dimensions

    def generate(
        self,
        entity_id: str,
        pattern_type: str,
        metrics: dict,
        sample_count: int = 0
    ) -> PatternEmbedding:
        """
        Generate a pattern embedding from metrics.

        Args:
            entity_id: ID of the entity (judge, lawyer, etc.)
            pattern_type: Type of pattern being tracked
            metrics: Dictionary of metric values
            sample_count: Number of samples used to compute metrics

        Returns:
            PatternEmbedding representing the pattern
        """
        embedding = [0.0] * self.dimensions

        # Fill rate metrics (dimensions 0-15)
        for i, metric in enumerate(self.RATE_METRICS[:16]):
            if metric in metrics:
                embedding[i] = self._normalize_rate(metrics[metric])

        # Fill temporal metrics (dimensions 16-31)
        for i, metric in enumerate(self.TEMPORAL_METRICS[:16]):
            if metric in metrics:
                embedding[16 + i] = self._normalize_temporal(metrics[metric], metric)

        # Fill case type metrics (dimensions 32-47)
        case_types = metrics.get("case_types", {})
        if isinstance(case_types, dict):
            for i, metric in enumerate(self.CASE_TYPE_METRICS[:16]):
                base_type = metric.replace("_share", "")
                if base_type in case_types:
                    embedding[32 + i] = self._normalize_rate(case_types[base_type])

        # Fill procedural metrics (dimensions 48-63)
        for i, metric in enumerate(self.PROCEDURAL_METRICS[:16]):
            if metric in metrics:
                embedding[48 + i] = self._normalize_rate(metrics[metric])

        return PatternEmbedding(
            entity_id=entity_id,
            pattern_type=pattern_type,
            dimensions=self.dimensions,
            embedding=embedding,
            sample_count=sample_count,
            created_at=datetime.now(),
            metadata={"source_metrics": list(metrics.keys())}
        )

    def cosine_similarity(
        self,
        embedding1: PatternEmbedding,
        embedding2: PatternEmbedding
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Returns:
            Similarity score from -1 to 1 (1 = identical, 0 = orthogonal)
        """
        if embedding1.dimensions != embedding2.dimensions:
            raise ValueError("Embeddings must have same dimensions")

        dot_product = sum(
            a * b for a, b in zip(embedding1.embedding, embedding2.embedding)
        )
        mag1 = embedding1.magnitude()
        mag2 = embedding2.magnitude()

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def euclidean_distance(
        self,
        embedding1: PatternEmbedding,
        embedding2: PatternEmbedding
    ) -> float:
        """
        Calculate Euclidean distance between two embeddings.

        Returns:
            Distance (0 = identical, higher = more different)
        """
        if embedding1.dimensions != embedding2.dimensions:
            raise ValueError("Embeddings must have same dimensions")

        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(embedding1.embedding, embedding2.embedding)
        ))

    def manhattan_distance(
        self,
        embedding1: PatternEmbedding,
        embedding2: PatternEmbedding
    ) -> float:
        """
        Calculate Manhattan (L1) distance between two embeddings.

        Returns:
            Distance (0 = identical, higher = more different)
        """
        if embedding1.dimensions != embedding2.dimensions:
            raise ValueError("Embeddings must have same dimensions")

        return sum(
            abs(a - b) for a, b in zip(embedding1.embedding, embedding2.embedding)
        )

    def weighted_average(
        self,
        embeddings: list[PatternEmbedding],
        weights: Optional[list[float]] = None
    ) -> PatternEmbedding:
        """
        Compute weighted average of multiple embeddings.

        Useful for computing baseline embeddings from multiple samples.

        Args:
            embeddings: List of embeddings to average
            weights: Optional weights (defaults to equal weighting)

        Returns:
            New embedding representing the weighted average
        """
        if not embeddings:
            raise ValueError("Need at least one embedding")

        if weights is None:
            weights = [1.0 / len(embeddings)] * len(embeddings)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        dims = embeddings[0].dimensions
        avg_embedding = [0.0] * dims

        for emb, weight in zip(embeddings, weights):
            for i, val in enumerate(emb.embedding):
                avg_embedding[i] += val * weight

        total_samples = sum(e.sample_count for e in embeddings)

        return PatternEmbedding(
            entity_id=embeddings[0].entity_id,
            pattern_type=embeddings[0].pattern_type,
            dimensions=dims,
            embedding=avg_embedding,
            sample_count=total_samples,
            created_at=datetime.now(),
            metadata={"averaged_from": len(embeddings)}
        )

    def _normalize_rate(self, value: float) -> float:
        """Normalize a rate value to [0, 1]."""
        return max(0.0, min(1.0, float(value)))

    def _normalize_temporal(self, value: float, metric: str) -> float:
        """Normalize a temporal value to approximately [0, 1]."""
        # Days-based metrics: normalize assuming max ~365 days
        if "days" in metric:
            return min(1.0, float(value) / 365.0)
        # Rate-based temporal metrics are already normalized
        return self._normalize_rate(value)
