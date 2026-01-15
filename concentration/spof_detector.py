"""
Single Point of Failure (SPOF) Detector

Identifies when individual entities handle a disproportionate share of work,
creating operational risk. Unlike HHI which measures overall concentration,
SPOF detection focuses on individual entities that could disrupt operations.

Use cases:
- Judge handling 40% of patent cases (what if they retire?)
- One lawyer approving all discovery motions (vacation = bottleneck)
- Single law firm handling all government contracts (key man risk)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .hhi_calculator import HHICalculator, HHIResult


class RiskLevel(Enum):
    """SPOF risk severity levels."""
    LOW = "low"           # Share < 15%
    MODERATE = "moderate" # 15% <= share < 25%
    HIGH = "high"         # 25% <= share < 40%
    CRITICAL = "critical" # Share >= 40%


@dataclass
class SPOFRisk:
    """Risk assessment for a single entity."""
    entity_id: str
    share: float
    count: int
    risk_level: RiskLevel
    recommendation: str

    @property
    def is_spof(self) -> bool:
        """Check if this entity represents a SPOF."""
        return self.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}


@dataclass
class RiskAssessment:
    """Complete SPOF risk assessment for a domain."""
    entity_type: str  # "judge", "lawyer", "firm", etc.
    domain: str       # "patent", "employment", etc.
    total_volume: int
    entity_count: int

    # Concentration metrics
    hhi_result: HHIResult

    # Individual SPOF risks (sorted by severity)
    spof_risks: list[SPOFRisk] = field(default_factory=list)

    # Summary
    has_critical_spof: bool = False
    top_spof: Optional[SPOFRisk] = None

    @property
    def overall_health(self) -> str:
        """Get overall health assessment."""
        if self.has_critical_spof:
            return "CRITICAL - Immediate action required"
        elif any(r.risk_level == RiskLevel.HIGH for r in self.spof_risks):
            return "WARNING - High concentration detected"
        elif any(r.risk_level == RiskLevel.MODERATE for r in self.spof_risks):
            return "CAUTION - Monitor concentration levels"
        else:
            return "HEALTHY - Well-distributed workload"


class SPOFDetector:
    """
    Detects Single Point of Failure risks in entity distributions.

    Example usage:
        detector = SPOFDetector()

        # Analyze judge caseload distribution
        assessment = detector.analyze(
            entity_counts={"judge_a": 150, "judge_b": 80, "judge_c": 70},
            entity_type="judge",
            domain="patent"
        )

        print(assessment.overall_health)
        for risk in assessment.spof_risks:
            if risk.is_spof:
                print(f"SPOF: {risk.entity_id} - {risk.recommendation}")
    """

    # Default thresholds (can be customized)
    THRESHOLD_LOW = 0.15      # Below 15% is low risk
    THRESHOLD_MODERATE = 0.25 # 15-25% is moderate
    THRESHOLD_HIGH = 0.40     # 25-40% is high
    # Above 40% is critical

    def __init__(
        self,
        threshold_moderate: float = 0.25,
        threshold_high: float = 0.40
    ):
        """
        Initialize detector with custom thresholds if needed.

        Args:
            threshold_moderate: Share threshold for moderate risk (default 25%)
            threshold_high: Share threshold for high/critical risk (default 40%)
        """
        self.threshold_moderate = threshold_moderate
        self.threshold_high = threshold_high
        self.hhi_calc = HHICalculator()

    def analyze(
        self,
        entity_counts: dict[str, int | float],
        entity_type: str = "entity",
        domain: str = "general"
    ) -> RiskAssessment:
        """
        Perform complete SPOF risk analysis.

        Args:
            entity_counts: Dictionary mapping entity ID to count/volume
            entity_type: Type of entity (judge, lawyer, firm)
            domain: Domain context (patent, employment, etc.)

        Returns:
            RiskAssessment with complete analysis
        """
        if not entity_counts:
            return self._empty_assessment(entity_type, domain)

        total = sum(entity_counts.values())
        if total == 0:
            return self._empty_assessment(entity_type, domain)

        # Calculate HHI
        hhi_result = self.hhi_calc.from_counts(entity_counts)

        # Assess each entity
        spof_risks = []
        for entity_id, count in sorted(
            entity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            share = count / total
            risk_level = self._assess_risk_level(share)
            recommendation = self._generate_recommendation(
                entity_id, entity_type, share, risk_level
            )

            spof_risks.append(SPOFRisk(
                entity_id=entity_id,
                share=round(share, 4),
                count=int(count),
                risk_level=risk_level,
                recommendation=recommendation
            ))

        # Identify critical SPOFs
        has_critical = any(r.risk_level == RiskLevel.CRITICAL for r in spof_risks)
        top_spof = spof_risks[0] if spof_risks else None

        return RiskAssessment(
            entity_type=entity_type,
            domain=domain,
            total_volume=int(total),
            entity_count=len(entity_counts),
            hhi_result=hhi_result,
            spof_risks=spof_risks,
            has_critical_spof=has_critical,
            top_spof=top_spof
        )

    def _assess_risk_level(self, share: float) -> RiskLevel:
        """Determine risk level from share."""
        if share >= self.threshold_high:
            return RiskLevel.CRITICAL
        elif share >= self.threshold_moderate:
            return RiskLevel.HIGH
        elif share >= self.THRESHOLD_LOW:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _generate_recommendation(
        self,
        entity_id: str,
        entity_type: str,
        share: float,
        risk_level: RiskLevel
    ) -> str:
        """Generate actionable recommendation for an entity."""
        pct = f"{share:.0%}"

        if risk_level == RiskLevel.CRITICAL:
            return (
                f"CRITICAL: {entity_id} handles {pct} of {entity_type} workload. "
                f"Immediately develop backup capacity and cross-training plan. "
                f"Consider redistributing new cases to other {entity_type}s."
            )
        elif risk_level == RiskLevel.HIGH:
            return (
                f"HIGH RISK: {entity_id} handles {pct} of workload. "
                f"Develop succession/backup plan and begin gradual redistribution."
            )
        elif risk_level == RiskLevel.MODERATE:
            return (
                f"MODERATE: {entity_id} at {pct}. Monitor and prevent further concentration."
            )
        else:
            return f"LOW: {entity_id} at {pct}. Healthy distribution level."

    def _empty_assessment(self, entity_type: str, domain: str) -> RiskAssessment:
        """Return assessment for empty input."""
        return RiskAssessment(
            entity_type=entity_type,
            domain=domain,
            total_volume=0,
            entity_count=0,
            hhi_result=self.hhi_calc.from_counts({}),
            spof_risks=[],
            has_critical_spof=False,
            top_spof=None
        )

    def compare_periods(
        self,
        period1_counts: dict[str, int | float],
        period2_counts: dict[str, int | float],
        entity_type: str = "entity",
        domain: str = "general"
    ) -> dict:
        """
        Compare concentration between two time periods.

        Args:
            period1_counts: Entity counts for first period
            period2_counts: Entity counts for second period
            entity_type: Type of entity
            domain: Domain context

        Returns:
            Dictionary with comparison results
        """
        assessment1 = self.analyze(period1_counts, entity_type, domain)
        assessment2 = self.analyze(period2_counts, entity_type, domain)

        hhi_delta = assessment2.hhi_result.hhi - assessment1.hhi_result.hhi

        # Identify new SPOFs
        spof_ids_1 = {r.entity_id for r in assessment1.spof_risks if r.is_spof}
        spof_ids_2 = {r.entity_id for r in assessment2.spof_risks if r.is_spof}

        new_spofs = spof_ids_2 - spof_ids_1
        resolved_spofs = spof_ids_1 - spof_ids_2

        return {
            "period1": assessment1,
            "period2": assessment2,
            "hhi_delta": round(hhi_delta, 2),
            "concentration_trend": "increasing" if hhi_delta > 100 else (
                "decreasing" if hhi_delta < -100 else "stable"
            ),
            "new_spofs": list(new_spofs),
            "resolved_spofs": list(resolved_spofs),
        }
