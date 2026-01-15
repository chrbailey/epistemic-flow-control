"""
Herfindahl-Hirschman Index (HHI) Calculator

The HHI measures market concentration by summing the squares of market shares.
Originally used in antitrust economics, it's perfect for detecting when
legal work is concentrated with specific entities.

Scale (0-10,000):
- 0: Perfect competition (infinite equal participants)
- 1,500: Unconcentrated market (DOJ threshold)
- 2,500: Moderately concentrated
- 10,000: Pure monopoly (one participant)

For legal analytics:
- Low HHI: Work distributed across many judges/lawyers (healthy)
- High HHI: Work concentrated with few entities (SPOF risk)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConcentrationLevel(Enum):
    """DOJ-inspired concentration thresholds."""
    UNCONCENTRATED = "unconcentrated"       # HHI < 1,500
    MODERATE = "moderate"                    # 1,500 <= HHI < 2,500
    CONCENTRATED = "concentrated"            # HHI >= 2,500
    HIGHLY_CONCENTRATED = "highly_concentrated"  # HHI >= 5,000
    MONOPOLISTIC = "monopolistic"            # HHI >= 7,500


@dataclass
class HHIResult:
    """Result of HHI calculation."""
    hhi: float  # 0-10,000 scale
    level: ConcentrationLevel
    participant_count: int
    top_share: float  # Largest participant's share
    top_entity: Optional[str]  # ID of largest participant
    equivalent_firms: float  # 1/HHI * 10000 - conceptual "equal firms"

    @property
    def is_healthy(self) -> bool:
        """Check if concentration is at a healthy level."""
        return self.level in {ConcentrationLevel.UNCONCENTRATED, ConcentrationLevel.MODERATE}

    @property
    def normalized(self) -> float:
        """Return HHI normalized to 0-1 scale."""
        return self.hhi / 10000.0


class HHICalculator:
    """
    Calculates the Herfindahl-Hirschman Index for concentration analysis.

    Example usage:
        calc = HHICalculator()

        # From raw counts
        result = calc.from_counts({"judge_a": 50, "judge_b": 30, "judge_c": 20})
        print(f"HHI: {result.hhi}, Level: {result.level}")

        # From shares (must sum to 1.0)
        result = calc.from_shares({"judge_a": 0.5, "judge_b": 0.3, "judge_c": 0.2})
    """

    # DOJ merger guidelines thresholds
    THRESHOLD_UNCONCENTRATED = 1500
    THRESHOLD_MODERATE = 2500
    THRESHOLD_HIGH = 5000
    THRESHOLD_MONOPOLISTIC = 7500

    def from_counts(
        self,
        entity_counts: dict[str, int | float]
    ) -> HHIResult:
        """
        Calculate HHI from raw counts per entity.

        Args:
            entity_counts: Dictionary mapping entity ID to count/volume

        Returns:
            HHIResult with concentration metrics
        """
        if not entity_counts:
            return self._empty_result()

        total = sum(entity_counts.values())
        if total == 0:
            return self._empty_result()

        # Convert to shares
        shares = {k: v / total for k, v in entity_counts.items()}
        return self.from_shares(shares)

    def from_shares(
        self,
        entity_shares: dict[str, float]
    ) -> HHIResult:
        """
        Calculate HHI from market shares.

        Args:
            entity_shares: Dictionary mapping entity ID to share (should sum to ~1.0)

        Returns:
            HHIResult with concentration metrics
        """
        if not entity_shares:
            return self._empty_result()

        # Normalize shares to ensure they sum to 1.0
        total = sum(entity_shares.values())
        if total == 0:
            return self._empty_result()

        normalized = {k: v / total for k, v in entity_shares.items()}

        # Calculate HHI: sum of squared shares * 10,000
        hhi = sum(share ** 2 for share in normalized.values()) * 10000

        # Find top participant
        top_entity = max(normalized, key=normalized.get)
        top_share = normalized[top_entity]

        # Determine concentration level
        level = self._get_level(hhi)

        # Equivalent number of equal-sized firms
        # If n firms each have 1/n share, HHI = n * (1/n)^2 * 10000 = 10000/n
        # So equivalent_firms = 10000/HHI
        equivalent_firms = 10000 / hhi if hhi > 0 else float("inf")

        return HHIResult(
            hhi=round(hhi, 2),
            level=level,
            participant_count=len(entity_shares),
            top_share=round(top_share, 4),
            top_entity=top_entity,
            equivalent_firms=round(equivalent_firms, 2)
        )

    def from_list(self, values: list[int | float]) -> HHIResult:
        """
        Calculate HHI from a list of values (anonymous entities).

        Args:
            values: List of counts/volumes per entity

        Returns:
            HHIResult with concentration metrics
        """
        entity_counts = {f"entity_{i}": v for i, v in enumerate(values)}
        return self.from_counts(entity_counts)

    def _get_level(self, hhi: float) -> ConcentrationLevel:
        """Determine concentration level from HHI value."""
        if hhi >= self.THRESHOLD_MONOPOLISTIC:
            return ConcentrationLevel.MONOPOLISTIC
        elif hhi >= self.THRESHOLD_HIGH:
            return ConcentrationLevel.HIGHLY_CONCENTRATED
        elif hhi >= self.THRESHOLD_MODERATE:
            return ConcentrationLevel.CONCENTRATED
        elif hhi >= self.THRESHOLD_UNCONCENTRATED:
            return ConcentrationLevel.MODERATE
        else:
            return ConcentrationLevel.UNCONCENTRATED

    def _empty_result(self) -> HHIResult:
        """Return result for empty input."""
        return HHIResult(
            hhi=0.0,
            level=ConcentrationLevel.UNCONCENTRATED,
            participant_count=0,
            top_share=0.0,
            top_entity=None,
            equivalent_firms=float("inf")
        )

    def calculate_delta(self, before: HHIResult, after: HHIResult) -> float:
        """
        Calculate the change in HHI (useful for merger analysis).

        Args:
            before: HHI before a change
            after: HHI after a change

        Returns:
            Delta HHI (positive means increased concentration)
        """
        return after.hhi - before.hhi

    def merge_simulation(
        self,
        current_shares: dict[str, float],
        entities_to_merge: list[str],
        merged_name: str = "merged_entity"
    ) -> tuple[HHIResult, HHIResult, float]:
        """
        Simulate the effect of merging entities on concentration.

        Args:
            current_shares: Current market shares
            entities_to_merge: List of entity IDs to merge
            merged_name: Name for the merged entity

        Returns:
            Tuple of (before_result, after_result, delta_hhi)
        """
        before = self.from_shares(current_shares)

        # Create post-merger shares
        merged_share = sum(
            current_shares.get(e, 0) for e in entities_to_merge
        )
        after_shares = {
            k: v for k, v in current_shares.items()
            if k not in entities_to_merge
        }
        if merged_share > 0:
            after_shares[merged_name] = merged_share

        after = self.from_shares(after_shares)
        delta = self.calculate_delta(before, after)

        return before, after, delta
