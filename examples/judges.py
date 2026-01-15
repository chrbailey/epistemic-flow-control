"""
Example Judge Profiles

Five judges with distinct behavioral profiles demonstrating different
aspects of the epistemic flow control system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class JudgeProfile:
    """A demonstration judge profile with behavioral patterns."""
    judge_id: str
    name: str
    court: str
    description: str
    story_arc: str

    # Pattern data
    summary_judgment_grant_rate: float
    total_cases: int
    specialty_areas: List[str]

    # For demonstrating evolution
    historical_rates: List[Dict] = field(default_factory=list)

    # Key insight this judge demonstrates
    demonstrates: str = ""


JUDGE_PROFILES = [
    # Story 1: The Changing Judge - Demonstrates Temporal Decay
    JudgeProfile(
        judge_id="rodriguez",
        name="Judge Maria Rodriguez",
        court="N.D. California",
        description="Experienced judge who became Chief Judge mid-2024",
        story_arc="The Changing Judge",
        summary_judgment_grant_rate=0.42,  # Current rate (after becoming Chief)
        total_cases=89,
        specialty_areas=["patent", "trade_secret", "antitrust"],
        historical_rates=[
            {"period": "2023-Q1", "rate": 0.78, "cases": 18, "note": "High grant rate period"},
            {"period": "2023-Q2", "rate": 0.75, "cases": 15, "note": "Consistent pattern"},
            {"period": "2023-Q3", "rate": 0.72, "cases": 12, "note": "Slight decline"},
            {"period": "2023-Q4", "rate": 0.68, "cases": 14, "note": "Trend continuing"},
            {"period": "2024-Q1", "rate": 0.55, "cases": 10, "note": "Became Chief Judge - workload shift"},
            {"period": "2024-Q2", "rate": 0.45, "cases": 8, "note": "New pattern emerging"},
            {"period": "2024-Q3", "rate": 0.42, "cases": 7, "note": "Stabilizing at new baseline"},
            {"period": "2024-Q4", "rate": 0.40, "cases": 5, "note": "Current pattern"},
        ],
        demonstrates="Temporal decay: Old patterns (78% grant rate) become stale. "
                    "System must recognize the shift and recalibrate. "
                    "Predictions based on 2023 data would be dangerously wrong."
    ),

    # Story 2: The Reliable Constant - Demonstrates Stable Bayesian Weights
    JudgeProfile(
        judge_id="chen",
        name="Judge William Chen",
        court="E.D. Texas",
        description="Veteran patent judge with highly predictable patterns",
        story_arc="The Reliable Constant",
        summary_judgment_grant_rate=0.32,
        total_cases=156,
        specialty_areas=["patent"],
        historical_rates=[
            {"period": "2022-Q1", "rate": 0.30, "cases": 12, "note": "Baseline established"},
            {"period": "2022-Q2", "rate": 0.33, "cases": 14, "note": "Slight variation"},
            {"period": "2022-Q3", "rate": 0.31, "cases": 11, "note": "Return to mean"},
            {"period": "2022-Q4", "rate": 0.32, "cases": 13, "note": "Stable"},
            {"period": "2023-Q1", "rate": 0.34, "cases": 15, "note": "Minor uptick"},
            {"period": "2023-Q2", "rate": 0.31, "cases": 12, "note": "Correction"},
            {"period": "2023-Q3", "rate": 0.32, "cases": 14, "note": "Stable"},
            {"period": "2023-Q4", "rate": 0.33, "cases": 16, "note": "Stable"},
            {"period": "2024-Q1", "rate": 0.31, "cases": 13, "note": "Stable"},
            {"period": "2024-Q2", "rate": 0.32, "cases": 18, "note": "Stable"},
            {"period": "2024-Q3", "rate": 0.32, "cases": 11, "note": "Highly predictable"},
            {"period": "2024-Q4", "rate": 0.33, "cases": 7, "note": "Continues pattern"},
        ],
        demonstrates="Stable Bayesian weights: 156 cases with 32% grant rate means "
                    "very narrow confidence interval. System can AUTO_PASS predictions "
                    "about this judge because the pattern is rock-solid."
    ),

    # Story 3: The New Judge - Demonstrates Wilson Score for Small Samples
    JudgeProfile(
        judge_id="martinez",
        name="Judge Sofia Martinez",
        court="C.D. California",
        description="Recently appointed judge with limited case history",
        story_arc="The New Judge",
        summary_judgment_grant_rate=0.50,  # 4/8 - but what's the real rate?
        total_cases=8,
        specialty_areas=["patent", "copyright"],
        historical_rates=[
            {"period": "2024-Q3", "rate": 0.67, "cases": 3, "note": "First cases - 2/3 granted"},
            {"period": "2024-Q4", "rate": 0.40, "cases": 5, "note": "2/5 granted - pattern unclear"},
        ],
        demonstrates="Wilson score intervals: Raw rate is 50% (4/8), but Wilson lower bound "
                    "is only 21.5%. The TRUE rate could be anywhere from 21% to 79%. "
                    "System correctly routes to REVIEW_REQUIRED due to uncertainty."
    ),

    # Story 4: Context Matters - Demonstrates Contextual Patterns
    JudgeProfile(
        judge_id="lee",
        name="Judge David Lee",
        court="D. Delaware",
        description="Judge with notably different behavior by patent type",
        story_arc="Context Matters",
        summary_judgment_grant_rate=0.40,  # Overall, but misleading!
        total_cases=75,
        specialty_areas=["patent"],
        historical_rates=[
            # Software patents - very different from hardware
            {"period": "software_patents", "rate": 0.25, "cases": 40,
             "note": "Low grant rate for software - fact-intensive"},
            {"period": "hardware_patents", "rate": 0.58, "cases": 35,
             "note": "High grant rate for hardware - clearer boundaries"},
        ],
        demonstrates="Context-sensitive patterns: Overall 40% rate is MISLEADING. "
                    "For software patents it's 25%, for hardware it's 58%. "
                    "System must track sub-patterns, not just aggregate."
    ),

    # Story 5: The Learning Curve - Demonstrates Gradual Evolution
    JudgeProfile(
        judge_id="albright",
        name="Judge Thomas Albright",
        court="W.D. Texas",
        description="Judge whose patterns have gradually evolved over time",
        story_arc="The Learning Curve",
        summary_judgment_grant_rate=0.55,  # Current
        total_cases=112,
        specialty_areas=["patent"],
        historical_rates=[
            {"period": "2020", "rate": 0.70, "cases": 25, "note": "Early career - higher grants"},
            {"period": "2021", "rate": 0.68, "cases": 28, "note": "Slight decline"},
            {"period": "2022", "rate": 0.62, "cases": 24, "note": "Trend continuing"},
            {"period": "2023", "rate": 0.58, "cases": 22, "note": "Gradual shift"},
            {"period": "2024", "rate": 0.55, "cases": 13, "note": "Current baseline"},
        ],
        demonstrates="Bayesian updating with gradual drift: Pattern evolved from 70% to 55% "
                    "over 4 years. System correctly updates weights with each new case, "
                    "tracking the slow evolution rather than assuming stationarity."
    ),
]


def get_judge_by_id(judge_id: str) -> Optional[JudgeProfile]:
    """Get a judge profile by ID."""
    for judge in JUDGE_PROFILES:
        if judge.judge_id == judge_id:
            return judge
    return None


def get_judges_for_story(story_arc: str) -> List[JudgeProfile]:
    """Get judges that demonstrate a particular story arc."""
    return [j for j in JUDGE_PROFILES if j.story_arc == story_arc]


def calculate_wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    """
    Calculate Wilson score lower bound.

    Demonstrates the statistical foundation:
    - 4/8 successes: raw=50%, Wilson lower=21.5%
    - 50/156 successes: raw=32%, Wilson lower=25.1%

    More data = tighter confidence interval.
    """
    if total == 0:
        return 0.0

    import math
    p = successes / total
    denominator = 1 + z*z/total
    center = p + z*z/(2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)

    return max(0.0, (center - spread) / denominator)


def get_confidence_comparison() -> Dict:
    """
    Compare confidence intervals across judges to demonstrate
    the importance of sample size.
    """
    comparisons = {}

    for judge in JUDGE_PROFILES:
        successes = int(judge.summary_judgment_grant_rate * judge.total_cases)
        wilson_lower = calculate_wilson_lower(successes, judge.total_cases)

        comparisons[judge.judge_id] = {
            "name": judge.name,
            "raw_rate": judge.summary_judgment_grant_rate,
            "total_cases": judge.total_cases,
            "wilson_lower": wilson_lower,
            "confidence_width": judge.summary_judgment_grant_rate - wilson_lower,
            "can_auto_pass": wilson_lower > 0.25,  # Arbitrary threshold for demo
        }

    return comparisons
