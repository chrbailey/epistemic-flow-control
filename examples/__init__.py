"""
Epistemic Flow Control - Example Datasets

This module provides compelling example data that demonstrates the system's
capabilities through realistic judicial domain scenarios.

Story Arcs:
-----------
1. The Changing Judge (Rodriguez) - Shows temporal decay and pattern shifts
2. The Reliable Constant (Chen) - Demonstrates stable Bayesian weights
3. The New Judge (Martinez) - Illustrates Wilson score intervals for small samples
4. Context Matters (Lee) - Shows context-sensitive pattern behavior
5. Source Degradation (Patent Buddy Blog) - Demonstrates reliability tracking

New Feature Examples:
--------------------
- Normalization Cases - Messy court data cleaning examples
- Concentration Scenarios - HHI and SPOF risk demonstrations
- Drift Scenarios - Pattern drift detection examples
- Jurisdictions Data - Court and judge comparison data

Usage:
------
    from examples import load_all_examples
    from unified_system import EpistemicFlowControl, SystemConfig

    config = SystemConfig(db_dir="./demo_data", domain="judicial")
    system = EpistemicFlowControl(config)
    load_all_examples(system)
"""

from .sources import DEMO_SOURCES
from .judges import JUDGE_PROFILES
from .events import DEMO_EVENTS, EVENT_STORIES
from .data_loader import ExampleDataLoader, load_all_examples

# New feature examples
from .normalization_cases import (
    JUDGE_VARIATIONS,
    INVALID_LAWYER_ENTRIES,
    VALID_LAWYER_ENTRIES,
    get_normalization_demo_data,
)
from .concentration_scenarios import (
    get_all_scenarios as get_concentration_scenarios,
    get_time_series_data as get_concentration_timeline,
    HEALTHY_DISTRIBUTION,
    HIGH_CONCENTRATION,
)
from .drift_scenarios import (
    get_all_drift_scenarios,
    get_baseline as get_drift_baseline,
    get_drift_timeline,
)
from .jurisdictions_data import (
    COURT_COMPARISON,
    JUDGE_PROFILES as JURISDICTION_JUDGE_PROFILES,
    get_court_data,
    get_judge_profile,
)

__all__ = [
    # Original exports
    "DEMO_SOURCES",
    "JUDGE_PROFILES",
    "DEMO_EVENTS",
    "EVENT_STORIES",
    "ExampleDataLoader",
    "load_all_examples",
    # Normalization
    "JUDGE_VARIATIONS",
    "INVALID_LAWYER_ENTRIES",
    "VALID_LAWYER_ENTRIES",
    "get_normalization_demo_data",
    # Concentration
    "get_concentration_scenarios",
    "get_concentration_timeline",
    "HEALTHY_DISTRIBUTION",
    "HIGH_CONCENTRATION",
    # Drift
    "get_all_drift_scenarios",
    "get_drift_baseline",
    "get_drift_timeline",
    # Jurisdictions
    "COURT_COMPARISON",
    "JURISDICTION_JUDGE_PROFILES",
    "get_court_data",
    "get_judge_profile",
]
