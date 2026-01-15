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

__all__ = [
    "DEMO_SOURCES",
    "JUDGE_PROFILES",
    "DEMO_EVENTS",
    "EVENT_STORIES",
    "ExampleDataLoader",
    "load_all_examples",
]
