"""
Example Data Loader

Utility to load all example data into an EpistemicFlowControl system.
"""

from datetime import datetime
from typing import Optional

from .sources import DEMO_SOURCES
from .judges import JUDGE_PROFILES
from .events import DEMO_EVENTS, EVENT_STORIES


class ExampleDataLoader:
    """
    Loads example data into an EpistemicFlowControl system.

    Usage:
        from unified_system import EpistemicFlowControl, SystemConfig
        from examples import ExampleDataLoader

        config = SystemConfig(db_dir="./demo_data", domain="judicial")
        system = EpistemicFlowControl(config)

        loader = ExampleDataLoader(system)
        loader.load_all_examples()
    """

    def __init__(self, system):
        """
        Initialize the loader with a system instance.

        Args:
            system: An EpistemicFlowControl instance
        """
        self.system = system
        self.stats = {
            "sources_loaded": 0,
            "events_loaded": 0,
            "patterns_extracted": 0,
            "errors": [],
        }

    def load_sources(self) -> int:
        """Load all demo sources into the system."""
        count = 0
        for source in DEMO_SOURCES:
            try:
                success = self.system.register_source(
                    source_id=source.source_id,
                    name=source.name,
                    source_type=source.source_type,
                    reliability=source.base_reliability,
                    notes=source.description
                )
                if success:
                    count += 1
            except Exception as e:
                self.stats["errors"].append(f"Source {source.source_id}: {e}")

        self.stats["sources_loaded"] = count
        return count

    def load_events(self, extract_patterns: bool = True) -> int:
        """
        Load all demo events into the system.

        Args:
            extract_patterns: Whether to automatically extract patterns

        Returns:
            Number of events successfully loaded
        """
        count = 0
        patterns_count = 0

        for event in DEMO_EVENTS:
            try:
                result = self.system.ingest_event(
                    what=event.what,
                    who=event.who,
                    when=event.when,
                    where=event.where,
                    source_id=event.source_id,
                    raw_text=event.raw_text,
                    why=event.why,
                    how=event.how,
                    event_type=event.event_type,
                    auto_extract_patterns=extract_patterns
                )

                if result.get("success"):
                    count += 1
                    patterns_count += len(result.get("patterns_extracted", []))
                else:
                    self.stats["errors"].append(
                        f"Event {event.event_id}: {result.get('message')}"
                    )

            except Exception as e:
                self.stats["errors"].append(f"Event {event.event_id}: {e}")

        self.stats["events_loaded"] = count
        self.stats["patterns_extracted"] = patterns_count
        return count

    def load_reviewers(self) -> int:
        """Load demo reviewers for the review gate."""
        reviewers = [
            {
                "reviewer_id": "expert_patent",
                "name": "Dr. Sarah Patent Expert",
                "role": "expert",
                "domains": ["judicial", "patent"]
            },
            {
                "reviewer_id": "analyst_01",
                "name": "Alex Analyst",
                "role": "standard",
                "domains": ["judicial"]
            },
            {
                "reviewer_id": "admin_01",
                "name": "Admin User",
                "role": "admin",
                "domains": ["judicial", "patent", "contract"]
            },
        ]

        count = 0
        for r in reviewers:
            try:
                success = self.system.register_reviewer(
                    reviewer_id=r["reviewer_id"],
                    name=r["name"],
                    role=r["role"],
                    domains=r["domains"]
                )
                if success:
                    count += 1
            except Exception as e:
                self.stats["errors"].append(f"Reviewer {r['reviewer_id']}: {e}")

        return count

    def load_all_examples(self) -> dict:
        """
        Load all example data: sources, events, patterns, and reviewers.

        Returns:
            Dictionary with loading statistics
        """
        print("Loading example data...")

        # Load in order: sources first (needed for events)
        print("  Loading sources...")
        sources = self.load_sources()
        print(f"    Loaded {sources} sources")

        # Load reviewers
        print("  Loading reviewers...")
        reviewers = self.load_reviewers()
        print(f"    Loaded {reviewers} reviewers")

        # Load events (will extract patterns)
        print("  Loading events and extracting patterns...")
        events = self.load_events(extract_patterns=True)
        print(f"    Loaded {events} events")
        print(f"    Extracted {self.stats['patterns_extracted']} patterns")

        if self.stats["errors"]:
            print(f"\n  Warnings: {len(self.stats['errors'])} issues encountered")
            for error in self.stats["errors"][:5]:  # Show first 5
                print(f"    - {error}")
            if len(self.stats["errors"]) > 5:
                print(f"    ... and {len(self.stats['errors']) - 5} more")

        print("\nExample data loaded successfully!")
        return self.stats

    def print_summary(self):
        """Print a summary of loaded data."""
        print("\n" + "=" * 60)
        print("EXAMPLE DATA SUMMARY")
        print("=" * 60)

        print(f"\nSources: {self.stats['sources_loaded']}")
        print(f"Events:  {self.stats['events_loaded']}")
        print(f"Patterns: {self.stats['patterns_extracted']}")

        print("\nStory Arcs Available:")
        for name, story in EVENT_STORIES.items():
            print(f"  - {name}")
            print(f"    Insight: {story['insight'][:60]}...")
            print(f"    Key Metric: {story['key_metric']}")

        print("\nJudge Profiles:")
        for judge in JUDGE_PROFILES:
            print(f"  - {judge.name} ({judge.court})")
            print(f"    {judge.total_cases} cases, {judge.summary_judgment_grant_rate:.0%} SJ grant rate")
            print(f"    Demonstrates: {judge.demonstrates[:50]}...")

        print("\n" + "=" * 60)


def load_all_examples(system) -> dict:
    """
    Convenience function to load all examples.

    Args:
        system: An EpistemicFlowControl instance

    Returns:
        Loading statistics dictionary
    """
    loader = ExampleDataLoader(system)
    return loader.load_all_examples()


# CLI entry point
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    from unified_system import EpistemicFlowControl, SystemConfig

    # Create system with demo data directory
    config = SystemConfig(db_dir="./demo_data", domain="judicial")
    system = EpistemicFlowControl(config)

    # Load all examples
    loader = ExampleDataLoader(system)
    loader.load_all_examples()
    loader.print_summary()

    print("\nDemo data is ready! Run the Streamlit demo to explore.")
