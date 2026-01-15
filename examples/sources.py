"""
Example Information Sources

Demonstrates source reliability tracking with a spectrum from
highly reliable (PACER: 0.99) to unreliable (Twitter: 0.40).
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class DemoSource:
    """A demonstration information source."""
    source_id: str
    name: str
    source_type: str  # "official", "professional", "news", "social"
    base_reliability: float
    description: str
    example_content: str
    reliability_history: Optional[List[dict]] = None


DEMO_SOURCES = [
    # Tier 1: Official Sources (0.95-0.99)
    DemoSource(
        source_id="pacer",
        name="PACER (Public Access to Court Electronic Records)",
        source_type="official",
        base_reliability=0.99,
        description="Official federal court electronic records system",
        example_content="Case 2:24-cv-00123 - Order granting summary judgment",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.99, "reason": "Initial assessment"},
        ]
    ),
    DemoSource(
        source_id="ecf",
        name="CM/ECF (Case Management/Electronic Case Files)",
        source_type="official",
        base_reliability=0.98,
        description="Federal court case management system",
        example_content="Document 45: Motion for Summary Judgment [GRANTED]",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.98, "reason": "Initial assessment"},
        ]
    ),

    # Tier 2: Professional Sources (0.80-0.94)
    DemoSource(
        source_id="law360",
        name="Law360",
        source_type="professional",
        base_reliability=0.92,
        description="Premium legal news service with reporter verification",
        example_content="Judge Chen Denies TechCorp's Summary Judgment Bid in Patent Dispute",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.92, "reason": "Initial assessment"},
            {"date": "2024-06-15", "reliability": 0.93, "reason": "Consistent accuracy on 50+ cases"},
        ]
    ),
    DemoSource(
        source_id="reuters_legal",
        name="Reuters Legal",
        source_type="professional",
        base_reliability=0.90,
        description="Major news wire legal coverage",
        example_content="Federal judge sides with defendant in high-profile patent case",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.90, "reason": "Initial assessment"},
        ]
    ),
    DemoSource(
        source_id="lexis_news",
        name="LexisNexis Legal News",
        source_type="professional",
        base_reliability=0.88,
        description="Legal database news aggregation",
        example_content="E.D. Texas: Summary judgment granted in Acme v. Beta Corp",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.88, "reason": "Initial assessment"},
        ]
    ),

    # Tier 3: News Sources (0.60-0.79)
    DemoSource(
        source_id="techcrunch",
        name="TechCrunch",
        source_type="news",
        base_reliability=0.72,
        description="Tech news with occasional legal coverage",
        example_content="Startup wins patent battle against tech giant",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.75, "reason": "Initial assessment"},
            {"date": "2024-09-01", "reliability": 0.72, "reason": "Misreported 2 case outcomes"},
        ]
    ),
    DemoSource(
        source_id="ars_technica",
        name="Ars Technica",
        source_type="news",
        base_reliability=0.70,
        description="Tech news with detailed legal analysis",
        example_content="Judge rules against patent troll in landmark decision",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.70, "reason": "Initial assessment"},
        ]
    ),

    # Tier 4: Blog Sources (0.50-0.69) - DEMONSTRATES DEGRADATION
    DemoSource(
        source_id="patent_buddy_blog",
        name="Patent Buddy Blog",
        source_type="blog",
        base_reliability=0.65,
        description="Independent patent analysis blog",
        example_content="BREAKING: Judge Rodriguez expected to rule for plaintiff!",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.65, "reason": "Initial assessment"},
            {"date": "2024-04-15", "reliability": 0.58, "reason": "Wrong prediction on Rodriguez ruling"},
            {"date": "2024-07-20", "reliability": 0.52, "reason": "Another incorrect prediction"},
            {"date": "2024-10-01", "reliability": 0.45, "reason": "Third wrong prediction - downgraded"},
        ]
    ),

    # Tier 5: Social Media (0.30-0.49)
    DemoSource(
        source_id="twitter_legal",
        name="Twitter/X Legal Commentary",
        source_type="social",
        base_reliability=0.40,
        description="Social media legal commentary - unverified",
        example_content="Just heard Judge Chen is definitely ruling tomorrow! #PatentLaw",
        reliability_history=[
            {"date": "2024-01-01", "reliability": 0.40, "reason": "Initial - social media baseline"},
        ]
    ),
]


def get_source_by_id(source_id: str) -> Optional[DemoSource]:
    """Get a demo source by its ID."""
    for source in DEMO_SOURCES:
        if source.source_id == source_id:
            return source
    return None


def get_sources_by_reliability(min_reliability: float = 0.0) -> List[DemoSource]:
    """Get all sources above a reliability threshold."""
    return [s for s in DEMO_SOURCES if s.base_reliability >= min_reliability]


def get_reliability_spectrum() -> dict:
    """Get sources organized by reliability tier for visualization."""
    return {
        "official": [s for s in DEMO_SOURCES if s.source_type == "official"],
        "professional": [s for s in DEMO_SOURCES if s.source_type == "professional"],
        "news": [s for s in DEMO_SOURCES if s.source_type == "news"],
        "blog": [s for s in DEMO_SOURCES if s.source_type == "blog"],
        "social": [s for s in DEMO_SOURCES if s.source_type == "social"],
    }
