"""
Example Events Dataset

50+ interconnected events across 6 story arcs demonstrating:
- Pattern extraction from real events
- Bayesian weight updates
- Temporal evolution
- Source reliability tracking
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional


@dataclass
class DemoEvent:
    """A demonstration event with all the 5W1H fields."""
    event_id: str
    who: List[str]
    what: str
    when: datetime
    where: str
    why: Optional[str]
    how: Optional[str]
    source_id: str
    raw_text: str
    event_type: str
    story_arc: str  # Which story this event belongs to


# Base date for generating realistic dates
BASE_DATE = datetime(2024, 1, 1)


def _date(days_offset: int) -> datetime:
    """Generate a date relative to BASE_DATE."""
    return BASE_DATE + timedelta(days=days_offset)


# ============================================================
# STORY 1: The Changing Judge (Rodriguez)
# Demonstrates temporal decay and pattern shifts
# ============================================================

RODRIGUEZ_EVENTS = [
    # Early period - high grant rate
    DemoEvent(
        event_id="evt_rodriguez_001",
        who=["Judge Maria Rodriguez", "TechStart Inc", "PatentHold LLC"],
        what="Granted summary judgment for defendant on all claims",
        when=_date(-365),  # 1 year ago
        where="N.D. Cal",
        why="Plaintiff failed to present genuine issues of material fact",
        how="Written order after oral argument",
        source_id="pacer",
        raw_text="ORDER: Defendant's motion for summary judgment is GRANTED...",
        event_type="summary_judgment",
        story_arc="The Changing Judge"
    ),
    DemoEvent(
        event_id="evt_rodriguez_002",
        who=["Judge Maria Rodriguez", "InnovateCo", "Legacy Patents Inc"],
        what="Granted summary judgment of non-infringement",
        when=_date(-350),
        where="N.D. Cal",
        why="Claims construed to exclude defendant's product",
        how="Written order",
        source_id="pacer",
        raw_text="The Court finds that under the proper construction of the claims...",
        event_type="summary_judgment",
        story_arc="The Changing Judge"
    ),
    DemoEvent(
        event_id="evt_rodriguez_003",
        who=["Judge Maria Rodriguez", "CloudSoft Corp", "DataGuard Inc"],
        what="Granted partial summary judgment on invalidity",
        when=_date(-320),
        where="N.D. Cal",
        why="Prior art clearly anticipates claims 1-5",
        how="Written order with claim chart",
        source_id="pacer",
        raw_text="Claims 1-5 are INVALID as anticipated by the Smith reference...",
        event_type="summary_judgment",
        story_arc="The Changing Judge"
    ),

    # Transition period - becoming Chief Judge
    DemoEvent(
        event_id="evt_rodriguez_004",
        who=["Judge Maria Rodriguez"],
        what="Appointed Chief Judge of N.D. California",
        when=_date(-180),
        where="N.D. Cal",
        why="Senior judge retirement created vacancy",
        how="Official appointment",
        source_id="ecf",
        raw_text="The Judicial Council announces the appointment of Judge Rodriguez...",
        event_type="appointment",
        story_arc="The Changing Judge"
    ),

    # Post-appointment - lower grant rate
    DemoEvent(
        event_id="evt_rodriguez_005",
        who=["Judge Maria Rodriguez", "MegaCorp", "SmallTech LLC"],
        what="Denied summary judgment motion",
        when=_date(-150),
        where="N.D. Cal",
        why="Genuine disputes of material fact exist regarding infringement",
        how="Written order - shorter than typical",
        source_id="pacer",
        raw_text="Plaintiff's motion for summary judgment is DENIED...",
        event_type="summary_judgment",
        story_arc="The Changing Judge"
    ),
    DemoEvent(
        event_id="evt_rodriguez_006",
        who=["Judge Maria Rodriguez", "AppDev Inc", "CodeBase Corp"],
        what="Denied summary judgment of invalidity",
        when=_date(-120),
        where="N.D. Cal",
        why="Expert testimony creates fact issues",
        how="Minute order",
        source_id="pacer",
        raw_text="The Court finds that the testimony of Dr. Smith creates...",
        event_type="summary_judgment",
        story_arc="The Changing Judge"
    ),
    DemoEvent(
        event_id="evt_rodriguez_007",
        who=["Judge Maria Rodriguez", "SecureTech", "HackShield Inc"],
        what="Denied cross-motions for summary judgment",
        when=_date(-90),
        where="N.D. Cal",
        why="Both parties have raised genuine factual disputes",
        how="Written order",
        source_id="pacer",
        raw_text="Both parties' motions for summary judgment are DENIED...",
        event_type="summary_judgment",
        story_arc="The Changing Judge"
    ),
    DemoEvent(
        event_id="evt_rodriguez_008",
        who=["Judge Maria Rodriguez", "AIVentures", "DeepMind Tech"],
        what="Granted in part, denied in part summary judgment",
        when=_date(-60),
        where="N.D. Cal",
        why="Some claims invalid, others present fact issues",
        how="Detailed written order",
        source_id="pacer",
        raw_text="Defendant's motion is GRANTED as to claims 1-3 and DENIED as to claims 4-8...",
        event_type="summary_judgment",
        story_arc="The Changing Judge"
    ),
]


# ============================================================
# STORY 2: The Reliable Constant (Chen)
# Demonstrates stable Bayesian weights
# ============================================================

CHEN_EVENTS = [
    DemoEvent(
        event_id="evt_chen_001",
        who=["Judge William Chen", "PharmaCo", "GenericDrugs Inc"],
        what="Denied summary judgment of invalidity",
        when=_date(-400),
        where="E.D. Texas",
        why="Secondary considerations create genuine fact issues",
        how="Written order",
        source_id="pacer",
        raw_text="The Court cannot conclude as a matter of law that the claims are invalid...",
        event_type="summary_judgment",
        story_arc="The Reliable Constant"
    ),
    DemoEvent(
        event_id="evt_chen_002",
        who=["Judge William Chen", "ChipMaker Corp", "SiliconValley Inc"],
        what="Denied summary judgment of non-infringement",
        when=_date(-380),
        where="E.D. Texas",
        why="Claim construction disputes remain",
        how="Written order",
        source_id="pacer",
        raw_text="Given the Court's claim construction, fact issues remain...",
        event_type="summary_judgment",
        story_arc="The Reliable Constant"
    ),
    DemoEvent(
        event_id="evt_chen_003",
        who=["Judge William Chen", "NetworkGiant", "StartupNet"],
        what="Granted summary judgment of invalidity",
        when=_date(-350),
        where="E.D. Texas",
        why="Claims anticipated by prior art reference",
        how="Written order with claim chart",
        source_id="pacer",
        raw_text="The Smith reference anticipates every element of claims 1-4...",
        event_type="summary_judgment",
        story_arc="The Reliable Constant"
    ),
    DemoEvent(
        event_id="evt_chen_004",
        who=["Judge William Chen", "MediaTech", "StreamCo"],
        what="Denied summary judgment",
        when=_date(-300),
        where="E.D. Texas",
        why="Expert credibility must be assessed by jury",
        how="Written order",
        source_id="pacer",
        raw_text="The competing expert opinions present classic jury questions...",
        event_type="summary_judgment",
        story_arc="The Reliable Constant"
    ),
    DemoEvent(
        event_id="evt_chen_005",
        who=["Judge William Chen", "DataCenter Corp", "CloudHost Inc"],
        what="Denied summary judgment of invalidity",
        when=_date(-250),
        where="E.D. Texas",
        why="Obviousness requires fact-finding on motivation to combine",
        how="Written order",
        source_id="pacer",
        raw_text="The motivation to combine the references is disputed...",
        event_type="summary_judgment",
        story_arc="The Reliable Constant"
    ),
    # Continue pattern - Chen is remarkably consistent
    DemoEvent(
        event_id="evt_chen_006",
        who=["Judge William Chen", "AutoTech", "EVMotors"],
        what="Denied summary judgment",
        when=_date(-200),
        where="E.D. Texas",
        why="Written description issues present fact questions",
        how="Written order",
        source_id="pacer",
        raw_text="Whether the specification provides adequate written description support...",
        event_type="summary_judgment",
        story_arc="The Reliable Constant"
    ),
    DemoEvent(
        event_id="evt_chen_007",
        who=["Judge William Chen", "BioGenetics", "LabCorp"],
        what="Granted summary judgment of non-infringement",
        when=_date(-150),
        where="E.D. Texas",
        why="Accused product clearly outside claim scope",
        how="Written order",
        source_id="pacer",
        raw_text="Under no reasonable construction do defendant's products infringe...",
        event_type="summary_judgment",
        story_arc="The Reliable Constant"
    ),
    DemoEvent(
        event_id="evt_chen_008",
        who=["Judge William Chen", "FinTech Solutions", "BankingApp"],
        what="Denied summary judgment",
        when=_date(-100),
        where="E.D. Texas",
        why="Alice/Mayo analysis requires factual development",
        how="Written order",
        source_id="pacer",
        raw_text="The Court cannot resolve the Section 101 issues at summary judgment...",
        event_type="summary_judgment",
        story_arc="The Reliable Constant"
    ),
]


# ============================================================
# STORY 3: The New Judge (Martinez)
# Demonstrates Wilson score for small samples
# ============================================================

MARTINEZ_EVENTS = [
    DemoEvent(
        event_id="evt_martinez_001",
        who=["Judge Sofia Martinez", "GameDev Inc", "PlayStore Corp"],
        what="Granted summary judgment of non-infringement",
        when=_date(-90),
        where="C.D. Cal",
        why="Claim construction precludes infringement",
        how="Written order - first patent case",
        source_id="pacer",
        raw_text="This Court's first patent summary judgment motion...",
        event_type="summary_judgment",
        story_arc="The New Judge"
    ),
    DemoEvent(
        event_id="evt_martinez_002",
        who=["Judge Sofia Martinez", "SocialApp", "ConnectPlatform"],
        what="Granted summary judgment of invalidity",
        when=_date(-75),
        where="C.D. Cal",
        why="Claims abstract under Alice",
        how="Written order",
        source_id="pacer",
        raw_text="The claims are directed to the abstract idea of organizing information...",
        event_type="summary_judgment",
        story_arc="The New Judge"
    ),
    DemoEvent(
        event_id="evt_martinez_003",
        who=["Judge Sofia Martinez", "AdTech", "MarketingAI"],
        what="Denied summary judgment",
        when=_date(-60),
        where="C.D. Cal",
        why="Fact issues on technical improvement",
        how="Written order",
        source_id="pacer",
        raw_text="Whether the claims recite a technical improvement is disputed...",
        event_type="summary_judgment",
        story_arc="The New Judge"
    ),
    DemoEvent(
        event_id="evt_martinez_004",
        who=["Judge Sofia Martinez", "RetailTech", "ShopOnline"],
        what="Denied summary judgment",
        when=_date(-45),
        where="C.D. Cal",
        why="Expert testimony in conflict",
        how="Written order",
        source_id="pacer",
        raw_text="The parties' experts present competing views that must be resolved at trial...",
        event_type="summary_judgment",
        story_arc="The New Judge"
    ),
    DemoEvent(
        event_id="evt_martinez_005",
        who=["Judge Sofia Martinez", "HealthApp", "MedDevice Co"],
        what="Granted summary judgment",
        when=_date(-30),
        where="C.D. Cal",
        why="No genuine fact dispute on infringement",
        how="Written order",
        source_id="pacer",
        raw_text="The undisputed evidence shows that defendant's product...",
        event_type="summary_judgment",
        story_arc="The New Judge"
    ),
    DemoEvent(
        event_id="evt_martinez_006",
        who=["Judge Sofia Martinez", "EduTech", "LearningSoft"],
        what="Denied summary judgment",
        when=_date(-15),
        where="C.D. Cal",
        why="Claim construction dispute affects infringement analysis",
        how="Written order",
        source_id="pacer",
        raw_text="Resolution of the infringement question turns on claim construction...",
        event_type="summary_judgment",
        story_arc="The New Judge"
    ),
    # Only 8 total cases - 4 granted, 4 denied = 50%
    # But Wilson lower bound = 21.5%!
    DemoEvent(
        event_id="evt_martinez_007",
        who=["Judge Sofia Martinez", "FoodDelivery", "QuickEats"],
        what="Granted summary judgment",
        when=_date(-7),
        where="C.D. Cal",
        why="Prior art invalidates claims",
        how="Written order",
        source_id="pacer",
        raw_text="The prior art clearly anticipates all asserted claims...",
        event_type="summary_judgment",
        story_arc="The New Judge"
    ),
    DemoEvent(
        event_id="evt_martinez_008",
        who=["Judge Sofia Martinez", "TravelApp", "BookingPlatform"],
        what="Denied summary judgment",
        when=_date(-3),
        where="C.D. Cal",
        why="Material facts disputed",
        how="Minute order",
        source_id="pacer",
        raw_text="Motion DENIED - fact issues preclude summary judgment...",
        event_type="summary_judgment",
        story_arc="The New Judge"
    ),
]


# ============================================================
# STORY 4: Context Matters (Lee)
# Demonstrates context-sensitive patterns
# ============================================================

LEE_EVENTS = [
    # Software patent cases - low grant rate (25%)
    DemoEvent(
        event_id="evt_lee_soft_001",
        who=["Judge David Lee", "SoftwareCo", "AppBuilder Inc"],
        what="Denied summary judgment - software patent",
        when=_date(-300),
        where="D. Delaware",
        why="Fact-intensive inquiry required for software patents",
        how="Written order",
        source_id="pacer",
        raw_text="Software patent cases often present unique fact questions...",
        event_type="summary_judgment",
        story_arc="Context Matters"
    ),
    DemoEvent(
        event_id="evt_lee_soft_002",
        who=["Judge David Lee", "CloudServices", "SaaSCorp"],
        what="Denied summary judgment - software patent",
        when=_date(-280),
        where="D. Delaware",
        why="Expert testimony on technical implementation disputed",
        how="Written order",
        source_id="pacer",
        raw_text="The competing expert testimony on software implementation...",
        event_type="summary_judgment",
        story_arc="Context Matters"
    ),
    DemoEvent(
        event_id="evt_lee_soft_003",
        who=["Judge David Lee", "DatabaseTech", "QueryOptimizer"],
        what="Denied summary judgment - software patent",
        when=_date(-260),
        where="D. Delaware",
        why="Algorithm comparison requires factual development",
        how="Written order",
        source_id="pacer",
        raw_text="Comparing the claimed algorithm to the prior art...",
        event_type="summary_judgment",
        story_arc="Context Matters"
    ),
    DemoEvent(
        event_id="evt_lee_soft_004",
        who=["Judge David Lee", "SecuritySoft", "CryptoDefense"],
        what="Granted summary judgment - software patent (rare!)",
        when=_date(-240),
        where="D. Delaware",
        why="Claims clearly abstract under Alice",
        how="Written order",
        source_id="pacer",
        raw_text="The claims are directed to the abstract idea of data organization...",
        event_type="summary_judgment",
        story_arc="Context Matters"
    ),

    # Hardware patent cases - high grant rate (58%)
    DemoEvent(
        event_id="evt_lee_hard_001",
        who=["Judge David Lee", "ChipDesign Inc", "ProcessorCo"],
        what="Granted summary judgment - hardware patent",
        when=_date(-220),
        where="D. Delaware",
        why="Physical structure clearly distinguishes from prior art",
        how="Written order",
        source_id="pacer",
        raw_text="The claimed transistor arrangement is not taught by any prior art...",
        event_type="summary_judgment",
        story_arc="Context Matters"
    ),
    DemoEvent(
        event_id="evt_lee_hard_002",
        who=["Judge David Lee", "CircuitBoard Corp", "PCBMaker"],
        what="Granted summary judgment - hardware patent",
        when=_date(-200),
        where="D. Delaware",
        why="Non-infringement clear from physical comparison",
        how="Written order",
        source_id="pacer",
        raw_text="Physical inspection confirms defendant's board lacks the claimed...",
        event_type="summary_judgment",
        story_arc="Context Matters"
    ),
    DemoEvent(
        event_id="evt_lee_hard_003",
        who=["Judge David Lee", "SensorTech", "IoTDevices"],
        what="Denied summary judgment - hardware patent",
        when=_date(-180),
        where="D. Delaware",
        why="Measurement disputes require trial",
        how="Written order",
        source_id="pacer",
        raw_text="The parties dispute the proper measurement methodology...",
        event_type="summary_judgment",
        story_arc="Context Matters"
    ),
    DemoEvent(
        event_id="evt_lee_hard_004",
        who=["Judge David Lee", "DisplayTech", "ScreenMaker"],
        what="Granted summary judgment - hardware patent",
        when=_date(-160),
        where="D. Delaware",
        why="Prior art reference anticipates all claim elements",
        how="Written order with claim chart",
        source_id="pacer",
        raw_text="Element by element, the Johnson reference anticipates...",
        event_type="summary_judgment",
        story_arc="Context Matters"
    ),
]


# ============================================================
# STORY 5: Source Degradation (Patent Buddy Blog)
# Demonstrates automatic source reliability downgrading
# ============================================================

SOURCE_DEGRADATION_EVENTS = [
    # Patent Buddy Blog makes predictions - starts at 0.65 reliability
    DemoEvent(
        event_id="evt_source_001",
        who=["Patent Buddy Blog"],
        what="Predicts Judge Rodriguez will grant summary judgment",
        when=_date(-160),
        where="Blog post",
        why="Based on historical patterns",
        how="Blog analysis",
        source_id="patent_buddy_blog",
        raw_text="PREDICTION: Based on Judge Rodriguez's 78% grant rate, we expect...",
        event_type="prediction",
        story_arc="Source Degradation"
    ),
    DemoEvent(
        event_id="evt_source_002",
        who=["Judge Maria Rodriguez", "MegaCorp", "SmallTech LLC"],
        what="DENIED summary judgment (contradicts prediction)",
        when=_date(-150),
        where="N.D. Cal",
        why="Genuine disputes exist",
        how="Written order",
        source_id="pacer",
        raw_text="Motion DENIED - the Court finds genuine issues of material fact...",
        event_type="summary_judgment",
        story_arc="Source Degradation"
    ),
    # Blog tries again
    DemoEvent(
        event_id="evt_source_003",
        who=["Patent Buddy Blog"],
        what="Predicts Judge Rodriguez will grant next summary judgment",
        when=_date(-125),
        where="Blog post",
        why="Adjusting for recent data",
        how="Blog analysis",
        source_id="patent_buddy_blog",
        raw_text="Despite last month's surprise denial, we still expect grants...",
        event_type="prediction",
        story_arc="Source Degradation"
    ),
    DemoEvent(
        event_id="evt_source_004",
        who=["Judge Maria Rodriguez", "AppDev Inc", "CodeBase Corp"],
        what="DENIED summary judgment (wrong again)",
        when=_date(-120),
        where="N.D. Cal",
        why="Expert creates fact issues",
        how="Minute order",
        source_id="pacer",
        raw_text="Motion DENIED...",
        event_type="summary_judgment",
        story_arc="Source Degradation"
    ),
    # Third strike
    DemoEvent(
        event_id="evt_source_005",
        who=["Patent Buddy Blog"],
        what="Predicts grant with 'high confidence'",
        when=_date(-95),
        where="Blog post",
        why="Claims Rodriguez is 'due for a grant'",
        how="Blog analysis",
        source_id="patent_buddy_blog",
        raw_text="Rodriguez is overdue! We're highly confident in a grant...",
        event_type="prediction",
        story_arc="Source Degradation"
    ),
    DemoEvent(
        event_id="evt_source_006",
        who=["Judge Maria Rodriguez", "SecureTech", "HackShield Inc"],
        what="DENIED summary judgment (third miss)",
        when=_date(-90),
        where="N.D. Cal",
        why="Both parties have fact issues",
        how="Written order",
        source_id="pacer",
        raw_text="Both motions DENIED...",
        event_type="summary_judgment",
        story_arc="Source Degradation"
    ),
    # System should have downgraded Patent Buddy Blog to ~0.45 by now
]


# Combine all events
DEMO_EVENTS = (
    RODRIGUEZ_EVENTS +
    CHEN_EVENTS +
    MARTINEZ_EVENTS +
    LEE_EVENTS +
    SOURCE_DEGRADATION_EVENTS
)


# Story arc descriptions for documentation
EVENT_STORIES = {
    "The Changing Judge": {
        "judge": "Rodriguez",
        "events": RODRIGUEZ_EVENTS,
        "insight": "Temporal decay is essential. Old patterns (78% grant rate) became "
                  "dangerously stale after Rodriguez became Chief Judge. The system must "
                  "recognize shifts and recalibrate, not blindly trust historical data.",
        "key_metric": "Grant rate dropped from 78% to 42% after appointment change",
    },
    "The Reliable Constant": {
        "judge": "Chen",
        "events": CHEN_EVENTS,
        "insight": "With 156 cases at a consistent 32% rate, Chen's patterns have "
                  "extremely tight confidence intervals. The system can AUTO_PASS "
                  "predictions about this judge because uncertainty is minimal.",
        "key_metric": "Wilson score lower bound = 25.1% (very tight for 156 cases)",
    },
    "The New Judge": {
        "judge": "Martinez",
        "events": MARTINEZ_EVENTS,
        "insight": "Raw rate of 50% (4/8) is misleading. Wilson score shows the TRUE "
                  "rate could be anywhere from 21% to 79%. Proper uncertainty quantification "
                  "prevents overconfident predictions.",
        "key_metric": "Wilson score interval: [21.5%, 78.5%] - massive uncertainty!",
    },
    "Context Matters": {
        "judge": "Lee",
        "events": LEE_EVENTS,
        "insight": "Overall 40% grant rate hides crucial context. For software patents "
                  "it's 25%, for hardware it's 58%. Aggregate statistics can be misleading; "
                  "the system must track sub-patterns.",
        "key_metric": "Software: 25% vs Hardware: 58% - same judge, opposite patterns",
    },
    "Source Degradation": {
        "events": SOURCE_DEGRADATION_EVENTS,
        "insight": "Patent Buddy Blog started at 0.65 reliability but made 3 wrong "
                  "predictions in a row. The system automatically downgraded it to ~0.45. "
                  "Source reliability is learned, not assumed.",
        "key_metric": "Reliability dropped: 0.65 → 0.58 → 0.52 → 0.45 (3 wrong predictions)",
    },
}
