"""
Northern District of California Context

The N.D. Cal is one of the busiest federal courts, particularly known for:
- Technology and intellectual property litigation
- Securities class actions
- Privacy and data breach cases
- Government technology contracts

This context provides the court's local rules and typical patterns.
"""

from .base import (
    JurisdictionalContext,
    FormatRequirement,
    ProceduralRule,
    ExpertiseArea,
    MotionType,
)


class NDCalContext(JurisdictionalContext):
    """
    Northern District of California jurisdictional context.

    Local rules: https://www.cand.uscourts.gov/local-rules
    """

    @property
    def court_name(self) -> str:
        return "United States District Court for the Northern District of California"

    @property
    def court_code(self) -> str:
        return "nd_cal"

    def get_format_requirements(self) -> list[FormatRequirement]:
        """N.D. Cal Civil Local Rule 3-4 requirements."""
        return [
            FormatRequirement(
                name="page_limit",
                value="25 pages",
                is_mandatory=True,
                notes="For motions and oppositions (Civil L.R. 3-4(a))"
            ),
            FormatRequirement(
                name="reply_page_limit",
                value="15 pages",
                is_mandatory=True,
                notes="For reply briefs (Civil L.R. 3-4(a))"
            ),
            FormatRequirement(
                name="font",
                value="Times New Roman or similar proportional font",
                is_mandatory=True,
                notes="Civil L.R. 3-4(c)(2)"
            ),
            FormatRequirement(
                name="font_size",
                value="14pt",
                is_mandatory=True,
                notes="Body text (Civil L.R. 3-4(c)(2))"
            ),
            FormatRequirement(
                name="footnote_font_size",
                value="12pt",
                is_mandatory=True,
                notes="Footnotes minimum (Civil L.R. 3-4(c)(2))"
            ),
            FormatRequirement(
                name="line_spacing",
                value="Double-spaced",
                is_mandatory=True,
                notes="Text must be double-spaced (Civil L.R. 3-4(c)(1))"
            ),
            FormatRequirement(
                name="margins",
                value="1 inch minimum",
                is_mandatory=True,
                notes="All margins (Civil L.R. 3-4(c)(3))"
            ),
            FormatRequirement(
                name="footer",
                value="Case number required",
                is_mandatory=True,
                notes="Must appear on each page"
            ),
        ]

    def get_procedural_rules(self) -> list[ProceduralRule]:
        """Key N.D. Cal procedural rules."""
        return [
            ProceduralRule(
                rule_id="meet_and_confer",
                title="Meet and Confer Requirement",
                description="Before filing any motion, counsel must meet and confer "
                           "in good faith to attempt to resolve the dispute.",
                source="Civil L.R. 37-1, 7-1",
                is_mandatory=True,
                applies_to=[
                    MotionType.DISCOVERY_MOTION,
                    MotionType.MOTION_IN_LIMINE,
                    MotionType.EXTENSION,
                ]
            ),
            ProceduralRule(
                rule_id="joint_case_management",
                title="Joint Case Management Statement",
                description="Parties must file a joint case management statement "
                           "at least 14 days before the initial case management conference.",
                source="Civil L.R. 16-9",
                is_mandatory=True,
                applies_to=[MotionType.ADMINISTRATIVE]
            ),
            ProceduralRule(
                rule_id="adi_disclosure",
                title="ADR Disclosure",
                description="Initial case management statement must include ADR "
                           "certification and discuss ADR options.",
                source="ADR L.R. 3-5",
                is_mandatory=True,
                applies_to=[MotionType.ADMINISTRATIVE]
            ),
            ProceduralRule(
                rule_id="patent_local_rules",
                title="Patent Local Rules",
                description="Patent cases follow special Patent Local Rules including "
                           "mandatory infringement/invalidity contentions schedule.",
                source="Patent L.R. 1-1 through 6-1",
                is_mandatory=True,
                applies_to=[MotionType.SUMMARY_JUDGMENT, MotionType.MOTION_IN_LIMINE]
            ),
            ProceduralRule(
                rule_id="summary_judgment_statement",
                title="Separate Statement of Undisputed Facts",
                description="Motion for summary judgment must include a separate statement "
                           "setting forth material facts as to which moving party contends "
                           "there is no genuine dispute.",
                source="Civil L.R. 56-2",
                is_mandatory=True,
                applies_to=[MotionType.SUMMARY_JUDGMENT]
            ),
            ProceduralRule(
                rule_id="e_filing",
                title="Electronic Filing Required",
                description="All documents must be filed electronically via CM/ECF.",
                source="Civil L.R. 5-1",
                is_mandatory=True,
                applies_to=[]  # All motion types
            ),
            ProceduralRule(
                rule_id="administrative_motion",
                title="Administrative Motions",
                description="Requests for extensions, stipulations, and scheduling matters "
                           "must be filed as administrative motions on shortened time.",
                source="Civil L.R. 7-11",
                is_mandatory=True,
                applies_to=[MotionType.EXTENSION, MotionType.ADMINISTRATIVE]
            ),
        ]

    def get_expertise_areas(self) -> list[ExpertiseArea]:
        """N.D. Cal's notable areas of practice."""
        return [
            ExpertiseArea(
                area="Technology/IP",
                description="Major venue for patent litigation, especially "
                           "software and technology patents",
                experience_level="extensive",
                notable_cases=["Oracle v. Google", "Apple v. Samsung"]
            ),
            ExpertiseArea(
                area="Securities",
                description="Significant securities class action docket, "
                           "particularly tech company IPO cases",
                experience_level="extensive",
                notable_cases=["In re Facebook IPO Securities Litigation"]
            ),
            ExpertiseArea(
                area="Privacy/Data",
                description="Leading jurisdiction for privacy and data breach "
                           "class actions due to Silicon Valley presence",
                experience_level="extensive",
                notable_cases=["In re Yahoo! Customer Data Security Breach Litigation"]
            ),
            ExpertiseArea(
                area="Antitrust",
                description="Tech-focused antitrust cases",
                experience_level="moderate",
                notable_cases=["Epic Games v. Apple"]
            ),
        ]

    def get_pattern_adjustments(self) -> dict:
        """N.D. Cal tends to have specific patterns."""
        return {
            # N.D. Cal judges are often more tech-savvy
            "patent_claim_construction_reversal_rate": 0.9,
            # Complex litigation often means longer timelines
            "days_to_summary_judgment": 1.2,
        }
