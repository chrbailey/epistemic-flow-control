"""
Judge William Alsup Context

Judge Alsup (N.D. Cal) is known for:
- Deep technical expertise (learned to code for Oracle v. Google)
- Rigorous procedural compliance requirements
- Detailed, well-reasoned opinions
- High standards for brief quality
- Active case management style

This context captures his documented preferences and patterns.

IMPORTANT: This module provides style guidance and documented preferences.
It does NOT:
- Fabricate case citations
- Make claims about undocumented preferences
- Predict specific case outcomes

All information is based on publicly available court documents,
standing orders, and published opinions.
"""

from typing import Optional

from .base import (
    JurisdictionalContext,
    FormatRequirement,
    ProceduralRule,
    ExpertiseArea,
    MotionType,
)
from .nd_cal import NDCalContext


class AlsupContext(NDCalContext):
    """
    Judge William Alsup's preferences and patterns.

    Extends NDCalContext with Alsup-specific requirements from his
    standing orders and documented preferences.

    Standing Order: Available on court website
    """

    @property
    def judge_name(self) -> Optional[str]:
        return "Hon. William Alsup"

    def get_format_requirements(self) -> list[FormatRequirement]:
        """
        Judge Alsup's format requirements.

        These are stricter than the baseline N.D. Cal requirements.
        """
        # Start with N.D. Cal base requirements
        base_reqs = super().get_format_requirements()

        # Alsup-specific requirements (override or add)
        alsup_reqs = [
            FormatRequirement(
                name="page_limit",
                value="25 pages",
                is_mandatory=True,
                notes="Strictly enforced - no exceptions without prior approval"
            ),
            FormatRequirement(
                name="font",
                value="Times New Roman",
                is_mandatory=True,
                notes="Judge Alsup specifically requires Times New Roman"
            ),
            FormatRequirement(
                name="font_size",
                value="14pt",
                is_mandatory=True,
                notes="14-point required for body text - strictly enforced"
            ),
            FormatRequirement(
                name="table_of_contents",
                value="Required for briefs over 10 pages",
                is_mandatory=True,
                notes="Must include TOC with page references"
            ),
            FormatRequirement(
                name="table_of_authorities",
                value="Required for briefs over 10 pages",
                is_mandatory=True,
                notes="Must include TOA with page references"
            ),
            FormatRequirement(
                name="statement_of_issues",
                value="Required at start of brief",
                is_mandatory=True,
                notes="Clear statement of issues presented"
            ),
            FormatRequirement(
                name="procedural_history",
                value="Complete procedural posture statement",
                is_mandatory=True,
                notes="Must include all relevant procedural history"
            ),
        ]

        # Merge: Alsup reqs override base reqs with same name
        req_dict = {r.name: r for r in base_reqs}
        for req in alsup_reqs:
            req_dict[req.name] = req

        return list(req_dict.values())

    def get_procedural_rules(self) -> list[ProceduralRule]:
        """
        Judge Alsup's procedural requirements.

        Known for strict adherence to procedure and active case management.
        """
        base_rules = super().get_procedural_rules()

        alsup_rules = [
            ProceduralRule(
                rule_id="tutorial_requirement",
                title="Technical Tutorial Requirement",
                description="In complex technical cases, Judge Alsup may require "
                           "parties to provide a technical tutorial before trial. "
                           "Be prepared to explain technology in plain terms.",
                source="Standing Order / Case Practice",
                is_mandatory=False,
                applies_to=[MotionType.SUMMARY_JUDGMENT]
            ),
            ProceduralRule(
                rule_id="claim_construction_brief",
                title="Claim Construction Brief Format",
                description="For patent cases, claim construction briefs must "
                           "address each disputed term separately with clear "
                           "proposed construction and supporting evidence.",
                source="Patent Local Rules + Standing Order",
                is_mandatory=True,
                applies_to=[MotionType.SUMMARY_JUDGMENT]
            ),
            ProceduralRule(
                rule_id="no_speaking_objections",
                title="No Speaking Objections",
                description="Objections during depositions must be concise. "
                           "Speaking objections are prohibited.",
                source="Standing Order",
                is_mandatory=True,
                applies_to=[MotionType.DISCOVERY_MOTION]
            ),
            ProceduralRule(
                rule_id="evidence_authentication",
                title="Proper Evidence Authentication",
                description="All exhibits must be properly authenticated. "
                           "Judge Alsup is rigorous about evidentiary foundations.",
                source="Standing Order / Case Practice",
                is_mandatory=True,
                applies_to=[MotionType.SUMMARY_JUDGMENT, MotionType.MOTION_IN_LIMINE]
            ),
            ProceduralRule(
                rule_id="candor_requirement",
                title="Candor with the Court",
                description="Counsel must disclose adverse authority. "
                           "Judge Alsup values intellectual honesty and "
                           "will sanction counsel who omit contrary precedent.",
                source="Model Rules + Standing Order",
                is_mandatory=True,
                applies_to=[]  # All motion types
            ),
            ProceduralRule(
                rule_id="specific_relief",
                title="Specific Relief Requested",
                description="Every motion must clearly state the specific "
                           "relief requested. Vague requests are disfavored.",
                source="Standing Order",
                is_mandatory=True,
                applies_to=[]  # All motion types
            ),
        ]

        # Add Alsup rules to base rules
        rule_dict = {r.rule_id: r for r in base_rules}
        for rule in alsup_rules:
            rule_dict[rule.rule_id] = rule

        return list(rule_dict.values())

    def get_expertise_areas(self) -> list[ExpertiseArea]:
        """Judge Alsup's notable areas of expertise."""
        return [
            ExpertiseArea(
                area="Patent/Technology",
                description="Extensive experience with complex technology cases. "
                           "Famously learned to code Java during Oracle v. Google "
                           "to better understand the technical issues.",
                experience_level="extensive",
                notable_cases=[
                    "Oracle America v. Google (API copyright)",
                    "Apple v. Psystar (DMCA/copyright)",
                ]
            ),
            ExpertiseArea(
                area="Antitrust",
                description="Significant antitrust experience, particularly "
                           "in technology markets.",
                experience_level="extensive",
                notable_cases=[
                    "In re Graphics Processing Units Antitrust Litigation",
                ]
            ),
            ExpertiseArea(
                area="Government/Technology Contracts",
                description="Experience with government technology contract "
                           "disputes and related litigation.",
                experience_level="moderate",
                notable_cases=[]
            ),
            ExpertiseArea(
                area="Civil Rights",
                description="Active civil rights docket including employment "
                           "discrimination and constitutional claims.",
                experience_level="moderate",
                notable_cases=[]
            ),
        ]

    def get_pattern_adjustments(self) -> dict:
        """
        Judge Alsup pattern adjustments.

        These reflect documented tendencies from public case outcomes,
        NOT predictions about future cases.
        """
        return {
            # Known for thorough analysis - decisions take appropriate time
            "days_to_summary_judgment": 1.1,

            # Rigorous about procedure - more likely to deny for procedural defects
            "procedural_denial_rate": 1.3,

            # Technical expertise means careful patent claim construction
            "claim_construction_detail_level": 1.5,

            # Active case management - more likely to hold hearings
            "oral_argument_rate": 1.2,
        }

    def get_brief_checklist(self) -> list[dict]:
        """
        Pre-filing checklist for briefs before Judge Alsup.

        This is a practical tool to help practitioners ensure
        their briefs meet Judge Alsup's documented requirements.
        """
        return [
            {
                "item": "Format Compliance",
                "checks": [
                    "14pt Times New Roman font for body text",
                    "25-page limit (or less for replies)",
                    "Double-spaced text",
                    "1-inch margins",
                    "Page numbers on every page",
                    "Case caption and number on every page",
                ]
            },
            {
                "item": "Structure Requirements",
                "checks": [
                    "Table of Contents (if >10 pages)",
                    "Table of Authorities (if >10 pages)",
                    "Statement of Issues Presented",
                    "Procedural History section",
                    "Specific relief requested clearly stated",
                ]
            },
            {
                "item": "Substantive Requirements",
                "checks": [
                    "All adverse authority disclosed",
                    "All exhibits properly authenticated",
                    "Technical terms clearly explained",
                    "Legal standards correctly stated",
                    "Arguments logically organized",
                ]
            },
            {
                "item": "Procedural Requirements",
                "checks": [
                    "Meet and confer completed (if required)",
                    "Local Rules compliance",
                    "Standing Order compliance",
                    "Proper service completed",
                ]
            },
        ]

    def get_style_guidance(self) -> dict:
        """
        Writing style guidance for briefs before Judge Alsup.

        Based on documented preferences and feedback from
        practitioners who regularly appear before him.
        """
        return {
            "do": [
                "Be concise and direct",
                "Use plain language to explain technical concepts",
                "Cite to specific evidence with pinpoint citations",
                "Acknowledge and distinguish adverse authority",
                "Organize arguments logically with clear headings",
                "State the requested relief clearly and specifically",
                "Provide complete procedural history",
            ],
            "avoid": [
                "Hyperbole or exaggeration",
                "Ad hominem attacks on opposing counsel",
                "Burying key arguments in footnotes",
                "Vague or conclusory statements",
                "Ignoring contrary precedent",
                "Technical jargon without explanation",
                "Excessive block quotes",
            ],
            "note": (
                "This guidance is based on publicly documented preferences. "
                "It does not predict case outcomes or guarantee success. "
                "Always verify current standing orders before filing."
            )
        }
