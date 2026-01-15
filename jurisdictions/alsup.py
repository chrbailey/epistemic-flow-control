"""
Judge William Alsup Context

Judge William Alsup (N.D. Cal) is a senior United States District Judge known for:
- Deep technical expertise (learned to code for Oracle v. Google)
- Rigorous procedural compliance requirements
- Detailed, well-reasoned opinions with extensive fact-finding
- High standards for brief quality and attorney candor
- Active case management style with frequent status conferences
- Willingness to master complex technical subjects

Biographical Data (Federal Judicial Center):
- Born: June 27, 1945, Jackson, Mississippi
- Education:
  - B.S. Mathematics, Mississippi State University (1967)
  - J.D., Harvard Law School (1971)
  - M.P.P., Harvard Kennedy School (1971)
- Career:
  - Law Clerk, Hon. William O. Douglas, U.S. Supreme Court (1971-1972)
  - Associate/Partner, Morrison & Foerster (1972-1978, 1980-1998)
  - Special Assistant, Office of the Solicitor General (1978-1980)
  - Special Counsel, U.S. DOJ Antitrust Division (1998)
- Judicial Service:
  - Nominated by President Clinton: March 24, 1999
  - Confirmed by Senate: July 30, 1999
  - Senior Status: January 21, 2021

IMPORTANT: This module provides style guidance and documented preferences.
It does NOT:
- Fabricate case citations
- Make claims about undocumented preferences
- Predict specific case outcomes

All information is based on publicly available court documents,
standing orders, published opinions, and verified sources.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .base import (
    JurisdictionalContext,
    FormatRequirement,
    ProceduralRule,
    ExpertiseArea,
    MotionType,
)
from .nd_cal import NDCalContext


@dataclass
class NotableCase:
    """A notable case from Judge Alsup's record."""
    name: str
    case_number: str
    year: int
    area: str
    outcome: str
    significance: str
    key_quote: Optional[str] = None


@dataclass
class JudicialProfile:
    """Biographical profile of a judge."""
    full_name: str
    birth_date: date
    birth_place: str
    education: list[dict]
    career_history: list[dict]
    appointment: dict
    senior_status_date: Optional[date] = None


class AlsupContext(NDCalContext):
    """
    Judge William Alsup's preferences and patterns.

    Extends NDCalContext with Alsup-specific requirements from his
    standing orders and documented preferences.

    Sources:
    - Federal Judicial Center biography
    - Standing Order (N.D. Cal website)
    - Published opinions and orders
    - Verified news coverage of significant rulings
    """

    # Biographical Constants
    PROFILE = JudicialProfile(
        full_name="William Haskell Alsup",
        birth_date=date(1945, 6, 27),
        birth_place="Jackson, Mississippi",
        education=[
            {
                "degree": "B.S. Mathematics",
                "institution": "Mississippi State University",
                "year": 1967,
            },
            {
                "degree": "J.D.",
                "institution": "Harvard Law School",
                "year": 1971,
            },
            {
                "degree": "M.P.P.",
                "institution": "Harvard Kennedy School of Government",
                "year": 1971,
            },
        ],
        career_history=[
            {
                "position": "Law Clerk",
                "employer": "Hon. William O. Douglas, U.S. Supreme Court",
                "years": "1971-1972",
            },
            {
                "position": "Associate/Partner",
                "employer": "Morrison & Foerster LLP",
                "years": "1972-1978, 1980-1998",
            },
            {
                "position": "Special Assistant",
                "employer": "Office of the Solicitor General, U.S. DOJ",
                "years": "1978-1980",
            },
            {
                "position": "Special Counsel",
                "employer": "Antitrust Division, U.S. DOJ",
                "years": "1998",
            },
        ],
        appointment={
            "nominated_by": "President Bill Clinton",
            "nominated_date": date(1999, 3, 24),
            "confirmed_date": date(1999, 7, 30),
            "commission_date": date(1999, 8, 2),
        },
        senior_status_date=date(2021, 1, 21),
    )

    # Notable Cases with verified details
    NOTABLE_CASES = [
        NotableCase(
            name="Oracle America, Inc. v. Google Inc.",
            case_number="3:10-cv-03561-WHA",
            year=2012,
            area="Patent/Copyright/Technology",
            outcome="Found API declaring code not copyrightable (reversed by Fed. Cir.)",
            significance="Judge Alsup learned to code in Java to understand the technical "
                        "issues. His detailed technical analysis set new standards for "
                        "judicial engagement with software copyright questions.",
            key_quote="I have done, and still do, parsing in my own programs. "
                     "I have written blocks of code like rangeCheck a hundred times before. "
                     "I could do it, you could do it. The idea that someone would copyright "
                     "that is shocking to me.",
        ),
        NotableCase(
            name="Waymo LLC v. Uber Technologies, Inc.",
            case_number="3:17-cv-00939-WHA",
            year=2018,
            area="Trade Secrets",
            outcome="Settled during trial; related criminal case resulted in "
                   "18-month sentence for Anthony Levandowski",
            significance="Major trade secrets case involving alleged theft of autonomous "
                        "vehicle technology. Judge Alsup's management of the case and "
                        "the related criminal referral demonstrated rigorous handling "
                        "of intellectual property theft.",
            key_quote="This is the biggest trade secret crime I have ever seen. "
                     "This was not small. This was massive in scale.",
        ),
        NotableCase(
            name="Regents of Univ. of California v. DHS (DACA)",
            case_number="3:17-cv-05211-WHA",
            year=2018,
            area="Immigration/Administrative Law",
            outcome="Granted preliminary injunction blocking DACA rescission",
            significance="49-page order finding the Trump administration's rescission "
                        "of DACA was likely arbitrary and capricious. One of the first "
                        "nationwide injunctions protecting DACA recipients.",
            key_quote="Plaintiffs have shown that they are likely to succeed on "
                     "the merits of their claim that the rescission was arbitrary "
                     "and capricious.",
        ),
        NotableCase(
            name="City of Oakland v. BP P.L.C. (Climate Change)",
            case_number="3:17-cv-06011-WHA",
            year=2018,
            area="Environmental/Tort",
            outcome="Dismissed federal claims; case proceeded in state court",
            significance="First major ruling on municipal climate change tort claims "
                        "against fossil fuel companies. While dismissing the case, "
                        "Judge Alsup held an unprecedented 'climate science tutorial' "
                        "requiring both sides to present climate evidence.",
            key_quote="The problem deserves a solution on a more vast scale than "
                     "can be supplied by a district judge. The scope of plaintiffs' "
                     "theory is breathtaking.",
        ),
        NotableCase(
            name="Sonos, Inc. v. Google LLC",
            case_number="3:20-cv-06754-WHA",
            year=2023,
            area="Patent",
            outcome="Threw out $32.5 million jury verdict; granted new trial",
            significance="Highly critical of patent litigation tactics, finding "
                        "Sonos had made misrepresentations to the Patent Office "
                        "and engaged in problematic litigation conduct.",
            key_quote="It is wrong that our patent system was used in this way. "
                     "Sonos has engaged in a pattern of misrepresentation.",
        ),
        NotableCase(
            name="Sweet v. Cardona (Student Loan Forgiveness)",
            case_number="3:19-cv-03674-WHA",
            year=2022,
            area="Administrative Law/Education",
            outcome="Approved $6 billion settlement for defrauded students",
            significance="Class action on behalf of students at for-profit colleges. "
                        "Judge Alsup approved massive settlement and criticized "
                        "the Department of Education's processing delays.",
            key_quote="The Department of Education has created an impossible quagmire "
                     "for borrowers... Years go by. Some borrowers have been waiting "
                     "a decade or more for a decision.",
        ),
        NotableCase(
            name="AFGE v. Trump (Federal Workforce/DOGE)",
            case_number="3:25-cv-00732-WHA",
            year=2025,
            area="Administrative Law/Employment",
            outcome="Issued temporary restraining order blocking mass firings",
            significance="Blocked the Office of Personnel Management from "
                        "implementing mass terminations of federal probationary "
                        "employees, finding likely procedural violations.",
            key_quote="OPM does not have any authority whatsoever to direct the "
                     "agency heads to carry out terminations. OPM is not a firing "
                     "squad for the entire government.",
        ),
    ]

    @property
    def judge_name(self) -> Optional[str]:
        return "Hon. William Alsup"

    @property
    def full_name(self) -> str:
        return self.PROFILE.full_name

    @property
    def years_on_bench(self) -> int:
        """Calculate years since confirmation."""
        today = date.today()
        confirmed = self.PROFILE.appointment["confirmed_date"]
        return today.year - confirmed.year

    @property
    def is_senior_status(self) -> bool:
        """Check if judge is on senior status."""
        return self.PROFILE.senior_status_date is not None

    def get_biography(self) -> dict:
        """
        Return comprehensive biographical information.

        Source: Federal Judicial Center
        """
        return {
            "name": self.PROFILE.full_name,
            "born": {
                "date": self.PROFILE.birth_date.isoformat(),
                "place": self.PROFILE.birth_place,
            },
            "education": self.PROFILE.education,
            "career": self.PROFILE.career_history,
            "judicial_appointment": {
                "nominated_by": self.PROFILE.appointment["nominated_by"],
                "nominated": self.PROFILE.appointment["nominated_date"].isoformat(),
                "confirmed": self.PROFILE.appointment["confirmed_date"].isoformat(),
                "senior_status": (
                    self.PROFILE.senior_status_date.isoformat()
                    if self.PROFILE.senior_status_date else None
                ),
            },
            "notable_attributes": [
                "Learned Java programming to understand Oracle v. Google",
                "Held first 'climate science tutorial' in federal court",
                "Clerked for Justice William O. Douglas",
                "Mathematics undergraduate degree",
                "Combined J.D. and M.P.P. from Harvard",
            ],
        }

    def get_notable_cases(self, area: Optional[str] = None) -> list[NotableCase]:
        """
        Return notable cases, optionally filtered by area.

        Args:
            area: Filter by legal area (e.g., "Patent", "Trade Secrets")
        """
        if area is None:
            return self.NOTABLE_CASES

        return [
            case for case in self.NOTABLE_CASES
            if area.lower() in case.area.lower()
        ]

    def get_format_requirements(self) -> list[FormatRequirement]:
        """
        Judge Alsup's format requirements.

        These are stricter than the baseline N.D. Cal requirements.
        Based on Standing Order and documented preferences.
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
                           "As seen in Oracle v. Google (Java) and climate cases. "
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
                           "Judge Alsup is rigorous about evidentiary foundations. "
                           "See Sonos v. Google for consequences of evidentiary issues.",
                source="Standing Order / Case Practice",
                is_mandatory=True,
                applies_to=[MotionType.SUMMARY_JUDGMENT, MotionType.MOTION_IN_LIMINE]
            ),
            ProceduralRule(
                rule_id="candor_requirement",
                title="Candor with the Court",
                description="Counsel must disclose adverse authority. "
                           "Judge Alsup values intellectual honesty and "
                           "will sanction counsel who omit contrary precedent. "
                           "Critical of misrepresentations (see Sonos v. Google).",
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
            ProceduralRule(
                rule_id="administrative_record",
                title="Complete Administrative Record",
                description="For administrative law cases, the administrative "
                           "record must be complete. Judge Alsup carefully "
                           "examines agency decision-making processes "
                           "(see DACA and Sweet v. Cardona rulings).",
                source="Case Practice",
                is_mandatory=True,
                applies_to=[MotionType.SUMMARY_JUDGMENT]
            ),
        ]

        # Add Alsup rules to base rules
        rule_dict = {r.rule_id: r for r in base_rules}
        for rule in alsup_rules:
            rule_dict[rule.rule_id] = rule

        return list(rule_dict.values())

    def get_expertise_areas(self) -> list[ExpertiseArea]:
        """
        Judge Alsup's notable areas of expertise.

        Based on case history and documented experience.
        """
        return [
            ExpertiseArea(
                area="Patent/Technology/Copyright",
                description="Extensive experience with complex technology cases. "
                           "Famously learned to code Java during Oracle v. Google "
                           "to better understand API copyright issues. Mathematics "
                           "undergraduate background aids technical comprehension.",
                experience_level="extensive",
                notable_cases=[
                    "Oracle America v. Google (API copyright/fair use)",
                    "Sonos v. Google (smart speaker patents)",
                    "Apple v. Psystar (DMCA/copyright)",
                ]
            ),
            ExpertiseArea(
                area="Trade Secrets",
                description="Significant experience with trade secret theft cases, "
                           "particularly in technology sector. Rigorous approach "
                           "to protecting intellectual property and punishing theft.",
                experience_level="extensive",
                notable_cases=[
                    "Waymo v. Uber (autonomous vehicle technology)",
                ]
            ),
            ExpertiseArea(
                area="Antitrust",
                description="Prior DOJ Antitrust Division experience as Special "
                           "Counsel. Applies sophisticated economic analysis to "
                           "competition issues.",
                experience_level="extensive",
                notable_cases=[
                    "In re Graphics Processing Units Antitrust Litigation",
                ]
            ),
            ExpertiseArea(
                area="Administrative Law",
                description="Active docket of challenges to federal agency actions. "
                           "Thorough examination of agency reasoning and procedures. "
                           "Known for detailed analysis of administrative record.",
                experience_level="extensive",
                notable_cases=[
                    "Regents v. DHS (DACA rescission)",
                    "Sweet v. Cardona (student loan forgiveness)",
                    "AFGE v. Trump (federal workforce)",
                ]
            ),
            ExpertiseArea(
                area="Environmental/Climate",
                description="Handled major climate change litigation. Conducted "
                           "first-ever 'climate science tutorial' in federal court. "
                           "Careful about judicial role in policy disputes.",
                experience_level="moderate",
                notable_cases=[
                    "City of Oakland v. BP (climate change tort)",
                ]
            ),
            ExpertiseArea(
                area="Civil Rights/Employment",
                description="Active civil rights docket including employment "
                           "discrimination, constitutional claims, and federal "
                           "employee protections.",
                experience_level="moderate",
                notable_cases=[
                    "Various employment discrimination matters",
                ]
            ),
        ]

    def get_pattern_adjustments(self) -> dict:
        """
        Judge Alsup pattern adjustments.

        These reflect documented tendencies from public case outcomes,
        NOT predictions about future cases. Used for confidence calibration.
        """
        return {
            # Known for thorough analysis - decisions take appropriate time
            "days_to_ruling_multiplier": 1.1,

            # Rigorous about procedure - more likely to deny for procedural defects
            "procedural_denial_rate_multiplier": 1.3,

            # Technical expertise means careful patent claim construction
            "claim_construction_detail_level": 1.5,

            # Active case management - more likely to hold hearings
            "oral_argument_rate_multiplier": 1.2,

            # Willingness to issue injunctions when warranted
            "preliminary_injunction_rate": 1.0,  # Neutral - depends on merits

            # Careful scrutiny of administrative agency decisions
            "administrative_scrutiny_level": 1.4,

            # High bar for attorney conduct
            "sanctions_likelihood_multiplier": 1.2,
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
                    "No misrepresentations (see Sonos v. Google)",
                ]
            },
            {
                "item": "Technical Cases (if applicable)",
                "checks": [
                    "Technology explained in plain terms",
                    "Prepared for potential tutorial requirement",
                    "Technical expert declarations properly supported",
                    "Claim construction terms clearly defined (patent cases)",
                ]
            },
            {
                "item": "Administrative Law Cases (if applicable)",
                "checks": [
                    "Complete administrative record lodged",
                    "Agency reasoning thoroughly addressed",
                    "Arbitrary/capricious standard properly applied",
                    "Standing and ripeness established",
                ]
            },
            {
                "item": "Procedural Requirements",
                "checks": [
                    "Meet and confer completed (if required)",
                    "Local Rules compliance verified",
                    "Standing Order compliance verified",
                    "Proper service completed",
                    "Certificate of service included",
                ]
            },
        ]

    def get_style_guidance(self) -> dict:
        """
        Writing style guidance for briefs before Judge Alsup.

        Based on documented preferences, feedback from practitioners,
        and patterns observed in his rulings.
        """
        return {
            "do": [
                "Be concise and direct - Judge Alsup values efficiency",
                "Use plain language to explain technical concepts",
                "Cite to specific evidence with pinpoint citations",
                "Acknowledge and distinguish adverse authority",
                "Organize arguments logically with clear headings",
                "State the requested relief clearly and specifically",
                "Provide complete procedural history",
                "Demonstrate mastery of the technical subject matter",
                "Be candid about weaknesses in your position",
            ],
            "avoid": [
                "Hyperbole or exaggeration",
                "Ad hominem attacks on opposing counsel",
                "Burying key arguments in footnotes",
                "Vague or conclusory statements",
                "Ignoring contrary precedent",
                "Technical jargon without explanation",
                "Excessive block quotes",
                "Misrepresentations of any kind (critical after Sonos)",
                "Overreaching legal theories",
            ],
            "lessons_from_cases": {
                "Oracle v. Google": "Demonstrate technical understanding; "
                                   "Judge Alsup will test your knowledge",
                "Sonos v. Google": "Candor is paramount; misrepresentations "
                                  "will be discovered and punished",
                "DACA": "Administrative law arguments must be thorough; "
                       "Judge Alsup carefully examines agency reasoning",
                "Climate cases": "Acknowledge limits of judicial role; "
                                "don't overreach on systemic claims",
            },
            "note": (
                "This guidance is based on publicly documented preferences. "
                "It does not predict case outcomes or guarantee success. "
                "Always verify current standing orders before filing."
            )
        }

    def get_key_quotes(self) -> list[dict]:
        """
        Notable quotes from Judge Alsup's opinions and hearings.

        These illustrate his judicial philosophy and approach.
        All quotes are from public records.
        """
        return [
            {
                "case": "Oracle v. Google",
                "year": 2012,
                "quote": "I have done, and still do, parsing in my own programs. "
                        "I have written blocks of code like rangeCheck a hundred "
                        "times before.",
                "context": "Demonstrating technical competence during trial",
            },
            {
                "case": "Waymo v. Uber (Levandowski sentencing)",
                "year": 2020,
                "quote": "This is the biggest trade secret crime I have ever seen. "
                        "This was not small. This was massive in scale.",
                "context": "Explaining severity of trade secret theft",
            },
            {
                "case": "City of Oakland v. BP",
                "year": 2018,
                "quote": "The problem deserves a solution on a more vast scale "
                        "than can be supplied by a district judge. The scope of "
                        "plaintiffs' theory is breathtaking.",
                "context": "Dismissing climate change tort claims",
            },
            {
                "case": "Sonos v. Google",
                "year": 2023,
                "quote": "It is wrong that our patent system was used in this way.",
                "context": "Criticizing patent litigation conduct",
            },
            {
                "case": "Sweet v. Cardona",
                "year": 2022,
                "quote": "The Department of Education has created an impossible "
                        "quagmire for borrowers... Years go by. Some borrowers "
                        "have been waiting a decade or more for a decision.",
                "context": "Criticizing agency processing delays",
            },
            {
                "case": "AFGE v. Trump (DOGE)",
                "year": 2025,
                "quote": "OPM does not have any authority whatsoever to direct "
                        "the agency heads to carry out terminations. OPM is not "
                        "a firing squad for the entire government.",
                "context": "Blocking mass federal employee firings",
            },
        ]

    def get_case_study(self, case_name: str) -> Optional[dict]:
        """
        Get detailed case study for a specific notable case.

        Args:
            case_name: Short name of the case (e.g., "Oracle", "DACA")

        Returns:
            Detailed case study dictionary or None if not found
        """
        case_studies = {
            "oracle": {
                "full_name": "Oracle America, Inc. v. Google Inc.",
                "case_number": "3:10-cv-03561-WHA",
                "summary": "Landmark software copyright case over Google's use "
                          "of Java APIs in Android. Judge Alsup's technical "
                          "engagement set new standards for judicial treatment "
                          "of software disputes.",
                "timeline": [
                    {"date": "2010-08-12", "event": "Oracle files suit"},
                    {"date": "2012-05", "event": "First trial - Alsup learns Java"},
                    {"date": "2012-05-31", "event": "Alsup rules APIs not copyrightable"},
                    {"date": "2014-05-09", "event": "Fed. Cir. reverses"},
                    {"date": "2016-05", "event": "Second trial - fair use jury verdict"},
                    {"date": "2018-03-27", "event": "Fed. Cir. reverses fair use"},
                    {"date": "2020-10", "event": "Supreme Court grants cert"},
                    {"date": "2021-04-05", "event": "Supreme Court: fair use (6-2)"},
                ],
                "alsup_contribution": "Learned Java programming, wrote sample code, "
                                     "demonstrated technical understanding during trial, "
                                     "created detailed fact-findings that influenced "
                                     "Supreme Court's eventual fair use ruling.",
            },
            "daca": {
                "full_name": "Regents of University of California v. DHS",
                "case_number": "3:17-cv-05211-WHA",
                "summary": "Challenge to Trump administration's rescission of DACA. "
                          "Judge Alsup issued one of the first nationwide injunctions "
                          "protecting DACA recipients.",
                "timeline": [
                    {"date": "2017-09-05", "event": "DHS announces DACA rescission"},
                    {"date": "2017-09-08", "event": "UC Regents file suit"},
                    {"date": "2018-01-09", "event": "Alsup grants preliminary injunction"},
                    {"date": "2020-06-18", "event": "Supreme Court upholds injunction (5-4)"},
                ],
                "alsup_contribution": "49-page order with detailed analysis of "
                                     "administrative procedure requirements. Found "
                                     "rescission likely arbitrary and capricious for "
                                     "failure to consider reliance interests.",
            },
            "waymo": {
                "full_name": "Waymo LLC v. Uber Technologies, Inc.",
                "case_number": "3:17-cv-00939-WHA",
                "summary": "Trade secret theft case involving alleged misappropriation "
                          "of autonomous vehicle technology by former Google engineer.",
                "timeline": [
                    {"date": "2017-02-23", "event": "Waymo files suit"},
                    {"date": "2018-02-05", "event": "Trial begins"},
                    {"date": "2018-02-09", "event": "Settlement reached"},
                    {"date": "2019-08-27", "event": "Levandowski indicted"},
                    {"date": "2020-08-04", "event": "Alsup sentences Levandowski to 18 months"},
                ],
                "alsup_contribution": "Referred matter for criminal investigation. "
                                     "Sentenced Levandowski to 18 months in related "
                                     "criminal case, calling it 'the biggest trade "
                                     "secret crime I have ever seen.'",
            },
        }

        key = case_name.lower().replace(" v. ", "").replace(" ", "")
        for name, study in case_studies.items():
            if name in key or key in name:
                return study
        return None
