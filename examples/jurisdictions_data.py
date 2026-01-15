"""
Jurisdictional Context Examples

Example data for different courts and judges,
demonstrating jurisdictional variations.
"""

# Court comparison data
COURT_COMPARISON = {
    "nd_cal": {
        "name": "Northern District of California",
        "code": "nd_cal",
        "notable_for": ["Technology litigation", "Patent cases", "Securities class actions"],
        "judges": 14,
        "typical_case_duration_months": 18,
        "summary_judgment_grant_rate": 0.42,
        "local_rules_url": "https://www.cand.uscourts.gov/local-rules",
    },
    "ed_tex": {
        "name": "Eastern District of Texas",
        "code": "ed_tex",
        "notable_for": ["Patent trolls", "High volume patent cases", "Plaintiff-friendly"],
        "judges": 8,
        "typical_case_duration_months": 24,
        "summary_judgment_grant_rate": 0.35,
        "local_rules_url": "https://www.txed.uscourts.gov/local-rules",
    },
    "d_del": {
        "name": "District of Delaware",
        "code": "d_del",
        "notable_for": ["Corporate litigation", "Patent cases", "Chancery expertise"],
        "judges": 6,
        "typical_case_duration_months": 20,
        "summary_judgment_grant_rate": 0.38,
        "local_rules_url": "https://www.ded.uscourts.gov/local-rules",
    },
    "sdny": {
        "name": "Southern District of New York",
        "code": "sdny",
        "notable_for": ["Securities", "White collar crime", "International disputes"],
        "judges": 28,
        "typical_case_duration_months": 16,
        "summary_judgment_grant_rate": 0.45,
        "local_rules_url": "https://www.nysd.uscourts.gov/local-rules",
    },
}


# Judge profiles
JUDGE_PROFILES = {
    "alsup": {
        "name": "Hon. William Alsup",
        "court": "nd_cal",
        "appointed": 1999,
        "appointed_by": "Clinton",
        "expertise": ["Technology", "Patent", "Antitrust"],
        "notable_cases": [
            "Oracle America v. Google (API copyright)",
            "Apple v. Psystar (DMCA)",
        ],
        "characteristics": [
            "Technical depth - learned to code for Oracle v. Google",
            "Rigorous procedural compliance",
            "Detailed, well-reasoned opinions",
            "Active case management",
        ],
        "format_strictness": "high",
        "oral_argument_frequency": "above_average",
    },
    "gilstrap": {
        "name": "Hon. Rodney Gilstrap",
        "court": "ed_tex",
        "appointed": 2011,
        "appointed_by": "Obama",
        "expertise": ["Patent litigation", "Complex civil"],
        "notable_cases": [
            "Handled more patent cases than any other judge",
            "VirnetX v. Apple",
        ],
        "characteristics": [
            "High volume patent docket",
            "Efficient case management",
            "Experienced with patent claim construction",
        ],
        "format_strictness": "standard",
        "oral_argument_frequency": "average",
    },
    "stark": {
        "name": "Hon. Leonard P. Stark",
        "court": "d_del",
        "appointed": 2010,
        "appointed_by": "Obama",
        "expertise": ["Patent", "Corporate", "ERISA"],
        "notable_cases": [
            "Major pharmaceutical patent disputes",
        ],
        "characteristics": [
            "Thorough claim construction analysis",
            "Academic approach to legal issues",
        ],
        "format_strictness": "standard",
        "oral_argument_frequency": "average",
    },
}


# Format requirement variations by court
FORMAT_VARIATIONS = {
    "page_limits": {
        "nd_cal": {"motion": 25, "reply": 15, "opposition": 25},
        "ed_tex": {"motion": 25, "reply": 10, "opposition": 25},
        "d_del": {"motion": 30, "reply": 15, "opposition": 30},
        "sdny": {"motion": 25, "reply": 15, "opposition": 25},
    },
    "font_requirements": {
        "nd_cal": {"font": "Times New Roman", "size": 14, "footnote_size": 12},
        "ed_tex": {"font": "Times New Roman or Century", "size": 14, "footnote_size": 12},
        "d_del": {"font": "Times New Roman", "size": 13, "footnote_size": 11},
        "sdny": {"font": "Times New Roman", "size": 12, "footnote_size": 10},
    },
    "line_spacing": {
        "nd_cal": "double",
        "ed_tex": "double",
        "d_del": "double",
        "sdny": "double",
    },
}


# Local rule differences that matter
LOCAL_RULE_DIFFERENCES = [
    {
        "topic": "Meet and Confer",
        "nd_cal": "Required for all discovery motions",
        "ed_tex": "Required, must certify in motion",
        "d_del": "Required, joint status report format",
    },
    {
        "topic": "Summary Judgment",
        "nd_cal": "Separate statement of undisputed facts required",
        "ed_tex": "Appendix of evidence required",
        "d_del": "Statement of material facts required",
    },
    {
        "topic": "Patent Local Rules",
        "nd_cal": "Detailed rules, infringement/invalidity contentions schedule",
        "ed_tex": "Similar to N.D. Cal, additional disclosure requirements",
        "d_del": "Default standard incorporated by reference",
    },
    {
        "topic": "E-Filing",
        "nd_cal": "All documents via CM/ECF",
        "ed_tex": "All documents via CM/ECF",
        "d_del": "All documents via CM/ECF",
    },
]


def get_court_data(court_code: str):
    """Get data for a specific court."""
    return COURT_COMPARISON.get(court_code)


def get_judge_profile(judge_id: str):
    """Get profile for a specific judge."""
    return JUDGE_PROFILES.get(judge_id)


def get_format_requirements(court_code: str):
    """Get format requirements for a court."""
    return {
        "page_limits": FORMAT_VARIATIONS["page_limits"].get(court_code, {}),
        "font": FORMAT_VARIATIONS["font_requirements"].get(court_code, {}),
        "line_spacing": FORMAT_VARIATIONS["line_spacing"].get(court_code, "double"),
    }


def compare_courts(court_codes: list):
    """Compare multiple courts."""
    return {code: COURT_COMPARISON.get(code) for code in court_codes if code in COURT_COMPARISON}
