"""
Normalization Example Cases

Real-world examples of messy court data that needs cleaning.
These demonstrate the challenges of legal data normalization.
"""

# Judge name variations - same person, different formats
JUDGE_VARIATIONS = {
    "john_roberts": [
        "https://www.courtlistener.com/person/john-g-roberts-jr/",
        "ROBERTS, JOHN G. JR.",
        "John G. Roberts Jr.",
        "Chief Justice Roberts",
        "Hon. John Roberts",
        "john-g-roberts-jr",
        "J. Roberts",
    ],
    "william_alsup": [
        "https://courtlistener.com/person/william-h-alsup/",
        "ALSUP, WILLIAM H.",
        "William H. Alsup",
        "Judge Alsup",
        "Hon. William Alsup",
        "william-h-alsup",
        "W. Alsup",
    ],
    "lucy_koh": [
        "https://www.courtlistener.com/person/lucy-h-koh/",
        "KOH, LUCY H.",
        "Lucy H. Koh",
        "Judge Lucy Koh",
        "Hon. Lucy Koh",
        "lucy-h-koh",
    ],
    "rodney_gilstrap": [
        "https://courtlistener.com/person/rodney-gilstrap/",
        "GILSTRAP, RODNEY",
        "Rodney Gilstrap",
        "Judge Gilstrap",
        "Hon. Rodney Gilstrap",
        "rodney-gilstrap",
    ],
}

# Invalid lawyer entries that should be filtered
INVALID_LAWYER_ENTRIES = [
    # Geographic locations
    {"input": "San Francisco", "reason": "geographic", "notes": "City name mistakenly in lawyer field"},
    {"input": "New York, NY", "reason": "geographic", "notes": "City/state combination"},
    {"input": "Los Angeles", "reason": "geographic", "notes": "City name"},
    {"input": "California", "reason": "geographic", "notes": "State name"},
    {"input": "TX", "reason": "geographic", "notes": "State abbreviation"},

    # Organizations
    {"input": "Morrison & Foerster LLP", "reason": "organization", "notes": "Law firm, not individual"},
    {"input": "Wilson Sonsini Goodrich & Rosati PC", "reason": "organization", "notes": "Law firm"},
    {"input": "U.S. Department of Justice", "reason": "organization", "notes": "Government agency"},
    {"input": "Apple Inc.", "reason": "organization", "notes": "Corporate party"},
    {"input": "Google LLC", "reason": "organization", "notes": "Corporate party"},

    # Court personnel
    {"input": "Deputy Clerk", "reason": "court_personnel", "notes": "Court staff, not attorney"},
    {"input": "Court Reporter", "reason": "court_personnel", "notes": "Court staff"},
    {"input": "Clerk of Court", "reason": "court_personnel", "notes": "Court staff"},

    # Pro se indicators
    {"input": "Pro Se", "reason": "pro_se", "notes": "Self-represented party"},
    {"input": "In Propria Persona", "reason": "pro_se", "notes": "Latin for self-represented"},
    {"input": "Without Attorney", "reason": "pro_se", "notes": "No legal representation"},

    # Other invalid
    {"input": "TBD", "reason": "invalid_pattern", "notes": "Placeholder text"},
    {"input": "N/A", "reason": "invalid_pattern", "notes": "Not applicable marker"},
    {"input": "123456", "reason": "numeric", "notes": "Numeric only"},
]

# Valid lawyer entries with various formats
VALID_LAWYER_ENTRIES = [
    {"input": "John Smith", "expected": "John Smith"},
    {"input": "JOHNSON, MARY ANN", "expected": "Johnson, Mary Ann"},
    {"input": "Robert O'Brien Jr.", "expected": "Robert O'Brien Jr."},
    {"input": "Sarah Chen-Williams", "expected": "Sarah Chen-Williams"},
    {"input": "Michael Davis, Esq.", "expected": "Michael Davis, Esq."},
    {"input": "J. William Thompson III", "expected": "J. William Thompson III"},
    {"input": "lisa marie patterson", "expected": "Lisa Marie Patterson"},
]

# Batch processing test cases
BATCH_TEST_DATA = {
    "judges": [
        "https://courtlistener.com/person/william-h-alsup/",
        "CHEN, EDWARD M",
        "Hon. Lucy H. Koh",
        "richard-seeborg",
        "vince-chhabria",
        "Invalid Entry 123",
        "",
        "BREYER, CHARLES",
    ],
    "lawyers": [
        "Michael Smith",
        "San Jose",
        "Lisa Johnson, Esq.",
        "Pro Se",
        "Wilson Sonsini LLP",
        "Sarah Chen",
        "Deputy Clerk",
        "Robert O'Brien Jr.",
        "California",
        "Jane Doe",
    ],
    "expected_valid_judges": 7,
    "expected_valid_lawyers": 4,
}


def get_normalization_demo_data():
    """Get structured data for normalization demos."""
    return {
        "judge_variations": JUDGE_VARIATIONS,
        "invalid_lawyers": INVALID_LAWYER_ENTRIES,
        "valid_lawyers": VALID_LAWYER_ENTRIES,
        "batch_test": BATCH_TEST_DATA,
    }
