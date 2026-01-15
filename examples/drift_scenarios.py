"""
Pattern Drift Scenarios

Example scenarios demonstrating different types of pattern drift
and their causes.
"""

# Baseline pattern for comparison
BASELINE_PATTERN = {
    "entity_id": "judge_example",
    "pattern_type": "summary_judgment",
    "metrics": {
        "grant_rate": 0.45,
        "denial_rate": 0.40,
        "partial_grant_rate": 0.15,
        "avg_days_to_decision": 120,
        "median_days": 105,
        "oral_argument_rate": 0.30,
        "discovery_limit_rate": 0.65,
        "extension_grant_rate": 0.85,
    },
    "sample_count": 500,
    "period": "2020-2023",
}


# Scenario 1: No drift (stable pattern)
NO_DRIFT = {
    "name": "Stable Pattern",
    "description": "Pattern remains consistent with baseline",
    "cause": "Normal variation within expected bounds",
    "metrics": {
        "grant_rate": 0.47,
        "denial_rate": 0.38,
        "partial_grant_rate": 0.15,
        "avg_days_to_decision": 115,
        "median_days": 100,
        "oral_argument_rate": 0.32,
        "discovery_limit_rate": 0.63,
        "extension_grant_rate": 0.87,
    },
    "expected_similarity": 0.98,
    "expected_severity": "negligible",
    "action": "Continue monitoring",
}

# Scenario 2: Minor drift
MINOR_DRIFT = {
    "name": "Minor Grant Rate Shift",
    "description": "Small but noticeable change in grant rate",
    "cause": "Slightly different case mix or natural evolution",
    "metrics": {
        "grant_rate": 0.52,
        "denial_rate": 0.35,
        "partial_grant_rate": 0.13,
        "avg_days_to_decision": 110,
        "median_days": 95,
        "oral_argument_rate": 0.35,
        "discovery_limit_rate": 0.60,
        "extension_grant_rate": 0.85,
    },
    "expected_similarity": 0.90,
    "expected_severity": "minor",
    "action": "Monitor for trend continuation",
}

# Scenario 3: Gradual drift
GRADUAL_DRIFT = {
    "name": "Gradual Philosophy Shift",
    "description": "Judge gradually becomes more plaintiff-friendly",
    "cause": "Evolving judicial philosophy over time",
    "metrics": {
        "grant_rate": 0.58,
        "denial_rate": 0.30,
        "partial_grant_rate": 0.12,
        "avg_days_to_decision": 100,
        "median_days": 90,
        "oral_argument_rate": 0.40,
        "discovery_limit_rate": 0.55,
        "extension_grant_rate": 0.80,
    },
    "expected_similarity": 0.78,
    "expected_severity": "moderate",
    "action": "Schedule recalibration, update baseline",
}

# Scenario 4: Sudden drift (new judge)
SUDDEN_DRIFT_NEW_JUDGE = {
    "name": "New Judge Takes Over",
    "description": "Dramatic change when a new judge inherits caseload",
    "cause": "Judge retirement/replacement",
    "metrics": {
        "grant_rate": 0.65,
        "denial_rate": 0.25,
        "partial_grant_rate": 0.10,
        "avg_days_to_decision": 80,
        "median_days": 70,
        "oral_argument_rate": 0.15,
        "discovery_limit_rate": 0.75,
        "extension_grant_rate": 0.95,
    },
    "expected_similarity": 0.55,
    "expected_severity": "severe",
    "action": "IMMEDIATE: Reset baseline for new judge",
}

# Scenario 5: Sudden drift (precedent change)
SUDDEN_DRIFT_PRECEDENT = {
    "name": "Supreme Court Precedent Change",
    "description": "Major precedent changes standard for summary judgment",
    "cause": "New Supreme Court ruling changes legal standard",
    "metrics": {
        "grant_rate": 0.30,
        "denial_rate": 0.55,
        "partial_grant_rate": 0.15,
        "avg_days_to_decision": 140,
        "median_days": 125,
        "oral_argument_rate": 0.45,
        "discovery_limit_rate": 0.70,
        "extension_grant_rate": 0.85,
    },
    "expected_similarity": 0.68,
    "expected_severity": "significant",
    "action": "Investigate precedent change, recalibrate all affected patterns",
}

# Scenario 6: Seasonal drift
SEASONAL_DRIFT = {
    "name": "End-of-Year Rush",
    "description": "Pattern changes due to year-end deadline pressure",
    "cause": "Seasonal workload and deadline effects",
    "metrics": {
        "grant_rate": 0.55,
        "denial_rate": 0.35,
        "partial_grant_rate": 0.10,
        "avg_days_to_decision": 60,
        "median_days": 50,
        "oral_argument_rate": 0.15,
        "discovery_limit_rate": 0.80,
        "extension_grant_rate": 0.60,
    },
    "expected_similarity": 0.75,
    "expected_severity": "moderate",
    "action": "Account for seasonality, expect reversion in new year",
}

# Scenario 7: Caseload shift drift
CASELOAD_SHIFT = {
    "name": "Case Type Mix Change",
    "description": "Different types of cases affecting patterns",
    "cause": "Court received influx of new case type",
    "metrics": {
        "grant_rate": 0.35,
        "denial_rate": 0.50,
        "partial_grant_rate": 0.15,
        "avg_days_to_decision": 150,
        "median_days": 140,
        "oral_argument_rate": 0.50,
        "discovery_limit_rate": 0.55,
        "extension_grant_rate": 0.90,
    },
    "expected_similarity": 0.72,
    "expected_severity": "moderate",
    "action": "Segment patterns by case type for better accuracy",
}


# Time series showing drift evolution
DRIFT_TIMELINE = {
    "entity_id": "judge_gilstrap",
    "pattern_type": "patent_case_duration",
    "checkpoints": [
        {
            "date": "2020-01",
            "similarity_to_baseline": 1.0,
            "grant_rate": 0.45,
            "note": "Baseline established",
        },
        {
            "date": "2020-06",
            "similarity_to_baseline": 0.97,
            "grant_rate": 0.46,
            "note": "Stable",
        },
        {
            "date": "2021-01",
            "similarity_to_baseline": 0.94,
            "grant_rate": 0.48,
            "note": "Minor drift",
        },
        {
            "date": "2021-06",
            "similarity_to_baseline": 0.88,
            "grant_rate": 0.52,
            "note": "Trend emerging",
        },
        {
            "date": "2022-01",
            "similarity_to_baseline": 0.82,
            "grant_rate": 0.55,
            "note": "Moderate drift - investigation triggered",
        },
        {
            "date": "2022-06",
            "similarity_to_baseline": 0.78,
            "grant_rate": 0.58,
            "note": "Pattern confirmed - recalibration performed",
        },
        {
            "date": "2023-01",
            "similarity_to_baseline": 0.95,  # New baseline
            "grant_rate": 0.57,
            "note": "New baseline, stable",
        },
    ],
}


def get_all_drift_scenarios():
    """Get all drift scenarios for demos."""
    return [
        NO_DRIFT,
        MINOR_DRIFT,
        GRADUAL_DRIFT,
        SUDDEN_DRIFT_NEW_JUDGE,
        SUDDEN_DRIFT_PRECEDENT,
        SEASONAL_DRIFT,
        CASELOAD_SHIFT,
    ]


def get_baseline():
    """Get the baseline pattern for comparisons."""
    return BASELINE_PATTERN


def get_drift_timeline():
    """Get the drift timeline example."""
    return DRIFT_TIMELINE
