"""
Concentration Risk Scenarios

Example scenarios demonstrating different concentration patterns
and their associated SPOF risks.
"""

# Scenario 1: Healthy distribution
HEALTHY_DISTRIBUTION = {
    "name": "Healthy Case Distribution",
    "description": "Well-balanced workload across multiple judges",
    "entity_type": "judge",
    "domain": "patent",
    "data": {
        "Judge Anderson": 150,
        "Judge Brooks": 140,
        "Judge Chen": 130,
        "Judge Davis": 120,
        "Judge Edwards": 110,
        "Judge Franklin": 100,
    },
    "expected_hhi": 1700,  # Approximate
    "expected_level": "moderate",
    "has_spof": False,
}

# Scenario 2: Moderate concentration
MODERATE_CONCENTRATION = {
    "name": "Moderate Concentration",
    "description": "One dominant entity but not critical",
    "entity_type": "judge",
    "domain": "employment",
    "data": {
        "Judge Miller": 250,
        "Judge Wilson": 150,
        "Judge Taylor": 100,
        "Judge Brown": 75,
        "Others": 75,
    },
    "expected_hhi": 2400,  # Approximate
    "expected_level": "moderate",
    "has_spof": False,
}

# Scenario 3: High concentration (SPOF risk)
HIGH_CONCENTRATION = {
    "name": "E.D. Texas Patent Corridor",
    "description": "Realistic example of high concentration in patent venue",
    "entity_type": "judge",
    "domain": "patent",
    "data": {
        "Judge Gilstrap": 450,
        "Judge Payne": 180,
        "Judge Schroeder": 150,
        "Judge Love": 80,
        "Judge Mazzant": 60,
        "Others": 80,
    },
    "expected_hhi": 2800,  # Approximate
    "expected_level": "concentrated",
    "has_spof": True,
    "spof_entity": "Judge Gilstrap",
}

# Scenario 4: Near monopoly
NEAR_MONOPOLY = {
    "name": "Single Approver Bottleneck",
    "description": "One person handles most of the work - classic SPOF",
    "entity_type": "lawyer",
    "domain": "discovery_motions",
    "data": {
        "Partner Smith": 85,
        "Associate Johnson": 8,
        "Associate Chen": 5,
        "Others": 2,
    },
    "expected_hhi": 7300,  # Approximate
    "expected_level": "monopolistic",
    "has_spof": True,
    "spof_entity": "Partner Smith",
}

# Scenario 5: Perfect competition (theoretical)
PERFECT_COMPETITION = {
    "name": "Equal Distribution",
    "description": "Perfectly distributed workload (theoretical ideal)",
    "entity_type": "judge",
    "domain": "civil",
    "data": {
        "Judge A": 100,
        "Judge B": 100,
        "Judge C": 100,
        "Judge D": 100,
        "Judge E": 100,
    },
    "expected_hhi": 2000,  # Exactly 5 equal shares
    "expected_level": "moderate",
    "has_spof": False,
}

# Scenario 6: Lawyer concentration by firm
FIRM_CONCENTRATION = {
    "name": "Law Firm Market Share",
    "description": "Patent litigation market share in N.D. Cal",
    "entity_type": "firm",
    "domain": "patent_litigation",
    "data": {
        "Fish & Richardson": 180,
        "Quinn Emanuel": 150,
        "Morrison & Foerster": 120,
        "Perkins Coie": 100,
        "Cooley LLP": 80,
        "Other Firms": 370,
    },
    "expected_hhi": 1200,  # Approximate
    "expected_level": "unconcentrated",
    "has_spof": False,
}

# Scenario 7: Source concentration (data pipeline risk)
DATA_SOURCE_CONCENTRATION = {
    "name": "Data Source Dependency",
    "description": "Where your legal data comes from - API concentration risk",
    "entity_type": "data_source",
    "domain": "legal_data",
    "data": {
        "PACER": 600,
        "CourtListener": 200,
        "Bloomberg Law": 100,
        "Westlaw": 50,
        "Manual Entry": 50,
    },
    "expected_hhi": 4100,  # Approximate
    "expected_level": "highly_concentrated",
    "has_spof": True,
    "spof_entity": "PACER",
    "risk_note": "PACER API changes could break 60% of data pipeline",
}


# Time-series data for concentration evolution
CONCENTRATION_OVER_TIME = {
    "name": "Patent Venue Concentration Evolution",
    "description": "How E.D. Texas concentration changed over time",
    "periods": [
        {
            "period": "2015",
            "data": {
                "E.D. Texas": 1500,
                "D. Delaware": 800,
                "N.D. California": 600,
                "C.D. California": 400,
                "Others": 2700,
            },
            "hhi": 1400,
        },
        {
            "period": "2017 (Pre-TC Heartland)",
            "data": {
                "E.D. Texas": 2200,
                "D. Delaware": 600,
                "N.D. California": 500,
                "C.D. California": 350,
                "Others": 2350,
            },
            "hhi": 1800,
        },
        {
            "period": "2018 (Post-TC Heartland)",
            "data": {
                "E.D. Texas": 800,
                "D. Delaware": 1100,
                "N.D. California": 700,
                "C.D. California": 500,
                "Others": 2900,
            },
            "hhi": 1100,
        },
        {
            "period": "2023",
            "data": {
                "D. Delaware": 1200,
                "W.D. Texas": 1000,
                "E.D. Texas": 600,
                "N.D. California": 500,
                "Others": 2700,
            },
            "hhi": 1050,
        },
    ],
    "insight": "TC Heartland decision dramatically reduced E.D. Texas concentration",
}


def get_all_scenarios():
    """Get all concentration scenarios for demos."""
    return [
        HEALTHY_DISTRIBUTION,
        MODERATE_CONCENTRATION,
        HIGH_CONCENTRATION,
        NEAR_MONOPOLY,
        PERFECT_COMPETITION,
        FIRM_CONCENTRATION,
        DATA_SOURCE_CONCENTRATION,
    ]


def get_scenario_by_name(name: str):
    """Get a specific scenario by name."""
    for scenario in get_all_scenarios():
        if scenario["name"] == name:
            return scenario
    return None


def get_time_series_data():
    """Get time series concentration data."""
    return CONCENTRATION_OVER_TIME
