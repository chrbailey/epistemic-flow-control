"""
Concentration Risk Analysis - Detect SPOF risks using HHI
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concentration import HHICalculator, SPOFDetector, ConcentrationLevel

st.set_page_config(page_title="Concentration Risk | EFC", page_icon="⚠️", layout="wide")

st.title("⚠️ Concentration Risk Analysis")
st.markdown("*Detect Single Points of Failure using the Herfindahl-Hirschman Index*")

st.markdown("""
The **HHI (Herfindahl-Hirschman Index)** measures market concentration.
Originally used in antitrust economics, it's perfect for detecting when:
- One judge handles too many cases
- One lawyer dominates a practice area
- One firm controls a market segment

**Scale:** 0 (perfect competition) → 10,000 (pure monopoly)
""")

st.markdown("---")

# Interactive HHI Calculator
st.header("📊 Interactive HHI Calculator")

st.markdown("Enter entity shares (must sum to 100%):")

calc = HHICalculator()

col1, col2 = st.columns([2, 1])

with col1:
    # Preset scenarios
    scenario = st.selectbox(
        "Choose a scenario or enter custom values:",
        [
            "Custom",
            "Healthy Distribution (4 equal firms)",
            "Moderate Concentration (one firm at 40%)",
            "High Concentration (one firm at 60%)",
            "Near Monopoly (one firm at 85%)",
            "Patent Troll Corridor (realistic E.D. Tex example)"
        ]
    )

    # Preset values
    if scenario == "Healthy Distribution (4 equal firms)":
        default_shares = {"Firm A": 25, "Firm B": 25, "Firm C": 25, "Firm D": 25}
    elif scenario == "Moderate Concentration (one firm at 40%)":
        default_shares = {"Firm A": 40, "Firm B": 30, "Firm C": 20, "Firm D": 10}
    elif scenario == "High Concentration (one firm at 60%)":
        default_shares = {"Firm A": 60, "Firm B": 20, "Firm C": 15, "Firm D": 5}
    elif scenario == "Near Monopoly (one firm at 85%)":
        default_shares = {"Firm A": 85, "Firm B": 10, "Firm C": 5}
    elif scenario == "Patent Troll Corridor (realistic E.D. Tex example)":
        default_shares = {"Judge Gilstrap": 45, "Judge Payne": 25, "Judge Schroeder": 20, "Others": 10}
    else:
        default_shares = {"Entity A": 40, "Entity B": 30, "Entity C": 20, "Entity D": 10}

    # Editable shares
    shares = {}
    total = 0
    for entity, default_val in default_shares.items():
        val = st.slider(
            entity,
            min_value=0,
            max_value=100,
            value=default_val,
            key=f"slider_{entity}"
        )
        if val > 0:
            shares[entity] = val
            total += val

    # Show total
    if total != 100:
        st.warning(f"Total: {total}% (should be 100%)")
    else:
        st.success(f"Total: {total}% ✓")

with col2:
    st.markdown("### Results")

    if shares:
        # Normalize to shares that sum to 1.0
        normalized_shares = {k: v / total for k, v in shares.items()}
        result = calc.from_shares(normalized_shares)

        # HHI Gauge
        st.metric(
            "HHI Score",
            f"{result.hhi:,.0f}",
            help="0-10,000 scale. <1,500 = healthy, >2,500 = concentrated"
        )

        # Color-coded level
        level_colors = {
            ConcentrationLevel.UNCONCENTRATED: "🟢",
            ConcentrationLevel.MODERATE: "🟡",
            ConcentrationLevel.CONCENTRATED: "🟠",
            ConcentrationLevel.HIGHLY_CONCENTRATED: "🔴",
            ConcentrationLevel.MONOPOLISTIC: "⚫"
        }

        st.markdown(f"### {level_colors[result.level]} {result.level.value.replace('_', ' ').title()}")

        st.metric("Top Entity", f"{result.top_entity}: {result.top_share:.0%}")
        st.metric("Equivalent Firms", f"{result.equivalent_firms:.1f}")

st.markdown("---")

# SPOF Detection Demo
st.header("🎯 SPOF Risk Detection")

st.markdown("""
Beyond overall concentration, we need to identify **specific entities** that create
Single Point of Failure (SPOF) risk. If they become unavailable, operations are disrupted.
""")

detector = SPOFDetector()

# Example: Judge caseload distribution
st.markdown("### Example: Patent Case Distribution by Judge")

example_data = {
    "Judge Gilstrap": 450,
    "Judge Payne": 180,
    "Judge Schroeder": 150,
    "Judge Love": 80,
    "Judge Mazzant": 60,
    "Other Judges": 80
}

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Case Distribution")
    total_cases = sum(example_data.values())
    for judge, cases in example_data.items():
        pct = cases / total_cases
        st.progress(pct, text=f"{judge}: {cases} cases ({pct:.0%})")

with col2:
    assessment = detector.analyze(
        example_data,
        entity_type="judge",
        domain="patent"
    )

    st.markdown("#### Risk Assessment")
    st.markdown(f"**Overall Health:** {assessment.overall_health}")

    if assessment.has_critical_spof:
        st.error("⚠️ CRITICAL SPOF DETECTED")

    for risk in assessment.spof_risks[:3]:
        if risk.risk_level.value == "critical":
            st.error(f"🔴 **{risk.entity_id}**: {risk.share:.0%} share")
        elif risk.risk_level.value == "high":
            st.warning(f"🟠 **{risk.entity_id}**: {risk.share:.0%} share")
        elif risk.risk_level.value == "moderate":
            st.info(f"🟡 **{risk.entity_id}**: {risk.share:.0%} share")
        else:
            st.success(f"🟢 **{risk.entity_id}**: {risk.share:.0%} share")

# Show recommendations
st.markdown("### Recommendations")
for risk in assessment.spof_risks[:2]:
    if risk.is_spof:
        st.markdown(f"""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
        <strong>{risk.entity_id}</strong><br/>
        {risk.recommendation}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# HHI Reference Table
st.header("📚 HHI Reference Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### DOJ Merger Guidelines

    | HHI Range | Classification | Action |
    |-----------|----------------|--------|
    | < 1,500 | Unconcentrated | Generally no concerns |
    | 1,500 - 2,500 | Moderately concentrated | Monitor |
    | > 2,500 | Highly concentrated | Review required |
    """)

with col2:
    st.markdown("""
    ### Legal Analytics Thresholds

    | Entity Share | Risk Level | Action |
    |--------------|------------|--------|
    | < 15% | Low | Healthy |
    | 15-25% | Moderate | Monitor |
    | 25-40% | High | Develop backup |
    | > 40% | Critical | Immediate action |
    """)

st.markdown("---")

# Insight box
st.markdown("""
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 12px; color: white;">
<h3 style="margin-top: 0; color: white;">💡 Why This Matters for Legal Tech</h3>
<p>In legal analytics, concentration risk manifests in several ways:</p>
<ul>
<li><strong>Judge Concentration:</strong> If one judge handles 45% of patent cases, their retirement creates chaos</li>
<li><strong>Lawyer Concentration:</strong> If one partner approves all filings, vacation = bottleneck</li>
<li><strong>Data Source Concentration:</strong> If 90% of data comes from one source, API changes break everything</li>
</ul>
<p style="margin-bottom: 0;">HHI and SPOF detection help you identify these risks <em>before</em> they become crises.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Navigate to Pattern Drift to see how patterns change over time →")
