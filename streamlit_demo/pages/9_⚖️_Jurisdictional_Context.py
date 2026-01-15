"""
Jurisdictional Context - Court and judge-specific guidance

This page demonstrates jurisdictional context with real data from
Judge William Alsup's N.D. California docket, including:
- Verified biographical information
- Notable case history with key quotes
- Pattern analysis examples
- Pre-filing checklist
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jurisdictions import NDCalContext, AlsupContext
from jurisdictions.base import MotionType

# Import real Alsup data
try:
    from examples.alsup_case_study import (
        ALSUP_TIMELINE,
        get_alsup_pattern_timeline,
        calculate_drift_example,
        confidence_calibration_example,
        generate_expertise_concentration,
        ORACLE_CASE_STUDY,
    )
    HAS_ALSUP_DATA = True
except ImportError:
    HAS_ALSUP_DATA = False

st.set_page_config(page_title="Jurisdictional Context | EFC", page_icon="⚖️", layout="wide")

st.title("⚖️ Jurisdictional Context")
st.markdown("*Court-specific and judge-specific guidance for legal analysis*")

st.markdown("""
Different courts and judges have different:
- **Format requirements** (page limits, fonts, margins)
- **Procedural rules** (filing deadlines, motion types)
- **Areas of expertise** (patent, antitrust, civil rights)
- **Behavioral patterns** (typical grant rates, timing)

This module makes that context available to inform your analysis.
""")

st.markdown("---")

# Court Selection
st.header("🏛️ Select Jurisdiction")

col1, col2 = st.columns(2)

with col1:
    court = st.selectbox(
        "Court",
        ["Northern District of California (N.D. Cal)"],
        key="court"
    )

with col2:
    judge = st.selectbox(
        "Judge (optional)",
        ["Any Judge", "Hon. William Alsup"],
        key="judge"
    )

# Load appropriate context
if judge == "Hon. William Alsup":
    context = AlsupContext()
    st.success("📋 Loading Judge Alsup's specific requirements and verified case data")
else:
    context = NDCalContext()

st.markdown("---")

# Judge Alsup Biography Section (only if Alsup is selected)
if isinstance(context, AlsupContext):
    st.header("👤 Judge William Alsup")

    # Get biography
    bio = context.get_biography()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); color: white; padding: 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 3em; margin-bottom: 0.5rem;">⚖️</div>
            <h2 style="margin: 0; color: white;">{bio["name"]}</h2>
            <p style="margin: 0.5rem 0; opacity: 0.9;">Senior U.S. District Judge</p>
            <p style="margin: 0; opacity: 0.8;">Northern District of California</p>
            <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
            <p style="font-size: 0.9em; margin: 0;">On the bench since 1999</p>
            <p style="font-size: 0.9em; margin: 0.25rem 0 0 0;">Senior status since 2021</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Education
        st.markdown("### 🎓 Education")
        for edu in bio["education"]:
            st.markdown(f"- **{edu['degree']}**, {edu['institution']} ({edu['year']})")

        # Notable attributes
        st.markdown("### ⭐ Notable Attributes")
        for attr in bio["notable_attributes"]:
            st.markdown(f"- {attr}")

    st.markdown("---")

    # Career Timeline
    st.header("📅 Career Timeline")

    career = bio["career"]
    appt = bio["judicial_appointment"]

    # Timeline visualization
    st.markdown("""
    <style>
    .timeline-item {
        display: flex;
        margin-bottom: 1rem;
    }
    .timeline-dot {
        width: 12px;
        height: 12px;
        background: #1e3a5f;
        border-radius: 50%;
        margin-right: 1rem;
        margin-top: 0.25rem;
        flex-shrink: 0;
    }
    .timeline-content {
        flex-grow: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    for item in career:
        st.markdown(f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>{item["position"]}</strong><br/>
                {item["employer"]}<br/>
                <small style="color: #666;">{item["years"]}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Judicial appointment highlight
    st.markdown(f"""
    <div style="background: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin-top: 1rem;">
        <strong>🏛️ Judicial Appointment</strong><br/>
        Nominated by {appt["nominated_by"]} on {appt["nominated"]}<br/>
        Confirmed {appt["confirmed"]}<br/>
        Senior Status {appt["senior_status"]}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Notable Cases Timeline
    if HAS_ALSUP_DATA:
        st.header("🔖 Notable Cases")

        st.markdown("Key rulings from Judge Alsup's docket, with verified quotes from public records:")

        # Get notable cases from the context
        notable_cases = context.get_notable_cases()

        for case in notable_cases:
            # Color coding by area
            area_colors = {
                "Patent": "#6f42c1",
                "Copyright": "#6f42c1",
                "Technology": "#6f42c1",
                "Trade": "#dc3545",
                "Administrative": "#17a2b8",
                "Immigration": "#17a2b8",
                "Environmental": "#28a745",
                "Education": "#fd7e14",
                "Employment": "#20c997",
            }

            color = "#6c757d"
            for keyword, c in area_colors.items():
                if keyword in case.area:
                    color = c
                    break

            # Quote section
            quote_html = ""
            if case.key_quote:
                quote_html = f"""
                <div style="background: #f8f9fa; padding: 1rem; border-left: 4px solid {color}; margin-top: 0.75rem; font-style: italic;">
                    "{case.key_quote}"
                </div>
                """

            st.markdown(f"""
            <div style="background: white; padding: 1.25rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <div>
                        <strong style="font-size: 1.1em; color: #1e3a5f;">{case.name}</strong>
                        <span style="background: {color}; color: white; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75em; margin-left: 0.5rem;">{case.area}</span>
                    </div>
                    <span style="color: #666; font-size: 0.9em;">{case.year}</span>
                </div>
                <p style="color: #495057; margin: 0.5rem 0;"><strong>Case No.:</strong> {case.case_number}</p>
                <p style="color: #495057; margin: 0.5rem 0;"><strong>Outcome:</strong> {case.outcome}</p>
                <p style="color: #666; margin: 0.5rem 0; font-size: 0.95em;">{case.significance}</p>
                {quote_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Key Quotes Section
        st.header("💬 Key Quotes")

        st.markdown("Notable statements from Judge Alsup's opinions and hearings:")

        quotes = context.get_key_quotes()

        cols = st.columns(2)
        for i, quote in enumerate(quotes):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.25rem; border-radius: 12px; margin-bottom: 1rem; min-height: 180px;">
                    <div style="font-size: 2em; color: #1e3a5f; line-height: 1;">"</div>
                    <p style="font-style: italic; color: #495057; margin: 0.5rem 0;">{quote["quote"]}</p>
                    <div style="border-top: 1px solid #dee2e6; padding-top: 0.75rem; margin-top: 0.75rem;">
                        <strong>{quote["case"]}</strong> ({quote["year"]})<br/>
                        <small style="color: #666;">{quote["context"]}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Oracle Case Study Deep Dive
        st.header("📚 Case Study: Oracle v. Google")

        st.markdown("""
        This landmark case demonstrates Judge Alsup's technical engagement
        and judicial approach. The case spanned over a decade and ultimately
        reached the Supreme Court.
        """)

        # Timeline visualization
        timeline_data = ORACLE_CASE_STUDY["key_dates"]

        st.markdown("### Timeline")

        timeline_items = [
            ("2010-08-12", "📁 Oracle files suit", "Patent and copyright claims over Java APIs"),
            ("2012-04-16", "⚖️ First trial begins", "Judge Alsup learns Java programming"),
            ("2012-05-31", "📜 Alsup ruling", "APIs not copyrightable - reversed by Fed. Cir."),
            ("2016-05-09", "⚖️ Second trial", "Fair use jury verdict - reversed by Fed. Cir."),
            ("2021-04-05", "🏛️ Supreme Court", "Fair use affirmed 6-2"),
        ]

        for date_str, title, desc in timeline_items:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                <div style="width: 100px; font-size: 0.85em; color: #666;">{date_str}</div>
                <div style="width: 30px; height: 30px; background: #1e3a5f; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8em; margin-right: 1rem;">→</div>
                <div>
                    <strong>{title}</strong><br/>
                    <small style="color: #666;">{desc}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Legacy
        st.markdown(f"""
        <div style="background: #e7f3ff; padding: 1.25rem; border-radius: 12px; margin-top: 1rem;">
            <h4 style="margin-top: 0;">🎯 Legacy</h4>
            <p style="margin-bottom: 0;">{ORACLE_CASE_STUDY["legacy"]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Pattern Analysis Section
        st.header("📊 Pattern Analysis")

        tab1, tab2 = st.tabs(["Drift Detection", "Confidence Calibration"])

        with tab1:
            st.markdown("### How Alsup's Patterns Evolved")

            drift_result = calculate_drift_example()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Period 1", drift_result["period1"])
            with col2:
                st.metric("Period 2", drift_result["period2"])
            with col3:
                similarity_pct = drift_result["similarity"] * 100
                st.metric("Similarity", f"{similarity_pct:.1f}%",
                         delta="-" if drift_result["drift_detected"] else "stable")

            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>Primary Change:</strong> {drift_result["primary_change"]}<br/>
                <strong>Magnitude:</strong> +{drift_result["change_magnitude"]:.0%}<br/>
                <p style="margin: 0.5rem 0 0 0; color: #495057;">{drift_result["interpretation"]}</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("### Calibrated Confidence Example")

            cal_result = confidence_calibration_example()

            st.markdown(f"**Scenario:** {cal_result['scenario']}")

            # Show adjustments
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Base → Adjusted")
                st.metric("Base Confidence", f"{cal_result['base_confidence']:.0%}")
                st.metric("Calibrated Confidence",
                         f"{cal_result['calibrated_confidence']:.0%}",
                         delta=f"{(cal_result['calibrated_confidence'] - cal_result['base_confidence']):.0%}")

            with col2:
                st.markdown("#### Confidence Interval")
                low, high = cal_result['confidence_interval']
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                    <strong>95% CI:</strong> {low:.0%} - {high:.0%}<br/>
                    <strong>Gate Decision:</strong>
                    <span style="background: #ffc107; padding: 0.25rem 0.5rem; border-radius: 4px;">
                        {cal_result['gate_decision'].upper()}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Adjustments table
            st.markdown("#### Adjustment Factors")
            for adj in cal_result['adjustments']:
                direction_icon = "📈" if adj["direction"] == "increase" else "📉" if adj["direction"] == "decrease" else "↔️"
                st.markdown(f"- {direction_icon} **{adj['factor']}**: {adj['reason']}")

            st.markdown(f"""
            <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>💡 Recommendation:</strong><br/>
                {cal_result['recommendation']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

# Context Summary (for both generic and Alsup)
st.header("📊 Context Summary")

summary = context.get_context_summary()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Court", summary["court_code"].upper().replace("_", " "))

with col2:
    st.metric("Format Rules", summary["format_requirements"])

with col3:
    st.metric("Procedural Rules", summary["procedural_rules"])

with col4:
    if summary["judge_name"]:
        st.metric("Judge", summary["judge_name"].split()[-1])
    else:
        st.metric("Judge", "General")

st.markdown("---")

# Format Requirements
st.header("📄 Format Requirements")

format_reqs = context.get_format_requirements()

col1, col2 = st.columns(2)

mandatory_reqs = [r for r in format_reqs if r.is_mandatory]
recommended_reqs = [r for r in format_reqs if not r.is_mandatory]

with col1:
    st.markdown("### Mandatory")
    for req in mandatory_reqs:
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #dc3545;">
        <strong>{req.name.replace('_', ' ').title()}</strong>: {req.value}<br/>
        <small style="color: #666;">{req.notes}</small>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### Recommended")
    if recommended_reqs:
        for req in recommended_reqs:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #17a2b8;">
            <strong>{req.name.replace('_', ' ').title()}</strong>: {req.value}<br/>
            <small style="color: #666;">{req.notes}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("*All requirements are mandatory for this judge.*")

st.markdown("---")

# Procedural Rules
st.header("📋 Procedural Rules")

# Motion type filter
motion_types = ["All Motions"] + [mt.value.replace("_", " ").title() for mt in MotionType]
selected_motion = st.selectbox("Filter by motion type:", motion_types)

rules = context.get_procedural_rules()

if selected_motion != "All Motions":
    # Convert back to enum
    selected_enum = MotionType(selected_motion.lower().replace(" ", "_"))
    rules = [r for r in rules if r.applies_to_motion(selected_enum)]

for rule in rules:
    mandatory_badge = "🔴 MANDATORY" if rule.is_mandatory else "🟢 Recommended"
    st.markdown(f"""
    <div style="background: #ffffff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #dee2e6;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="font-size: 1.1em;">{rule.title}</strong>
        <span style="font-size: 0.8em;">{mandatory_badge}</span>
    </div>
    <p style="margin: 0.5rem 0; color: #333;">{rule.description}</p>
    <small style="color: #666;">Source: {rule.source}</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Judge Expertise (if judge-specific)
if context.judge_name:
    st.header("🎓 Areas of Expertise")

    expertise = context.get_expertise_areas()

    for exp in expertise:
        level_colors = {
            "extensive": "#28a745",
            "moderate": "#ffc107",
            "limited": "#6c757d"
        }
        color = level_colors.get(exp.experience_level, "#6c757d")

        st.markdown(f"""
        <div style="background: linear-gradient(to right, {color}20, white); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {color};">
        <strong style="font-size: 1.1em;">{exp.area}</strong>
        <span style="background: {color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8em; margin-left: 0.5rem;">{exp.experience_level.upper()}</span>
        <p style="margin: 0.5rem 0;">{exp.description}</p>
        {"<small><strong>Notable cases:</strong> " + ", ".join(exp.notable_cases) + "</small>" if exp.notable_cases else ""}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Brief Checklist (Alsup-specific)
    if isinstance(context, AlsupContext):
        st.header("📝 Brief Checklist")

        st.markdown("Pre-filing checklist based on documented requirements:")

        checklist = context.get_brief_checklist()

        for section in checklist:
            with st.expander(f"✓ {section['item']}", expanded=False):
                for check in section['checks']:
                    st.checkbox(check, key=f"check_{hash(check)}")

        st.markdown("---")

        st.header("✍️ Style Guidance")

        guidance = context.get_style_guidance()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Do")
            for item in guidance["do"]:
                st.markdown(f"- {item}")

        with col2:
            st.markdown("### ❌ Avoid")
            for item in guidance["avoid"]:
                st.markdown(f"- {item}")

        # Lessons from cases
        if "lessons_from_cases" in guidance:
            st.markdown("### 📖 Lessons from Notable Cases")
            for case, lesson in guidance["lessons_from_cases"].items():
                st.markdown(f"- **{case}:** {lesson}")

        st.markdown(f"""
        <div style="background: #e7f3ff; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <strong>Note:</strong> {guidance["note"]}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Integration Example
st.header("🔗 Integration with Pattern Analysis")

st.markdown("""
When you configure the system with a jurisdiction, pattern analysis
automatically adjusts based on court-specific factors:
""")

st.code("""
from unified_system import EpistemicFlowControl, SystemConfig

# Configure for Judge Alsup's courtroom
config = SystemConfig(
    domain="judicial",
    jurisdiction="nd_cal",
    judge="alsup"
)

system = EpistemicFlowControl(config)

# Access judge-specific context
context = system.get_jurisdiction_context()
bio = context.get_biography()
cases = context.get_notable_cases(area="Patent")

# Pattern weights are automatically adjusted
# based on Alsup-specific factors
pattern = system.get_patterns_for_subject("tech_company_v_google")

# Format requirements are available for validation
requirements = system.get_format_requirements()
""", language="python")

# Important disclaimer
st.markdown("""
<div style="background: #fff3cd; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ffc107; margin-top: 2rem;">
<h3 style="margin-top: 0;">⚠️ Important Disclaimer</h3>
<p>This module provides <strong>style guidance and documented preferences</strong> based on publicly available information. It does <strong>NOT</strong>:</p>
<ul>
<li>Fabricate case citations or legal authorities</li>
<li>Make claims about undocumented preferences</li>
<li>Predict specific case outcomes</li>
<li>Replace professional legal judgment</li>
</ul>
<p><strong>Data Sources:</strong> Federal Judicial Center, PACER, published opinions, verified news coverage.</p>
<p style="margin-bottom: 0;">Always verify current standing orders and local rules before filing. This is guidance, not legal advice.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Return to the main demo to see the full system in action →")
