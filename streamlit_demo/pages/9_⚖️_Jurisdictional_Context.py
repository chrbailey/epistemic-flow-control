"""
Jurisdictional Context - Court and judge-specific guidance
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jurisdictions import NDCalContext, AlsupContext
from jurisdictions.base import MotionType

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
    st.info("📋 Loading Judge Alsup's specific requirements and preferences")
else:
    context = NDCalContext()

st.markdown("---")

# Context Summary
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

    # Judge Alsup specific sections
    if isinstance(context, AlsupContext):
        st.header("📝 Brief Checklist")

        st.markdown("Pre-filing checklist based on documented requirements:")

        checklist = context.get_brief_checklist()

        for section in checklist:
            with st.expander(f"✓ {section['item']}", expanded=True):
                for check in section['checks']:
                    st.checkbox(check, key=f"check_{check[:20]}")

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
<p style="margin-bottom: 0;">Always verify current standing orders and local rules before filing. This is guidance, not legal advice.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Return to the main demo to see the full system in action →")
