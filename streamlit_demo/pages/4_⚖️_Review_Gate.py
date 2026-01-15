"""
Review Gate - Human review interface
Enhanced with truth-validator checklist integration
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_system import EpistemicFlowControl, SystemConfig

st.set_page_config(page_title="Review Gate | EFC", page_icon="⚖️", layout="wide")

# Initialize session state for persistent toggle (shared across pages)
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = True


def get_system():
    if "system" not in st.session_state:
        config = SystemConfig(db_dir="./demo_data", domain="judicial")
        st.session_state.system = EpistemicFlowControl(config)
    return st.session_state.system


st.title("⚖️ Human Review Gate")
st.markdown("*The checkpoint where humans control the flow*")

system = get_system()

# Import chart components
try:
    from streamlit_demo.components.charts import plot_confidence_gauge
    charts_available = True
except ImportError:
    charts_available = False

st.markdown("---")

# Operating Mode Status
mode_col1, mode_col2, mode_col3 = st.columns([1, 1, 2])

with mode_col1:
    ai_mode = st.toggle(
        "AI Auto-Routing",
        value=st.session_state.ai_mode,
        key="review_ai_mode_toggle",
        help="Toggle between AI-managed and human-controlled modes"
    )
    # Update shared session state
    st.session_state.ai_mode = ai_mode

with mode_col2:
    if ai_mode:
        st.success("🤖 **AI MANAGED**")
    else:
        st.warning("👤 **HUMAN CONTROLLED**")

with mode_col3:
    if ai_mode:
        st.caption("High-confidence items auto-routed. You review exceptions and monitor calibration.")
    else:
        st.caption("All items require your explicit approval.")

st.markdown("---")

# Gate Decision Simulator
st.header("🎮 Gate Decision Simulator")

st.markdown("""
See how different confidence levels and stakes affect gate decisions:
""")

col1, col2 = st.columns(2)

with col1:
    # Use 0-100 integer slider to avoid sprintf formatting issues
    sim_confidence_pct = st.slider(
        "Prediction Confidence",
        min_value=0,
        max_value=100,
        value=75,
        step=1,
        format="%d%%",
        key="sim_conf"
    )
    sim_confidence = sim_confidence_pct / 100.0

with col2:
    sim_stakes = st.radio(
        "Stakes Level",
        options=["low", "medium", "high"],
        horizontal=True,
        key="sim_stakes"
    )

# Calculate thresholds with stakes penalty
stakes_penalties = {"low": 0.0, "medium": 0.05, "high": 0.10}
penalty = stakes_penalties[sim_stakes]

thresholds = {
    "auto_pass": 0.92 - penalty,
    "review": 0.70 - penalty,
    "block": 0.50 - penalty
}

# Visualize
if charts_available:
    fig = plot_confidence_gauge(sim_confidence, thresholds)
    st.plotly_chart(fig, use_container_width=True)
else:
    # Text-based visualization
    if sim_confidence >= thresholds["auto_pass"]:
        if ai_mode:
            st.success(f"✅ **AUTO PASS** - Confidence {sim_confidence_pct}% exceeds threshold {thresholds['auto_pass']:.0%} → Routed to production")
        else:
            st.success(f"✅ **READY FOR APPROVAL** - Confidence {sim_confidence_pct}% exceeds threshold {thresholds['auto_pass']:.0%}")
    elif sim_confidence >= thresholds["review"]:
        st.warning(f"🔍 **REVIEW REQUIRED** - Confidence {sim_confidence_pct}% needs human approval")
    elif sim_confidence >= thresholds["block"]:
        st.error(f"🚫 **BLOCKED** - Confidence {sim_confidence_pct}% too low, requires override")
    else:
        st.error(f"❌ **REJECTED** - Confidence {sim_confidence_pct}% below minimum threshold")

# Threshold explanation
with st.expander("📊 How Thresholds Work"):
    st.markdown(f"""
    | Stakes | Auto Pass | Review | Block | Rejected |
    |--------|-----------|--------|-------|----------|
    | Low | ≥92% | 70-92% | 50-70% | <50% |
    | Medium | ≥87% | 65-87% | 45-65% | <45% |
    | High | ≥82% | 60-82% | 40-60% | <40% |

    **Current ({sim_stakes} stakes):**
    - Auto Pass: ≥{thresholds['auto_pass']:.0%}
    - Review: {thresholds['block']:.0%}-{thresholds['auto_pass']:.0%}
    - Blocked: <{thresholds['block']:.0%}

    High-stakes decisions have **lower thresholds** because more human oversight
    is appropriate when consequences are severe.
    """)

st.markdown("---")

# Truth-Validator Integration
st.header("📋 Truth-Validator Checklist")

st.markdown("""
<div style="background: #fff8e1; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">

**From truth-validator skill**: Items are flagged for your review based on claim type.
This checklist helps you know WHAT to verify, not WHETHER claims are correct.

</div>
""", unsafe_allow_html=True)

# Flag types from truth-validator
flag_types = {
    "NEEDS_CITATION": {"emoji": "📚", "color": "#e57373", "action": "Verify the source exists and supports the claim"},
    "PARAPHRASE_CHECK": {"emoji": "✍️", "color": "#ffb74d", "action": "Compare to original source, confirm meaning preserved"},
    "INFERENCE_FLAG": {"emoji": "🧠", "color": "#ba68c8", "action": "Decide if inference is warranted or should be removed"},
    "CALCULATION_VERIFY": {"emoji": "🔢", "color": "#4fc3f7", "action": "Verify the math independently"},
    "ASSUMPTION_FLAG": {"emoji": "⚠️", "color": "#fff176", "action": "Confirm assumption is valid or make it explicit"},
    "SCOPE_QUESTION": {"emoji": "🔍", "color": "#81c784", "action": "Verify scope claim is supported"}
}

# Sample items for demonstration
sample_items = [
    {"flag": "NEEDS_CITATION", "claim": "The ruling established that software patents require specific technical improvement", "location": "Section 2.1"},
    {"flag": "CALCULATION_VERIFY", "claim": "This represents a 34% increase in case volume year-over-year", "location": "Summary"},
    {"flag": "INFERENCE_FLAG", "claim": "This suggests the court is moving toward stricter interpretation", "location": "Section 4.3"},
]

if sample_items:
    st.markdown("**Items flagged for your review:**")

    for i, item in enumerate(sample_items):
        flag_info = flag_types[item["flag"]]
        with st.expander(f"{flag_info['emoji']} {item['flag']}: {item['claim'][:50]}..."):
            st.markdown(f"""
            **Claim**: "{item['claim']}"

            **Location**: {item['location']}

            **Your Action**: {flag_info['action']}
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Verified", key=f"verify_{i}"):
                    st.success("Marked as verified")
            with col2:
                if st.button("✏️ Edit Needed", key=f"edit_{i}"):
                    st.warning("Flagged for editing")
            with col3:
                if st.button("❌ Remove", key=f"remove_{i}"):
                    st.error("Marked for removal")

st.markdown("---")

# Review Queue
st.header("📋 Review Queue")

try:
    items = system.get_items_needing_review(max_items=10)

    if items:
        st.markdown(f"**{len(items)} items awaiting review:**")

        for item in items:
            with st.expander(f"🔍 {item['item_type'].upper()}: {item['item_id'][:20]}... ({item['confidence']:.0%} confidence)"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Content:**")
                    st.json(item.get('content', {}))

                    st.markdown("**Gate Reasoning:**")
                    st.info(item.get('reasoning', 'Standard review required'))

                with col2:
                    st.metric("Confidence", f"{item['confidence']:.0%}")
                    st.metric("Stakes", item.get('stakes', 'medium').upper())

                    st.markdown("---")

                    # Review actions
                    st.markdown("**Your Decision:**")

                    decision = st.radio(
                        "Action",
                        options=["approve", "reject", "modify"],
                        key=f"decision_{item['item_id']}",
                        horizontal=True
                    )

                    notes = st.text_area(
                        "Notes",
                        placeholder="Reasoning for your decision...",
                        key=f"notes_{item['item_id']}"
                    )

                    if st.button("Submit Review", key=f"submit_{item['item_id']}"):
                        try:
                            success = system.submit_human_review(
                                item_id=item['item_id'],
                                reviewer_id="demo_user",
                                decision=decision,
                                notes=notes
                            )
                            if success:
                                st.success("✅ Review submitted!")
                                st.rerun()
                            else:
                                st.error("Failed to submit review")
                        except Exception as e:
                            st.error(f"Error: {e}")
    else:
        if ai_mode:
            st.info("✨ No items currently need review! AI auto-routing is handling high-confidence items.")
        else:
            st.info("✨ No items currently need review!")

        st.markdown("""
        Items appear here when:
        - Confidence is below the auto-pass threshold
        - Stakes are high
        - Pattern changes are detected
        - Sources have low reliability
        - **Truth-validator flags claims for verification**

        **Try making a prediction with medium confidence on the Live System page!**
        """)

except Exception as e:
    st.error(f"Could not load review queue: {e}")

st.markdown("---")

# Gate Analytics
st.header("📊 Gate Analytics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Decision Distribution")

    # Simulated data for demo
    st.markdown("""
    | Decision | Count | % |
    |----------|-------|---|
    | Auto Pass | 45 | 56% |
    | Review → Approved | 25 | 31% |
    | Review → Rejected | 8 | 10% |
    | Blocked | 2 | 3% |
    """)

with col2:
    st.markdown("### Gate Effectiveness")

    st.metric("Auto-Pass Accuracy", "94%", delta="+2%", help="How often auto-passed items were correct")
    st.metric("Review Value-Add", "23%", help="Errors caught by human review")
    st.metric("Block Rate", "3%", help="Items blocked from production")

st.markdown("---")

# Human Override Section
st.header("🎚️ Human Override")

st.markdown("""
Domain experts can override pattern weights when they have knowledge the system lacks:
""")

with st.form("override_form"):
    col1, col2 = st.columns(2)

    with col1:
        pattern_id = st.text_input(
            "Pattern ID",
            placeholder="pat_rodriguez_sj_001",
            help="ID of the pattern to override"
        )

        # Use 0-100 integer slider
        new_weight_pct = st.slider(
            "New Weight",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            format="%d%%"
        )
        new_weight = new_weight_pct / 100.0

    with col2:
        reason = st.text_area(
            "Override Reason",
            placeholder="Recent court policy change affects this pattern...",
            help="Explain why you're overriding the computed weight"
        )

        overrider = st.text_input(
            "Your ID",
            value="expert_001",
            help="Your reviewer ID"
        )

    if st.form_submit_button("Submit Override"):
        if pattern_id and reason:
            try:
                success = system.human_override_pattern(
                    pattern_id=pattern_id,
                    new_weight=new_weight,
                    reason=reason,
                    overrider=overrider
                )
                if success:
                    st.success(f"✅ Pattern {pattern_id} weight overridden to {new_weight_pct}%")
                else:
                    st.error("Failed to override pattern (pattern may not exist)")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please provide pattern ID and reason")

st.markdown("""
<div style="background: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #F5A623;">
<strong>⚠️ Important:</strong> Human overrides are tracked and their outcomes are measured.
If overrides are frequently wrong, the system learns to weight them differently.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Monitoring Dashboard (for AI mode)
if ai_mode:
    st.header("📈 AI Monitoring Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Items Auto-Routed Today", "127", delta="+12")

    with col2:
        st.metric("Human Reviews Today", "8", delta="-3")

    with col3:
        st.metric("Calibration Score", "0.94", delta="+0.01")

    with col4:
        st.metric("Drift Alert", "None", help="No significant drift detected")

    st.markdown("""
    <div style="background: #e8f5e9; padding: 1rem; border-radius: 8px; border-left: 4px solid #4caf50;">
    <strong>✅ System Status:</strong> Operating normally. 94% of items auto-routed with
    maintained calibration accuracy. No intervention required.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Navigate to Calibration to see how predictions match reality →")
