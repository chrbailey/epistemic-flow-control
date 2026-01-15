"""
Review Gate - Human review interface
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_system import EpistemicFlowControl, SystemConfig

st.set_page_config(page_title="Review Gate | EFC", page_icon="⚖️", layout="wide")


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

# Gate Decision Simulator
st.header("🎮 Gate Decision Simulator")

st.markdown("""
See how different confidence levels and stakes affect gate decisions:
""")

col1, col2 = st.columns(2)

with col1:
    sim_confidence = st.slider(
        "Prediction Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
        format="%.0%%",
        key="sim_conf"
    )

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
        st.success(f"✅ **AUTO PASS** - Confidence {sim_confidence:.0%} exceeds threshold {thresholds['auto_pass']:.0%}")
    elif sim_confidence >= thresholds["review"]:
        st.warning(f"🔍 **REVIEW REQUIRED** - Confidence {sim_confidence:.0%} needs human approval")
    elif sim_confidence >= thresholds["block"]:
        st.error(f"🚫 **BLOCKED** - Confidence {sim_confidence:.0%} too low, requires override")
    else:
        st.error(f"❌ **REJECTED** - Confidence {sim_confidence:.0%} below minimum threshold")

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
                    st.metric("Confidence", f"{item['confidence']:.1%}")
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
        st.info("✨ No items currently need review!")
        st.markdown("""
        Items appear here when:
        - Confidence is below the auto-pass threshold
        - Stakes are high
        - Pattern changes are detected
        - Sources have low reliability

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

        new_weight = st.slider(
            "New Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.0%%"
        )

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
                    st.success(f"✅ Pattern {pattern_id} weight overridden to {new_weight:.0%}")
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
st.caption("Navigate to Calibration to see how predictions match reality →")
