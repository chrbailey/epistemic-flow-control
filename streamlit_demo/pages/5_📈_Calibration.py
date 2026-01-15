"""
Calibration Dashboard - Track prediction accuracy
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_system import EpistemicFlowControl, SystemConfig

st.set_page_config(page_title="Calibration | EFC", page_icon="📈", layout="wide")


def get_system():
    if "system" not in st.session_state:
        config = SystemConfig(db_dir="./demo_data", domain="judicial")
        st.session_state.system = EpistemicFlowControl(config)
    return st.session_state.system


st.title("📈 Calibration Dashboard")
st.markdown("*Ensuring confidence scores match actual accuracy*")

system = get_system()

# Import chart components
try:
    from streamlit_demo.components.charts import plot_calibration_curve
    charts_available = True
except ImportError:
    charts_available = False

st.markdown("---")

# What is Calibration?
st.header("🎯 What is Calibration?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### The Goal

    A **well-calibrated** system means:

    - When we say **80% confident**, we're right **80% of the time**
    - When we say **50% confident**, we're right **50% of the time**

    This is **essential** for high-stakes decisions. If the system says
    "90% likely to win" but is only right 60% of the time, decisions
    based on that confidence are dangerously misguided.
    """)

with col2:
    st.markdown("""
    ### The Metric: ECE

    **Expected Calibration Error (ECE)** measures how far off our
    confidence scores are from reality:

    - **ECE = 0%**: Perfect calibration
    - **ECE < 5%**: Well calibrated
    - **ECE 5-10%**: Acceptable
    - **ECE > 10%**: Needs attention

    Lower is better!
    """)

st.markdown("---")

# Calibration Status
st.header("📊 Current Calibration Status")

try:
    status = system.get_calibration_status()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Domain", status.get('domain', 'judicial').upper())

    with col2:
        factor = status.get('current_factor', 1.0)
        st.metric(
            "Calibration Factor",
            f"{factor:.2f}",
            delta=f"{factor - 1.0:+.2f}" if factor != 1.0 else None,
            help="Multiplier applied to raw confidence. <1 means overconfident."
        )

    with col3:
        calibration_data = status.get('calibration_data', {})
        ece = calibration_data.get('expected_calibration_error', 0)
        st.metric(
            "ECE",
            f"{ece:.1%}" if ece else "N/A",
            help="Expected Calibration Error"
        )

    # Calibration curve
    st.markdown("### Calibration Curve")

    if calibration_data and 'calibration_curve' in calibration_data and charts_available:
        fig = plot_calibration_curve(calibration_data)
        st.plotly_chart(fig, use_container_width=True)
    elif calibration_data and 'calibration_curve' in calibration_data:
        # Text fallback
        curve = calibration_data['calibration_curve']
        st.markdown("| Expected | Actual | Error | Samples |")
        st.markdown("|----------|--------|-------|---------|")
        for bucket in curve:
            expected = bucket.get('expected_accuracy', 0)
            actual = bucket.get('actual_accuracy', 0)
            error = actual - expected
            samples = bucket.get('sample_size', 0)
            st.markdown(f"| {expected:.0%} | {actual:.0%} | {error:+.0%} | {samples} |")
    else:
        st.info("Insufficient data for calibration curve. Need more prediction outcomes.")

    # Recommendations
    st.markdown("### 💡 Recommendations")

    recommendations = status.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            if "overconfident" in rec.lower():
                st.warning(f"⚠️ {rec}")
            elif "good" in rec.lower():
                st.success(f"✅ {rec}")
            else:
                st.info(f"ℹ️ {rec}")
    else:
        st.success("✅ No calibration issues detected")

except Exception as e:
    st.error(f"Could not load calibration status: {e}")
    st.info("Make predictions and record outcomes to see calibration data.")

st.markdown("---")

# Interactive Calibration Demo
st.header("🎮 Interactive: Understanding Calibration")

st.markdown("""
Imagine a system makes predictions at different confidence levels.
Let's see if it's well-calibrated:
""")

# Simulated calibration data
demo_data = {
    "50%": {"predictions": 20, "correct": 9, "expected": 10},
    "60%": {"predictions": 25, "correct": 14, "expected": 15},
    "70%": {"predictions": 30, "correct": 18, "expected": 21},
    "80%": {"predictions": 40, "correct": 28, "expected": 32},
    "90%": {"predictions": 50, "correct": 42, "expected": 45},
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Prediction Outcomes")

    total_predictions = sum(d["predictions"] for d in demo_data.values())
    total_correct = sum(d["correct"] for d in demo_data.values())

    st.markdown(f"""
    | Confidence | Predictions | Correct | Expected | Gap |
    |------------|-------------|---------|----------|-----|
    """)

    for conf, data in demo_data.items():
        actual_rate = data["correct"] / data["predictions"]
        expected_rate = int(conf.replace("%", "")) / 100
        gap = actual_rate - expected_rate
        gap_str = f"{gap:+.0%}" if gap != 0 else "—"
        color = "🟢" if abs(gap) < 0.05 else "🟡" if abs(gap) < 0.10 else "🔴"

        st.markdown(f"| {conf} | {data['predictions']} | {data['correct']} | {data['expected']} | {color} {gap_str} |")

with col2:
    st.markdown("### Analysis")

    # Calculate ECE
    total_error = 0
    for conf, data in demo_data.items():
        actual_rate = data["correct"] / data["predictions"]
        expected_rate = int(conf.replace("%", "")) / 100
        weight = data["predictions"] / total_predictions
        total_error += weight * abs(actual_rate - expected_rate)

    st.metric("Overall ECE", f"{total_error:.1%}")

    if total_error < 0.05:
        st.success("✅ Well calibrated! Confidence scores are reliable.")
    elif total_error < 0.10:
        st.warning("⚠️ Acceptable calibration, but room for improvement.")
    else:
        st.error("🔴 Poor calibration. Confidence scores are misleading!")

    st.markdown("""
    **What this means:**

    If ECE is 5%, then on average, our confidence scores are off by 5 percentage points.
    When we say "80% confident", we might actually be right 75-85% of the time.
    """)

st.markdown("---")

# Record Outcome
st.header("📝 Record Prediction Outcome")

st.markdown("""
Record the actual outcome of a prediction to improve calibration:
""")

with st.form("outcome_form"):
    col1, col2 = st.columns(2)

    with col1:
        prediction_id = st.text_input(
            "Prediction ID",
            placeholder="pred_20240115_123456_789",
            help="ID of the prediction to record outcome for"
        )

        actual_value = st.text_input(
            "Actual Outcome",
            placeholder="Motion was granted",
            help="What actually happened?"
        )

    with col2:
        was_correct = st.radio(
            "Was the prediction correct?",
            options=["Yes", "No"],
            horizontal=True
        )

        notes = st.text_area(
            "Notes",
            placeholder="Any additional context...",
            help="Optional notes about this outcome"
        )

    if st.form_submit_button("Record Outcome"):
        if prediction_id and actual_value:
            try:
                success = system.record_prediction_outcome(
                    prediction_id=prediction_id,
                    actual_value=actual_value,
                    was_correct=(was_correct == "Yes"),
                    notes=notes
                )
                if success:
                    st.success("✅ Outcome recorded! Calibration will update.")
                else:
                    st.error("Failed to record outcome")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please provide prediction ID and actual outcome")

st.markdown("---")

# Recalibrate
st.header("🔄 Run Recalibration")

st.markdown("""
Recalibration analyzes all recorded outcomes and computes a new calibration factor:
""")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🔄 Recalibrate Now", type="primary"):
        with st.spinner("Running recalibration..."):
            try:
                success = system.recalibrate()
                if success:
                    new_status = system.get_calibration_status()
                    st.success(f"✅ Recalibration complete!")
                    st.metric(
                        "New Calibration Factor",
                        f"{new_status.get('current_factor', 1.0):.3f}"
                    )
                else:
                    st.warning("Recalibration skipped - insufficient data")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.info("""
    **When to recalibrate:**
    - After recording many new outcomes
    - Weekly for active systems
    - When you notice prediction drift

    Recalibration adjusts the confidence scores so future predictions
    better match observed accuracy.
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style="background: #e8f5e9; padding: 1rem; border-radius: 8px; border-left: 4px solid #7ED321;">
<strong>🎯 The Bottom Line:</strong> Calibration is what makes probabilistic predictions
<em>trustworthy</em>. Without it, confidence scores are just numbers. With it, they're
actionable intelligence.
</div>
""", unsafe_allow_html=True)
