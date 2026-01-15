"""
Pattern Drift Detection - Monitor changes in judicial behavior
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from drift import EmbeddingTracker, DriftDetector, DriftType
from drift.drift_detector import DriftSeverity

st.set_page_config(page_title="Pattern Drift | EFC", page_icon="📉", layout="wide")

st.title("📉 Pattern Drift Detection")
st.markdown("*Monitor changes in judicial behavior patterns over time*")

st.markdown("""
Judicial patterns aren't static. They evolve due to:
- **New precedents** changing legal standards
- **Judge transitions** (retirement, new appointments)
- **Policy changes** at the court level
- **Caseload shifts** affecting time allocation

Drift detection alerts you when patterns change significantly,
so you can recalibrate your confidence estimates.
""")

st.markdown("---")

# How Drift Detection Works
st.header("🔬 How Drift Detection Works")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### 1. Pattern Embeddings

    We convert judicial behavior into 64-dimensional vectors:

    | Dimensions | What They Capture |
    |------------|-------------------|
    | 0-15 | Rate metrics (grant/denial rates) |
    | 16-31 | Temporal patterns (timing, seasonality) |
    | 32-47 | Case type distribution |
    | 48-63 | Procedural preferences |

    These embeddings let us mathematically compare patterns.
    """)

with col2:
    st.markdown("""
    ### 2. Drift Measurement

    We use **cosine similarity** to compare current patterns against baselines:

    - **1.0** = Identical patterns
    - **0.95+** = Negligible drift
    - **0.85-0.95** = Minor drift
    - **0.70-0.85** = Moderate drift
    - **< 0.70** = Significant drift

    When similarity drops, we alert for investigation.
    """)

st.markdown("---")

# Interactive Drift Demo
st.header("🎮 Interactive Drift Simulation")

st.markdown("Adjust the current pattern metrics to see drift detection in action:")

tracker = EmbeddingTracker()
detector = DriftDetector()

# Baseline metrics (fixed)
baseline_metrics = {
    "grant_rate": 0.45,
    "denial_rate": 0.40,
    "partial_grant_rate": 0.15,
    "avg_days_to_decision": 120,
    "oral_argument_rate": 0.30,
    "case_types": {
        "patent": 0.60,
        "employment": 0.25,
        "other": 0.15
    }
}

# Set baseline
baseline_embedding = tracker.generate(
    entity_id="judge_example",
    pattern_type="summary_judgment",
    metrics=baseline_metrics,
    sample_count=500
)
detector.set_baseline(baseline_embedding)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Baseline Pattern (Historical)")
    st.markdown("*Based on 500 observations*")

    st.metric("Grant Rate", f"{baseline_metrics['grant_rate']:.0%}")
    st.metric("Denial Rate", f"{baseline_metrics['denial_rate']:.0%}")
    st.metric("Avg Days to Decision", baseline_metrics['avg_days_to_decision'])
    st.metric("Oral Argument Rate", f"{baseline_metrics['oral_argument_rate']:.0%}")

with col2:
    st.markdown("### Current Pattern (Adjust These)")

    current_grant = st.slider(
        "Grant Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        key="grant"
    )
    current_denial = st.slider(
        "Denial Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        key="denial"
    )
    current_days = st.slider(
        "Avg Days to Decision",
        min_value=30,
        max_value=300,
        value=120,
        step=10,
        key="days"
    )
    current_oral = st.slider(
        "Oral Argument Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        key="oral"
    )

# Build current metrics
current_metrics = {
    "grant_rate": current_grant,
    "denial_rate": current_denial,
    "partial_grant_rate": 1.0 - current_grant - current_denial,
    "avg_days_to_decision": current_days,
    "oral_argument_rate": current_oral,
    "case_types": baseline_metrics["case_types"]  # Keep same for simplicity
}

# Generate current embedding and detect drift
current_embedding = tracker.generate(
    entity_id="judge_example",
    pattern_type="summary_judgment",
    metrics=current_metrics,
    sample_count=50
)

drift_event = detector.detect_drift(current_embedding)

st.markdown("---")

# Drift Results
st.header("📊 Drift Analysis Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Baseline Similarity",
        f"{drift_event.baseline_similarity:.1%}",
        delta=f"{(drift_event.baseline_similarity - 1.0) * 100:.1f}%"
    )

with col2:
    # Color-coded severity
    severity_colors = {
        DriftSeverity.NEGLIGIBLE: "🟢",
        DriftSeverity.MINOR: "🟡",
        DriftSeverity.MODERATE: "🟠",
        DriftSeverity.SIGNIFICANT: "🔴",
        DriftSeverity.SEVERE: "⚫"
    }
    st.metric(
        "Drift Severity",
        f"{severity_colors[drift_event.severity]} {drift_event.severity.value.title()}"
    )

with col3:
    st.metric(
        "Confidence Impact",
        f"-{drift_event.confidence_impact:.0%}",
        help="Recommended reduction in confidence due to drift"
    )

# Status message
if drift_event.drift_type == DriftType.NONE:
    st.success("✅ Pattern is stable. No significant drift detected.")
elif drift_event.severity in [DriftSeverity.NEGLIGIBLE, DriftSeverity.MINOR]:
    st.info(f"ℹ️ {drift_event.drift_type.value.title()} drift detected, but within acceptable bounds.")
elif drift_event.severity == DriftSeverity.MODERATE:
    st.warning(f"⚠️ Moderate {drift_event.drift_type.value} drift. Monitor closely.")
else:
    st.error(f"🚨 {drift_event.severity.value.upper()} drift requires immediate attention!")

# Recommendation
st.markdown("### Recommendation")
st.markdown(f"""
<div style="background: {'#d4edda' if drift_event.severity == DriftSeverity.NEGLIGIBLE else '#fff3cd' if drift_event.severity in [DriftSeverity.MINOR, DriftSeverity.MODERATE] else '#f8d7da'};
    padding: 1rem; border-radius: 8px; border-left: 4px solid {'#28a745' if drift_event.severity == DriftSeverity.NEGLIGIBLE else '#ffc107' if drift_event.severity in [DriftSeverity.MINOR, DriftSeverity.MODERATE] else '#dc3545'};">
{drift_event.recommendation}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Drift Types Explained
st.header("📚 Understanding Drift Types")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Gradual Drift
    **What:** Slow evolution over many decisions

    **Causes:**
    - Changing legal standards
    - Evolving judicial philosophy
    - Caseload composition shifts

    **Response:** Monitor and recalibrate periodically
    """)

    st.markdown("""
    ### Sudden Drift
    **What:** Abrupt change in short period

    **Causes:**
    - Major precedent change
    - New procedural rules
    - Personnel change

    **Response:** Investigate immediately
    """)

with col2:
    st.markdown("""
    ### Seasonal Drift
    **What:** Cyclical patterns over time

    **Causes:**
    - End-of-term pressure
    - Holiday slowdowns
    - Fiscal year effects

    **Response:** Account for seasonality
    """)

    st.markdown("""
    ### Reverting Drift
    **What:** Temporary deviation returning to baseline

    **Causes:**
    - Unusual case batch
    - Temporary coverage
    - One-time policy exception

    **Response:** Wait and verify return
    """)

st.markdown("---")

# Real-world scenario
st.header("💼 Real-World Scenario")

st.markdown("""
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 12px; color: white;">
<h3 style="margin-top: 0; color: white;">Case Study: Judge Retirement</h3>

<p><strong>Situation:</strong> Your legal analytics system has been tracking Judge Martinez's summary judgment patterns for 5 years. She grants about 45% of motions with high consistency.</p>

<p><strong>Event:</strong> Judge Martinez retires. A new judge takes over her caseload.</p>

<p><strong>Without Drift Detection:</strong> Your system continues predicting 45% grant rate, but the new judge actually grants at 65%. Your predictions are systematically wrong for months.</p>

<p><strong>With Drift Detection:</strong></p>
<ol>
<li>After 10 cases, drift detection flags "sudden drift" (similarity dropped to 0.72)</li>
<li>System recommends recalibration with 30% confidence reduction</li>
<li>Human reviewer investigates, discovers judge change</li>
<li>Baseline is reset for the new judge</li>
</ol>

<p style="margin-bottom: 0;"><strong>Result:</strong> Pattern recalibrated within weeks, not months. Predictions accurate again.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Navigate to Jurisdictional Context to see court-specific guidance →")
