"""
Pattern Explorer - Browse patterns with Bayesian charts
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_system import EpistemicFlowControl, SystemConfig

st.set_page_config(page_title="Pattern Explorer | EFC", page_icon="🎯", layout="wide")


def get_system():
    if "system" not in st.session_state:
        config = SystemConfig(db_dir="./demo_data", domain="judicial")
        st.session_state.system = EpistemicFlowControl(config)
    return st.session_state.system


st.title("🎯 Pattern Explorer")
st.markdown("*Explore patterns with Bayesian weight visualization*")

system = get_system()

# Import visualization components
try:
    from streamlit_demo.components.charts import (
        plot_bayesian_update,
        plot_pattern_evolution,
        plot_source_reliability_spectrum
    )
    charts_available = True
except ImportError:
    charts_available = False
    st.warning("Plotly charts not available. Install with: pip install plotly")

st.markdown("---")

# Judge Profiles Section
st.header("👨‍⚖️ Judge Profiles")

st.markdown("""
Explore how different judges demonstrate different aspects of the system:
""")

# Import judge data
try:
    from examples.judges import JUDGE_PROFILES, get_confidence_comparison, calculate_wilson_lower

    tabs = st.tabs([j.name for j in JUDGE_PROFILES])

    for i, tab in enumerate(tabs):
        judge = JUDGE_PROFILES[i]

        with tab:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"Story: {judge.story_arc}")
                st.markdown(f"**Court:** {judge.court}")
                st.markdown(f"**Specialty:** {', '.join(judge.specialty_areas)}")

                st.markdown("---")

                st.markdown(f"### 💡 What This Demonstrates")
                st.info(judge.demonstrates)

            with col2:
                # Key metrics
                st.metric("Total Cases", judge.total_cases)
                st.metric("Grant Rate", f"{judge.summary_judgment_grant_rate:.0%}")

                # Wilson score
                successes = int(judge.summary_judgment_grant_rate * judge.total_cases)
                wilson_lower = calculate_wilson_lower(successes, judge.total_cases)
                st.metric(
                    "Wilson Lower Bound",
                    f"{wilson_lower:.1%}",
                    delta=f"{wilson_lower - judge.summary_judgment_grant_rate:.1%}",
                    help="Conservative estimate accounting for sample size"
                )

            # Historical evolution chart
            if judge.historical_rates and charts_available:
                st.markdown("### 📈 Pattern Evolution")
                fig = plot_pattern_evolution(judge.name, judge.historical_rates)
                st.plotly_chart(fig, use_container_width=True)
            elif judge.historical_rates:
                st.markdown("### 📈 Historical Rates")
                for h in judge.historical_rates:
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.write(h['period'])
                    with col2:
                        st.progress(h['rate'])
                    with col3:
                        st.caption(h.get('note', ''))

except ImportError:
    st.info("Load demo data to see judge profiles. Run the app from the main page first.")

st.markdown("---")

# Wilson Score Interactive Demo
st.header("📊 Wilson Score Interactive Demo")

st.markdown("""
The Wilson Score Interval gives us a **conservative confidence estimate** that properly
handles small sample sizes. Try adjusting the numbers:
""")

col1, col2, col3 = st.columns(3)

with col1:
    successes = st.number_input("Successes (grants)", min_value=0, max_value=1000, value=4)

with col2:
    total = st.number_input("Total Cases", min_value=1, max_value=1000, value=8)

with col3:
    conf_level = st.selectbox(
        "Confidence Level",
        options=["90%", "95%", "99%"],
        index=1
    )
    z_values = {"90%": 1.645, "95%": 1.96, "99%": 2.576}
    z_score = (conf_level, z_values[conf_level])

if total > 0 and successes <= total:
    import math

    raw_rate = successes / total
    z = z_score[1]

    # Wilson calculation
    p = successes / total
    denominator = 1 + z*z/total
    center = p + z*z/(2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)

    wilson_lower = max(0.0, (center - spread) / denominator)
    wilson_upper = min(1.0, (center + spread) / denominator)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Raw Rate", f"{raw_rate:.1%}")

    with col2:
        st.metric(
            "Wilson Lower",
            f"{wilson_lower:.1%}",
            delta=f"{wilson_lower - raw_rate:.1%}"
        )

    with col3:
        st.metric(
            "Wilson Upper",
            f"{wilson_upper:.1%}",
            delta=f"+{wilson_upper - raw_rate:.1%}"
        )

    with col4:
        st.metric(
            "Interval Width",
            f"{(wilson_upper - wilson_lower):.1%}"
        )

    # Visual representation
    st.markdown("### Confidence Interval Visualization")

    # Create a simple bar chart representation
    st.markdown(f"""
    ```
    0%                                                        100%
    |{'─' * int(wilson_lower * 50)}[{'█' * int((wilson_upper - wilson_lower) * 50)}]{'─' * int((1 - wilson_upper) * 50)}|
                          {'^' if wilson_lower < raw_rate < wilson_upper else ''}
                          Raw Rate: {raw_rate:.0%}
    ```
    """)

    st.markdown(f"""
    **Interpretation:**
    - With **{total} observations**, we're {z_score[0]} confident the true rate is between
      **{wilson_lower:.1%}** and **{wilson_upper:.1%}**
    - The raw rate ({raw_rate:.0%}) is {'likely' if wilson_upper - wilson_lower < 0.3 else 'possibly'} misleading
      due to {'small' if total < 30 else 'moderate'} sample size
    """)

else:
    st.error("Successes cannot exceed total cases")

st.markdown("---")

# Source Reliability
st.header("📡 Source Reliability Spectrum")

try:
    from examples.sources import DEMO_SOURCES, get_reliability_spectrum

    spectrum = get_reliability_spectrum()

    if charts_available:
        sources_data = [
            {"name": s.name, "reliability": s.base_reliability, "source_type": s.source_type}
            for s in DEMO_SOURCES
        ]
        fig = plot_source_reliability_spectrum(sources_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        for source_type, sources in spectrum.items():
            st.markdown(f"### {source_type.title()}")
            for s in sources:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(s.base_reliability)
                with col2:
                    st.write(f"{s.name}: {s.base_reliability:.0%}")

    st.markdown("""
    <div style="background: #f0f7ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #4A90E2;">
    <strong>Key Insight:</strong> Source reliability directly affects verification status.
    PACER (0.99) events are auto-verified; Twitter (0.40) events require human review.
    </div>
    """, unsafe_allow_html=True)

except ImportError:
    st.info("Load demo data to see source reliability spectrum")

st.markdown("---")

# Confidence Comparison
st.header("🔬 Confidence Comparison: Why Sample Size Matters")

try:
    from examples.judges import get_confidence_comparison

    comparison = get_confidence_comparison()

    st.markdown("""
    Compare Wilson score lower bounds across judges with different sample sizes:
    """)

    cols = st.columns(len(comparison))

    for i, (judge_id, data) in enumerate(comparison.items()):
        with cols[i]:
            st.markdown(f"**{data['name'].split()[-1]}**")
            st.metric("Cases", data['total_cases'])
            st.metric("Raw Rate", f"{data['raw_rate']:.0%}")
            st.metric(
                "Wilson Lower",
                f"{data['wilson_lower']:.1%}",
                help="Conservative estimate"
            )

            # Can auto-pass?
            if data['can_auto_pass']:
                st.success("✅ Can AUTO_PASS")
            else:
                st.warning("🔍 Needs REVIEW")

except ImportError:
    st.info("Load demo data for confidence comparison")

st.markdown("---")
st.caption("Navigate to Review Gate to see human oversight in action →")
