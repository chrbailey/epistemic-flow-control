"""
Epistemic Flow Control - Main Demo Application

Run with: streamlit run streamlit_demo/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_system import EpistemicFlowControl, SystemConfig


def init_session_state():
    """Initialize session state with system instance."""
    if "system" not in st.session_state:
        config = SystemConfig(db_dir="./demo_data", domain="judicial")
        st.session_state.system = EpistemicFlowControl(config)
        st.session_state.demo_loaded = False


def main():
    st.set_page_config(
        page_title="Epistemic Flow Control",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for beautiful styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(90deg, #4A90E2, #7ED321);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 1.3rem;
            color: #666;
            margin-top: 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
        }
        .insight-box {
            background: #f0f7ff;
            border-left: 4px solid #4A90E2;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .water-flow {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%);
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    init_session_state()

    # Main landing page content
    st.markdown('<p class="main-header">💧 Epistemic Flow Control</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Human-gated probabilistic intelligence for high-stakes domains</p>', unsafe_allow_html=True)

    st.markdown("---")

    # The Water in Sand Metaphor - Visual Explanation
    st.markdown("""
    <div class="water-flow">
        <h2>The Water in Sand Metaphor</h2>
        <p style="font-size: 1.1rem;">
        LLMs generate probabilistic output like <strong>water flowing</strong>.<br>
        Humans don't create the water — they control where it flows through <strong>gates</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Flow diagram using columns
    col1, col2, col3, col4, col5 = st.columns([1, 0.5, 1, 0.5, 1])

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #e3f2fd; border-radius: 10px;">
            <h3>💧 LLM Output</h3>
            <p><em>Probabilistic<br>"Water"</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h1 style='text-align: center; padding-top: 2rem;'>→</h1>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #fff3e0; border-radius: 10px;">
            <h3>🚪 Human Gate</h3>
            <p><em>Review &<br>Approval</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("<h1 style='text-align: center; padding-top: 2rem;'>→</h1>", unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #e8f5e9; border-radius: 10px;">
            <h3>✅ Production</h3>
            <p><em>Reliable<br>Output</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Features
    st.header("🎯 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>📊 Bayesian Weights</h4>
            <p>Pattern confidence grows with evidence using proper statistical updating. Small samples get appropriately wide confidence intervals.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-box">
            <h4>⏳ Temporal Decay</h4>
            <p>Old patterns fade without fresh confirming evidence. The system recognizes when historical data becomes stale.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="insight-box">
            <h4>🎚️ Calibrated Confidence</h4>
            <p>Confidence scores match actual accuracy. When the system says 80%, it's right 80% of the time.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Stats
    st.header("📈 System Status")

    if st.session_state.demo_loaded:
        health = st.session_state.system.get_system_health()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Events", health.get("event_store", {}).get("total_events", 0))
        with col2:
            st.metric("Patterns", health.get("pattern_database", {}).get("total_patterns", 0))
        with col3:
            st.metric("Predictions", health.get("calibration", {}).get("total_predictions", 0))
        with col4:
            st.metric("Reviews", health.get("review_gate", {}).get("total_reviews", 0))
    else:
        st.info("👆 Load the demo data to see live statistics!")

        if st.button("🚀 Load Demo Data", type="primary"):
            with st.spinner("Loading judicial domain examples..."):
                try:
                    from examples import load_all_examples
                    stats = load_all_examples(st.session_state.system)
                    st.session_state.demo_loaded = True
                    st.success(f"Loaded {stats['events_loaded']} events and extracted {stats['patterns_extracted']} patterns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading demo data: {e}")

    # Navigation hint
    st.markdown("---")
    st.markdown("""
    ### 🧭 Explore the Demo

    Use the sidebar to navigate between pages:

    - **💧 Water in Sand** - Deep dive into the core concept
    - **📊 Live System** - Add events and see patterns emerge
    - **🎯 Pattern Explorer** - Browse patterns with Bayesian charts
    - **⚖️ Review Gate** - Human review interface
    - **📈 Calibration** - Track prediction accuracy
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using <a href="https://github.com/chrbailey/epistemic-flow-control">Epistemic Flow Control</a></p>
        <p>Making LLMs reliable for high-stakes decisions</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
