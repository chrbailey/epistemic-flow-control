"""
Live System - Add events and see patterns emerge
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_system import EpistemicFlowControl, SystemConfig

st.set_page_config(page_title="Live System | EFC", page_icon="📊", layout="wide")


def get_system():
    """Get or create the system instance."""
    if "system" not in st.session_state:
        config = SystemConfig(db_dir="./demo_data", domain="judicial")
        st.session_state.system = EpistemicFlowControl(config)
        st.session_state.demo_loaded = False
    return st.session_state.system


st.title("📊 Live System Demo")
st.markdown("*Add events and watch patterns emerge in real-time*")

system = get_system()

# Check if demo data is loaded
if not st.session_state.get("demo_loaded", False):
    st.warning("⚠️ Demo data not loaded. Some features may be limited.")
    if st.button("Load Demo Data"):
        with st.spinner("Loading..."):
            try:
                from examples import load_all_examples
                load_all_examples(system)
                st.session_state.demo_loaded = True
                st.success("Demo data loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

# System Health Dashboard
st.header("🏥 System Health")

try:
    health = system.get_system_health()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        events = health.get("event_store", {}).get("total_events", 0)
        st.metric("Events", events)

    with col2:
        patterns = health.get("pattern_database", {}).get("total_patterns", 0)
        st.metric("Patterns", patterns)

    with col3:
        predictions = health.get("calibration", {}).get("total_predictions", 0)
        st.metric("Predictions", predictions)

    with col4:
        reviews = health.get("review_gate", {}).get("total_reviews", 0)
        st.metric("Reviews", reviews)

    with col5:
        pending = health.get("review_gate", {}).get("pending_reviews", 0)
        st.metric("Pending Reviews", pending, delta=None if pending == 0 else pending)

except Exception as e:
    st.error(f"Could not load system health: {e}")

st.markdown("---")

# Event Ingestion
st.header("📝 Ingest New Event")

with st.form("event_form"):
    col1, col2 = st.columns(2)

    with col1:
        what = st.text_area(
            "What happened?",
            placeholder="Judge granted summary judgment for defendant...",
            help="Describe the event that occurred"
        )

        who = st.text_input(
            "Who was involved?",
            placeholder="Judge Smith, Acme Corp, Beta Inc",
            help="Comma-separated list of parties"
        )

        when = st.date_input(
            "When?",
            value=datetime.now(),
            help="Date of the event"
        )

    with col2:
        where = st.text_input(
            "Where?",
            placeholder="N.D. Cal",
            help="Jurisdiction or location"
        )

        source = st.selectbox(
            "Source",
            options=["pacer", "ecf", "law360", "reuters_legal", "techcrunch"],
            help="Information source"
        )

        why = st.text_area(
            "Why? (reasoning)",
            placeholder="Plaintiff failed to establish genuine issues of material fact...",
            help="Optional: reasoning or context"
        )

    auto_extract = st.checkbox("Auto-extract patterns", value=True)

    submitted = st.form_submit_button("🚀 Ingest Event", type="primary")

    if submitted and what:
        with st.spinner("Ingesting event and extracting patterns..."):
            try:
                result = system.ingest_event(
                    what=what,
                    who=[w.strip() for w in who.split(",")] if who else ["Unknown"],
                    when=datetime.combine(when, datetime.min.time()),
                    where=where or "Unknown",
                    source_id=source,
                    raw_text=what,
                    why=why if why else None,
                    auto_extract_patterns=auto_extract
                )

                if result.get("success"):
                    st.success(f"✅ Event ingested! ID: `{result['event_id']}`")

                    # Show verification status
                    st.info(f"Verification: **{result['verification_status']}** (based on source reliability)")

                    # Show extracted patterns
                    if result.get("patterns_extracted"):
                        st.subheader("🎯 Patterns Extracted")

                        for pattern in result["patterns_extracted"]:
                            with st.expander(f"📌 {pattern['description'][:50]}... ({pattern['confidence']:.0%} confidence)"):
                                st.json(pattern)

                                if pattern.get('needs_validation'):
                                    st.warning("⚠️ This pattern needs human validation")
                    else:
                        st.info("No patterns extracted (rule-based extraction when LLM unavailable)")

                    # Check if needs review
                    if result.get("needs_human_review"):
                        st.warning("🔍 This event has been flagged for human review due to low source confidence")

                else:
                    st.error(f"❌ Failed: {result.get('message')}")

            except Exception as e:
                st.error(f"❌ Error: {e}")

st.markdown("---")

# Make Prediction
st.header("🔮 Make a Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pred_type = st.selectbox(
            "Prediction Type",
            options=["summary_judgment", "motion_outcome", "case_duration"],
            help="What are you predicting?"
        )

        pred_value = st.text_input(
            "Predicted Value",
            placeholder="Motion will be granted",
            help="Your prediction"
        )

    with col2:
        stakes = st.select_slider(
            "Stakes Level",
            options=["low", "medium", "high"],
            value="medium",
            help="How important is this decision?"
        )

        context_str = st.text_area(
            "Context (JSON)",
            value='{"case_type": "patent", "motion": "summary_judgment"}',
            help="Additional context as JSON"
        )

    predict_submitted = st.form_submit_button("🎯 Make Prediction", type="primary")

    if predict_submitted and pred_value:
        try:
            import json
            context = json.loads(context_str) if context_str else {}

            result = system.make_prediction(
                prediction_type=pred_type,
                predicted_value=pred_value,
                context=context,
                source_patterns=[],  # Would normally link to patterns
                stakes=stakes
            )

            st.success(f"✅ Prediction created! ID: `{result['prediction_id']}`")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Raw Confidence", f"{result['raw_confidence']:.1%}")

            with col2:
                st.metric("Calibrated", f"{result['calibrated_confidence']:.1%}")

            with col3:
                decision = result['gate_decision']
                color = "🟢" if decision == "auto_pass" else "🟡" if decision == "review" else "🔴"
                st.metric("Gate Decision", f"{color} {decision.upper()}")

            if result.get("needs_human_review"):
                st.warning(f"🔍 **Review Required**: {result.get('gate_reasoning', 'Confidence below threshold')}")

        except json.JSONDecodeError:
            st.error("Invalid JSON in context field")
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")

# View Patterns
st.header("📋 Recent Patterns")

try:
    # Get patterns for demo subjects
    demo_subjects = ["Judge Maria Rodriguez", "Judge William Chen", "Judge Sofia Martinez"]

    for subject in demo_subjects[:2]:  # Show first two
        patterns = system.get_patterns_for_subject(subject, min_confidence=0.0)

        if patterns:
            with st.expander(f"📊 {subject} ({len(patterns)} patterns)"):
                for p in patterns[:5]:  # Show first 5
                    st.markdown(f"""
                    **{p['description'][:60]}...**
                    - Weight: {p['weight']:.1%} (raw: {p['raw_weight']:.1%})
                    - Observations: {p['total_observations']}
                    - Last seen: {p['last_observed'][:10]}
                    """)
                    st.progress(p['weight'])

except Exception as e:
    st.info(f"Load demo data to see pattern examples")

st.markdown("---")
st.markdown("*Navigate to Pattern Explorer for detailed pattern analysis*")
