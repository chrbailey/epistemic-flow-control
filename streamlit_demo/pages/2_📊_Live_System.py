"""
Live System - Add events and see patterns emerge
Enhanced with real data flow and AI/Human mode integration
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_system import EpistemicFlowControl, SystemConfig

st.set_page_config(page_title="Live System | EFC", page_icon="📊", layout="wide")

# Initialize session state
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = True

if "activity_log" not in st.session_state:
    st.session_state.activity_log = []


def log_activity(action: str, details: str, status: str = "info"):
    """Add entry to activity log."""
    st.session_state.activity_log.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "details": details,
        "status": status
    })
    # Keep only last 20 entries
    st.session_state.activity_log = st.session_state.activity_log[:20]


def get_system():
    """Get or create the system instance."""
    if "system" not in st.session_state:
        config = SystemConfig(db_dir="./demo_data", domain="judicial")
        st.session_state.system = EpistemicFlowControl(config)
        st.session_state.demo_loaded = False
        log_activity("SYSTEM", "Initialized EpistemicFlowControl system", "info")
    return st.session_state.system


st.title("📊 Live System - Data Flow")
st.markdown("*See predictions flow through gates in real-time*")

system = get_system()

# Mode Toggle (synced with other pages)
mode_col1, mode_col2 = st.columns([1, 3])

with mode_col1:
    ai_mode = st.toggle(
        "AI Auto-Routing",
        value=st.session_state.ai_mode,
        key="live_ai_mode_toggle"
    )
    st.session_state.ai_mode = ai_mode

with mode_col2:
    if ai_mode:
        st.success("🤖 **AI MANAGED** - High-confidence items auto-pass, only exceptions need review")
    else:
        st.warning("👤 **HUMAN CONTROLLED** - All items go to review queue")

st.markdown("---")

# How to Use This System
with st.expander("📖 **How to Run This System**", expanded=not st.session_state.get("demo_loaded", False)):
    st.markdown("""
    ### Data Flow Pipeline

    ```
    1. INGEST EVENT → 2. EXTRACT PATTERNS → 3. MAKE PREDICTION → 4. GATE DECISION → 5. REVIEW/AUTO-PASS
    ```

    **Step-by-Step:**

    1. **Load Demo Data** (or ingest your own events)
       - Demo includes judicial events from N.D. Cal
       - Events contain: what happened, who, when, where, source

    2. **Watch Patterns Emerge**
       - System extracts patterns from events (e.g., "Judge Rodriguez grants SJ in patent cases 73% of time")
       - Patterns accumulate Bayesian weights based on observations

    3. **Make Predictions**
       - Enter a prediction about future outcomes
       - System calculates confidence from matching patterns
       - Gate decides: AUTO_PASS, REVIEW, or BLOCKED

    4. **AI Mode vs Human Mode**
       - **AI Mode**: High-confidence (≥92%) predictions auto-pass to production
       - **Human Mode**: All predictions require explicit approval

    5. **Monitor & Calibrate**
       - Track which predictions were correct
       - System adjusts confidence based on accuracy
    """)

st.markdown("---")

# System Health Dashboard
st.header("🏥 System Health")

# Check if demo data is loaded
if not st.session_state.get("demo_loaded", False):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.warning("⚠️ **No data loaded.** Click 'Load Demo Data' to populate the system with sample judicial events.")
    with col2:
        if st.button("🚀 Load Demo Data", type="primary"):
            with st.spinner("Loading demo data..."):
                try:
                    from examples import load_all_examples
                    load_all_examples(system)
                    st.session_state.demo_loaded = True
                    log_activity("DATA", "Loaded demo judicial events and patterns", "success")
                    st.success("✅ Demo data loaded!")
                    st.rerun()
                except ImportError:
                    # Create minimal demo data if examples not available
                    st.session_state.demo_loaded = True
                    log_activity("DATA", "Initialized with empty database (examples module not found)", "warning")
                    st.info("📝 System ready - add your own events below")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

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
        color = "inverse" if pending > 0 else "off"
        st.metric("Pending Reviews", pending, delta=f"+{pending}" if pending > 0 else None, delta_color=color)

except Exception as e:
    st.info("System health metrics will appear after loading data")

st.markdown("---")

# Main workflow in tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Ingest Event", "🔮 Make Prediction", "📋 View Patterns", "📜 Activity Log"])

with tab1:
    st.subheader("Add New Event")
    st.markdown("Events are the ground truth that patterns emerge from.")

    with st.form("event_form"):
        col1, col2 = st.columns(2)

        with col1:
            what = st.text_area(
                "What happened?",
                placeholder="Judge Rodriguez granted summary judgment for defendant in patent case...",
                help="Describe the event"
            )

            who = st.text_input(
                "Who was involved?",
                placeholder="Judge Maria Rodriguez, Apple Inc, Samsung",
                help="Comma-separated list of parties"
            )

            when = st.date_input("When?", value=datetime.now())

        with col2:
            where = st.text_input(
                "Where? (Jurisdiction)",
                placeholder="N.D. Cal",
                value="N.D. Cal"
            )

            source = st.selectbox(
                "Source",
                options=["pacer", "ecf", "law360", "reuters_legal", "direct_observation"],
                help="Higher reliability sources = higher confidence"
            )

            why = st.text_area(
                "Why? (reasoning)",
                placeholder="Plaintiff failed to establish genuine issues of material fact...",
            )

        auto_extract = st.checkbox("Auto-extract patterns from this event", value=True)

        submitted = st.form_submit_button("🚀 Ingest Event", type="primary")

        if submitted and what:
            with st.spinner("Ingesting event..."):
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
                        log_activity("EVENT", f"Ingested: {what[:40]}...", "success")
                        st.success(f"✅ Event ingested! ID: `{result['event_id']}`")
                        st.info(f"Source reliability: **{result.get('source_reliability', 'unknown')}** → Verification: **{result['verification_status']}**")

                        if result.get("patterns_extracted"):
                            st.subheader("🎯 Patterns Extracted")
                            for pattern in result["patterns_extracted"]:
                                log_activity("PATTERN", f"Extracted: {pattern['description'][:30]}...", "info")
                                st.markdown(f"- **{pattern['description']}** (confidence: {pattern['confidence']:.0%})")
                    else:
                        st.error(f"Failed: {result.get('message')}")

                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.subheader("Create Prediction")
    st.markdown("Predictions flow through the gate based on confidence and mode.")

    # Show current mode prominently
    if ai_mode:
        st.info("🤖 **AI Mode**: Predictions ≥92% confidence will **auto-pass**. Others go to review.")
    else:
        st.warning("👤 **Human Mode**: ALL predictions will go to **review queue** regardless of confidence.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pred_type = st.selectbox(
                "Prediction Type",
                options=["motion_outcome", "summary_judgment", "case_duration", "appeal_likelihood"],
            )

            pred_value = st.text_input(
                "Your Prediction",
                placeholder="Motion to dismiss will be granted",
            )

            stakes = st.select_slider(
                "Stakes Level",
                options=["low", "medium", "high"],
                value="medium",
                help="Higher stakes = stricter thresholds"
            )

        with col2:
            st.markdown("**Context (affects pattern matching):**")
            case_type = st.selectbox("Case Type", ["patent", "securities", "antitrust", "employment", "contract"])
            judge = st.text_input("Judge", placeholder="Judge Maria Rodriguez")

            context = {
                "case_type": case_type,
                "judge": judge,
                "prediction_type": pred_type
            }

        predict_submitted = st.form_submit_button("🎯 Submit Prediction", type="primary")

        if predict_submitted and pred_value:
            try:
                result = system.make_prediction(
                    prediction_type=pred_type,
                    predicted_value=pred_value,
                    context=context,
                    source_patterns=[],
                    stakes=stakes
                )

                confidence = result.get('calibrated_confidence', result.get('raw_confidence', 0.5))
                gate_decision = result.get('gate_decision', 'review')

                # Apply mode logic
                if not ai_mode:
                    gate_decision = "review"  # Force review in human mode
                    result['gate_reasoning'] = "Human mode: all items require review"

                log_activity("PREDICTION", f"{pred_value[:30]}... → {gate_decision.upper()}",
                           "success" if gate_decision == "auto_pass" else "warning")

                st.success(f"✅ Prediction created! ID: `{result['prediction_id']}`")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Raw Confidence", f"{result.get('raw_confidence', 0.5):.0%}")

                with col2:
                    st.metric("Calibrated", f"{confidence:.0%}")

                with col3:
                    if gate_decision == "auto_pass":
                        st.metric("Gate Decision", "✅ AUTO PASS")
                    elif gate_decision == "review":
                        st.metric("Gate Decision", "🔍 REVIEW")
                    else:
                        st.metric("Gate Decision", "🚫 BLOCKED")

                # Show routing
                st.markdown("---")
                if gate_decision == "auto_pass" and ai_mode:
                    st.success(f"**→ ROUTED TO PRODUCTION** (confidence {confidence:.0%} ≥ threshold)")
                else:
                    st.warning(f"**→ SENT TO REVIEW QUEUE** - Reason: {result.get('gate_reasoning', 'Below threshold or human mode active')}")

            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.subheader("Pattern Database")
    st.markdown("Patterns emerge from events and accumulate Bayesian weights.")

    try:
        demo_subjects = ["Judge Maria Rodriguez", "Judge William Chen", "Judge Sofia Martinez", "patent", "summary_judgment"]

        found_patterns = False
        for subject in demo_subjects:
            patterns = system.get_patterns_for_subject(subject, min_confidence=0.0)

            if patterns:
                found_patterns = True
                with st.expander(f"📊 {subject} ({len(patterns)} patterns)"):
                    for p in patterns[:5]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{p['description'][:60]}...**")
                            st.caption(f"Observations: {p['total_observations']} | Last seen: {p['last_observed'][:10]}")
                        with col2:
                            st.metric("Weight", f"{p['weight']:.0%}")
                        st.progress(p['weight'])

        if not found_patterns:
            st.info("No patterns yet. Ingest events to see patterns emerge.")

    except Exception as e:
        st.info("Load demo data or ingest events to see patterns")

with tab4:
    st.subheader("Activity Log")
    st.markdown("Real-time view of system operations.")

    if st.session_state.activity_log:
        for entry in st.session_state.activity_log:
            status_emoji = {"success": "✅", "warning": "⚠️", "error": "❌", "info": "ℹ️"}.get(entry["status"], "ℹ️")
            st.markdown(f"`{entry['time']}` {status_emoji} **{entry['action']}**: {entry['details']}")
    else:
        st.info("Activity will appear here as you use the system.")

    if st.button("Clear Log"):
        st.session_state.activity_log = []
        st.rerun()

st.markdown("---")

# Quick actions
st.markdown("### 🚀 Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 View Calibration →"):
        st.switch_page("pages/5_📈_Calibration.py")

with col2:
    if st.button("⚖️ Review Queue →"):
        st.switch_page("pages/4_⚖️_Review_Gate.py")

with col3:
    if st.button("🎯 Pattern Explorer →"):
        st.switch_page("pages/3_🎯_Pattern_Explorer.py")
