"""
Water in Sand - Deep dive into the core concept
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Water in Sand | EFC", page_icon="💧", layout="wide")

st.title("💧 The Water in Sand Metaphor")
st.markdown("*Understanding human-gated probabilistic intelligence*")

st.markdown("---")

# The Core Insight
st.header("🎯 The Core Insight")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### The Problem with LLMs

    Large Language Models produce output that is:

    - ✅ **Probabilistically reliable** - Often correct
    - ❌ **Not deterministically correct** - Sometimes wrong
    - ⚠️ **Confidently wrong** - No built-in uncertainty

    For **high-stakes decisions** (legal, medical, financial), this is dangerous.
    Traditional approaches try to make LLMs "more accurate" - but they can never
    reach 100% reliability.
    """)

with col2:
    st.markdown("""
    ### The Solution

    Instead of fighting the probabilistic nature, **embrace it**:

    - 💧 **LLM output is water** - It flows, it's abundant, it's useful
    - 🏖️ **Domain structure is sand** - Events, patterns, databases
    - 🚪 **Humans are gatekeepers** - They control where water flows

    The human doesn't create the water. **The human controls where it goes.**
    """)

st.markdown("---")

# Interactive Demo
st.header("🎮 Interactive: See the Gate in Action")

st.markdown("Adjust the confidence level to see how the gate responds:")

confidence = st.slider(
    "Prediction Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01,
    format="%d%%"
)

stakes = st.radio(
    "Stakes Level",
    options=["Low", "Medium", "High"],
    horizontal=True,
    index=1
)

# Adjust thresholds based on stakes
stakes_penalty = {"Low": 0, "Medium": 0.05, "High": 0.10}[stakes]
thresholds = {
    "auto_pass": 0.92 - stakes_penalty,
    "review": 0.70 - stakes_penalty,
    "block": 0.50 - stakes_penalty
}

# Determine decision
if confidence >= thresholds["auto_pass"]:
    decision = "AUTO PASS"
    color = "green"
    emoji = "✅"
    explanation = "High confidence! This flows directly to production."
elif confidence >= thresholds["review"]:
    decision = "REVIEW REQUIRED"
    color = "orange"
    emoji = "🔍"
    explanation = "Medium confidence. A human reviewer must approve this."
elif confidence >= thresholds["block"]:
    decision = "BLOCKED"
    color = "red"
    emoji = "🚫"
    explanation = "Low confidence. Requires explicit human override to proceed."
else:
    decision = "REJECTED"
    color = "darkred"
    emoji = "❌"
    explanation = "Very low confidence. This should not proceed at all."

# Display result
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 2rem;
        background: {'#e8f5e9' if decision == 'AUTO PASS' else '#fff3e0' if decision == 'REVIEW REQUIRED' else '#ffebee'};
        border-radius: 10px;
        border: 3px solid {color};
    ">
        <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
        <h2 style="color: {color}; margin: 0.5rem 0;">{decision}</h2>
        <p style="margin: 0;">Confidence: {confidence:.0%}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"### {explanation}")

    st.markdown("**Current Thresholds:**")
    st.markdown(f"""
    | Decision | Threshold |
    |----------|-----------|
    | Auto Pass | ≥ {thresholds['auto_pass']:.0%} |
    | Review Required | ≥ {thresholds['review']:.0%} |
    | Blocked | ≥ {thresholds['block']:.0%} |
    | Rejected | < {thresholds['block']:.0%} |
    """)

    st.info(f"*Stakes penalty applied: -{stakes_penalty:.0%} (for {stakes.lower()} stakes)*")

st.markdown("---")

# The Human Role
st.header("👤 The Human Role")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 🔓 Open Channels
    **Approve high-confidence outputs**

    When the system is confident AND the human agrees,
    the output flows to production.
    """)

with col2:
    st.markdown("""
    ### 🔒 Close Channels
    **Block risky outputs**

    When confidence is low or the output seems wrong,
    humans prevent bad outputs from escaping.
    """)

with col3:
    st.markdown("""
    ### 🎚️ Adjust Flow
    **Override weights**

    Domain experts can adjust pattern weights
    based on knowledge the system lacks.
    """)

with col4:
    st.markdown("""
    ### 🛤️ Build Paths
    **Identify new patterns**

    Humans spot patterns machines miss
    and feed them back into the system.
    """)

st.markdown("---")

# Why This Works
st.header("🧠 Why This Works")

st.markdown("""
<div style="background: #f0f7ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4A90E2;">

### The Competitive Advantage

The value isn't in making LLMs more accurate. It's in the **human oversight infrastructure**
that makes probabilistic output reliable for high-stakes use.

| Approach | Problem |
|----------|---------|
| "Better prompts" | Still probabilistic, still fails unpredictably |
| "Fine-tuning" | Expensive, domain-specific, still fails |
| "RAG" | Better context, still probabilistic output |
| **Human gates** | ✅ Probabilistic output + human control = reliable |

The system doesn't try to be perfect. It tries to **know when it's uncertain**
and route those cases to humans.

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Next Steps
st.markdown("""
### 🚀 Explore More

- **📊 Live System** - Add events and watch patterns emerge
- **🎯 Pattern Explorer** - See Bayesian weights in action
- **⚖️ Review Gate** - Try the human review interface
- **📈 Calibration** - See how confidence matches reality
""")
