"""
Water in Sand - Deep dive into the core concept
Enhanced with semantic entropy and AI auto-routing
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Water in Sand | EFC", page_icon="💧", layout="wide")

# Initialize session state for persistent toggle
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = True

st.title("💧 The Water in Sand Metaphor")
st.markdown("*Human-gated probabilistic intelligence for legal applications*")

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

# Operating Mode Toggle
st.header("🤖 Operating Mode")

mode_col1, mode_col2 = st.columns([1, 2])

with mode_col1:
    ai_mode = st.toggle(
        "AI Auto-Routing",
        value=st.session_state.ai_mode,
        key="ai_mode_toggle",
        help="When enabled, AI routes ~90% of decisions automatically based on calibrated thresholds"
    )
    # Update session state when toggled
    st.session_state.ai_mode = ai_mode

with mode_col2:
    if ai_mode:
        st.success("**AI MANAGED** - Humans monitor patterns and exceptions only")
        st.caption("System auto-routes high-confidence items. Humans review flagged items and monitor calibration drift.")
    else:
        st.warning("**HUMAN CONTROLLED** - All items require explicit approval")
        st.caption("Every item goes to human review queue regardless of confidence.")

st.markdown("---")

# Interactive Demo
st.header("🎮 Interactive: See the Gate in Action")

st.markdown("Adjust the confidence level to see how the gate responds:")

# Use 0-100 integer slider to avoid sprintf formatting issues
confidence_pct = st.slider(
    "Prediction Confidence",
    min_value=0,
    max_value=100,
    value=75,
    step=1,
    format="%d%%"
)

# Convert to 0.0-1.0 for internal use
confidence = confidence_pct / 100.0

col1, col2 = st.columns(2)

with col1:
    stakes = st.radio(
        "Stakes Level",
        options=["Low", "Medium", "High"],
        horizontal=True,
        index=1
    )

with col2:
    # Semantic entropy category (from uncertainty-analysis skill)
    uncertainty_type = st.selectbox(
        "Uncertainty Category",
        options=[
            "FACTUAL_CERTAIN",
            "FACTUAL_UNCERTAIN",
            "HEDGED_CLAIM",
            "INFERENTIAL_LEAP",
            "SCOPE_OVERREACH",
            "CONTESTED_DOMAIN"
        ],
        index=0,
        help="Semantic entropy category from uncertainty analysis"
    )

# Adjust thresholds based on stakes AND uncertainty type
stakes_penalty = {"Low": 0, "Medium": 0.05, "High": 0.10}[stakes]

# Additional penalty for high-uncertainty categories
uncertainty_penalty = {
    "FACTUAL_CERTAIN": 0,
    "FACTUAL_UNCERTAIN": 0.05,
    "HEDGED_CLAIM": 0.03,
    "INFERENTIAL_LEAP": 0.08,
    "SCOPE_OVERREACH": 0.06,
    "CONTESTED_DOMAIN": 0.10
}[uncertainty_type]

total_penalty = stakes_penalty + uncertainty_penalty

thresholds = {
    "auto_pass": max(0.50, 0.92 - total_penalty),
    "review": max(0.40, 0.70 - total_penalty),
    "block": max(0.30, 0.50 - total_penalty)
}

# Determine decision
if confidence >= thresholds["auto_pass"]:
    decision = "AUTO PASS"
    color = "green"
    emoji = "✅"
    explanation = "High confidence! This flows directly to production." if ai_mode else "High confidence. Ready for human approval."
    routing = "→ Production" if ai_mode else "→ Human Queue"
elif confidence >= thresholds["review"]:
    decision = "REVIEW REQUIRED"
    color = "orange"
    emoji = "🔍"
    explanation = "Medium confidence. A human reviewer must approve this."
    routing = "→ Human Queue"
elif confidence >= thresholds["block"]:
    decision = "BLOCKED"
    color = "red"
    emoji = "🚫"
    explanation = "Low confidence. Requires explicit human override to proceed."
    routing = "→ Blocked (needs override)"
else:
    decision = "REJECTED"
    color = "darkred"
    emoji = "❌"
    explanation = "Very low confidence. This should not proceed at all."
    routing = "→ Rejected"

# Display result
col1, col2 = st.columns([1, 2])

with col1:
    bg_color = '#e8f5e9' if decision == 'AUTO PASS' else '#fff3e0' if decision == 'REVIEW REQUIRED' else '#ffebee'
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 2rem;
        background: {bg_color};
        border-radius: 10px;
        border: 3px solid {color};
    ">
        <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
        <h2 style="color: {color}; margin: 0.5rem 0;">{decision}</h2>
        <p style="margin: 0; font-size: 1.5rem;"><strong>{confidence_pct}%</strong></p>
        <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #666;">{routing}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"### {explanation}")

    # Semantic entropy explanation
    if uncertainty_type != "FACTUAL_CERTAIN":
        st.info(f"""
        **Semantic Entropy Signal**: `{uncertainty_type}`

        Threshold adjustment: **-{uncertainty_penalty:.0%}** applied because this claim type
        shows high variance when sampled multiple times.
        """)

    st.markdown("**Current Thresholds:**")
    st.markdown(f"""
    | Decision | Threshold | Adjustment |
    |----------|-----------|------------|
    | Auto Pass | ≥ {thresholds['auto_pass']:.0%} | -{total_penalty:.0%} |
    | Review Required | ≥ {thresholds['review']:.0%} | -{total_penalty:.0%} |
    | Blocked | ≥ {thresholds['block']:.0%} | -{total_penalty:.0%} |
    | Rejected | < {thresholds['block']:.0%} | - |
    """)

    st.caption(f"*Stakes: -{stakes_penalty:.0%} ({stakes.lower()}) + Uncertainty: -{uncertainty_penalty:.0%} ({uncertainty_type})*")

st.markdown("---")

# Semantic Entropy Explanation
st.header("📊 Semantic Entropy & Uncertainty Categories")

st.markdown("""
<div style="background: #f5f5f5; padding: 1rem; border-radius: 8px; border-left: 4px solid #666;">

**From Kuhn et al. (Nature 2024)**: Traditional uncertainty measures token probability.
But "Paris" and "the capital of France" are semantically equivalent - raw probability misses this.

**Semantic entropy** clusters responses by meaning, then measures entropy across clusters:

```
H_semantic = -Σ P(cluster_i) × log P(cluster_i)
```

**High semantic entropy** = model generates semantically different answers = uncertain.

</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Low Entropy
    **FACTUAL_CERTAIN**

    Asked 100 times, answers cluster into one meaning:
    - "Paris" / "the French capital" → same
    - Entropy ≈ 0.2
    - **Trust the confidence score**
    """)

with col2:
    st.markdown("""
    ### Medium Entropy
    **HEDGED_CLAIM** / **SCOPE_OVERREACH**

    Answers show some variation:
    - "probably" / "likely" / "often"
    - Entropy ≈ 0.8
    - **Reduce threshold slightly**
    """)

with col3:
    st.markdown("""
    ### High Entropy
    **CONTESTED_DOMAIN** / **INFERENTIAL_LEAP**

    Answers scatter across meanings:
    - Different conclusions / interpretations
    - Entropy ≈ 1.5+
    - **Require human review**
    """)

st.markdown("---")

# The Human Role (updated for AI-managed mode)
st.header("👤 The Human Role" + (" (Monitoring Mode)" if ai_mode else " (Control Mode)"))

if ai_mode:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📈 Monitor Calibration
        **Watch for drift**

        Check that auto-passed items maintain expected accuracy.
        Flag when calibration degrades.
        """)

    with col2:
        st.markdown("""
        ### 🎯 Review Exceptions
        **Handle edge cases**

        Items flagged by semantic entropy analysis
        or unusual confidence patterns need attention.
        """)

    with col3:
        st.markdown("""
        ### 🛠️ Tune Thresholds
        **Adjust the system**

        Based on domain expertise, adjust thresholds
        for specific uncertainty categories.
        """)
else:
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
