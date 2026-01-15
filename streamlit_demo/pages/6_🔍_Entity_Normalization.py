"""
Entity Normalization Demo - Clean messy court data
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from normalizers import JudgeNormalizer, LawyerNormalizer

st.set_page_config(page_title="Entity Normalization | EFC", page_icon="🔍", layout="wide")

st.title("🔍 Entity Normalization")
st.markdown("*Clean messy court data into standardized formats*")

st.markdown("""
Legal data comes from many sources with inconsistent formatting:
- CourtListener URLs: `/person/john-g-roberts-jr/`
- PACER filings: `ROBERTS, JOHN G. JR.`
- News articles: `Chief Justice Roberts`

This module normalizes them all to a consistent format.
""")

st.markdown("---")

# Judge Normalizer Section
st.header("👨‍⚖️ Judge Name Normalizer")

judge_normalizer = JudgeNormalizer()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Input")
    judge_input = st.text_area(
        "Enter judge names (one per line):",
        value="""https://www.courtlistener.com/person/john-g-roberts-jr/
william-h-alsup
GILSTRAP, RODNEY JR
Hon. Sonia Sotomayor
judge ketanji brown jackson""",
        height=200
    )

with col2:
    st.markdown("### Normalized Output")
    if judge_input:
        lines = [l.strip() for l in judge_input.strip().split("\n") if l.strip()]
        for line in lines:
            result = judge_normalizer.normalize(line)
            if result.normalized_name:
                st.success(f"✓ **{result.normalized_name}**")
                st.caption(f"Source: {result.source_type.value} | Confidence: {result.confidence:.0%}")
            else:
                st.error(f"✗ Could not parse: {line}")

# Show detailed breakdown
with st.expander("See detailed breakdown"):
    if judge_input:
        lines = [l.strip() for l in judge_input.strip().split("\n") if l.strip()]
        for line in lines:
            result = judge_normalizer.normalize(line)
            st.markdown(f"**Input:** `{line}`")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("First", result.first_name or "—")
            with col2:
                st.metric("Middle", result.middle_name or "—")
            with col3:
                st.metric("Last", result.last_name or "—")
            with col4:
                st.metric("Suffix", result.suffix or "—")
            st.markdown("---")

st.markdown("---")

# Lawyer Validator Section
st.header("⚖️ Lawyer Entity Validator")

st.markdown("""
Lawyer fields in court data often contain invalid entries:
- Geographic locations (data entry errors)
- Organizations (should be in a different field)
- Pro se indicators (not actual lawyers)

This validator filters them out.
""")

lawyer_normalizer = LawyerNormalizer()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Input")
    lawyer_input = st.text_area(
        "Enter lawyer names (one per line):",
        value="""John Smith
San Francisco
Morrison & Foerster LLP
Jane Doe, Esq.
Pro Se
New York, NY
Deputy Clerk
Robert O'Brien Jr.""",
        height=200,
        key="lawyer_input"
    )

with col2:
    st.markdown("### Validation Results")
    if lawyer_input:
        lines = [l.strip() for l in lawyer_input.strip().split("\n") if l.strip()]
        valid_count = 0
        invalid_count = 0

        for line in lines:
            result = lawyer_normalizer.validate(line)
            if result.is_valid:
                valid_count += 1
                st.success(f"✓ **{result.normalized_name}**")
            else:
                invalid_count += 1
                st.error(f"✗ {line} — *{result.rejection_reason.value}*")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Valid", valid_count, delta=None)
        with col2:
            st.metric("Invalid", invalid_count, delta=None)

st.markdown("---")

# Batch Processing Demo
st.header("📦 Batch Processing")

st.markdown("""
In production, you'd process thousands of records. Here's what that looks like:
""")

# Sample messy data
sample_data = {
    "judges": [
        "https://courtlistener.com/person/william-h-alsup/",
        "CHEN, EDWARD M",
        "Hon. Lucy H. Koh",
        "richard-seeborg",
    ],
    "lawyers": [
        "Michael Smith",
        "San Jose",
        "Lisa Johnson, Esq.",
        "Pro Se",
        "Wilson Sonsini LLP",
        "Sarah Chen",
    ]
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Raw Judge Data")
    for j in sample_data["judges"]:
        st.code(j)

    st.markdown("### After Normalization")
    results = judge_normalizer.batch_normalize(sample_data["judges"])
    for r in results:
        if r.normalized_name:
            st.success(r.normalized_name)

with col2:
    st.markdown("### Raw Lawyer Data")
    for l in sample_data["lawyers"]:
        st.code(l)

    st.markdown("### After Validation")
    valid_lawyers = lawyer_normalizer.filter_valid(sample_data["lawyers"])
    for l in valid_lawyers:
        st.success(l)

    invalid = [l for l in sample_data["lawyers"]
               if l not in [v.raw_input for v in lawyer_normalizer.batch_validate(sample_data["lawyers"]) if v.is_valid]]

st.markdown("---")

# Insight box
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white;">
<h3 style="margin-top: 0; color: white;">🎯 Why Normalization Matters</h3>
<p>Without normalization, the same judge might appear as 3 different entities in your database:</p>
<ul>
<li><code>william-h-alsup</code> (from CourtListener URL)</li>
<li><code>ALSUP, WILLIAM H.</code> (from PACER filing)</li>
<li><code>Hon. William Alsup</code> (from news article)</li>
</ul>
<p style="margin-bottom: 0;">This breaks pattern matching, concentration analysis, and drift detection. Normalization ensures they're all recognized as the same person: <strong>William H. Alsup</strong></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Navigate to Concentration Risk to see how normalized entities enable SPOF analysis →")
