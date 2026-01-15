<p align="center">
  <h1 align="center">💧 Epistemic Flow Control</h1>
  <p align="center">
    <strong>Human-gated probabilistic intelligence for high-stakes domains</strong>
  </p>
  <p align="center">
    <a href="https://github.com/chrbailey/epistemic-flow-control/actions"><img src="https://github.com/chrbailey/epistemic-flow-control/workflows/CI/badge.svg" alt="CI Status"></a>
    <a href="https://github.com/chrbailey/epistemic-flow-control/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
    <a href="https://github.com/chrbailey/epistemic-flow-control/stargazers"><img src="https://img.shields.io/github/stars/chrbailey/epistemic-flow-control?style=social" alt="Stars"></a>
  </p>
</p>

<p align="center">
  <em>Make LLM outputs reliable for decisions that actually matter.</em>
</p>

---

## 🌊 The Problem

LLMs are **probabilistically reliable** but not **deterministically correct**. For casual use, that's fine. For high-stakes decisions—legal, medical, financial—it's dangerous.

Traditional approaches try to make LLMs "more accurate." But they can never reach 100%. **We need a different approach.**

## 💡 The Solution: Water in Sand

```
LLM Output (Water) → Human Gates (Channels) → Production (Destination)
```

- **💧 LLMs produce "water"** — Probabilistic output that flows abundantly
- **🏖️ Domain structure is "sand"** — Events, patterns, databases that shape the flow
- **🚪 Humans control the gates** — Opening, closing, and adjusting channels

**The human doesn't create the water. The human controls where it flows.**

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **📊 Bayesian Pattern Weights** | Confidence grows with evidence using proper statistical updating |
| **⏳ Temporal Decay** | Old patterns fade without fresh confirming evidence |
| **🎚️ Calibrated Confidence** | When we say 80%, we're right 80% of the time |
| **🚪 Human Review Gates** | High-stakes decisions require human approval |
| **📈 Outcome Learning** | Every outcome improves future predictions |
| **🔬 Wilson Score Intervals** | Proper uncertainty for small samples |

### 🆕 New Features (v2.0)

| Feature | Description |
|---------|-------------|
| **🔍 Entity Normalization** | Clean messy court data (judge names from URLs, lawyer validation) |
| **⚠️ SPOF Detection** | Identify concentration risk using Herfindahl-Hirschman Index |
| **📉 Pattern Drift Detection** | Monitor changes in judicial behavior with 64-dimensional embeddings |
| **⚖️ Jurisdictional Context** | Court-specific and judge-specific guidance (N.D. Cal, Judge Alsup) |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chrbailey/epistemic-flow-control.git
cd epistemic-flow-control

# Install core (no dependencies!)
pip install -e .

# Or with all features
pip install -e ".[all]"
```

### Try the Interactive Demo

```bash
# Install demo dependencies
pip install -e ".[demo]"

# Run the Streamlit demo
streamlit run streamlit_demo/app.py
```

### Basic Usage

```python
from unified_system import EpistemicFlowControl, SystemConfig
from datetime import datetime

# Initialize
config = SystemConfig(db_dir="./data", domain="judicial")
system = EpistemicFlowControl(config)

# Register an information source
system.register_source(
    source_id="pacer",
    name="PACER",
    source_type="official",
    reliability=0.99
)

# Ingest an event (ground truth)
result = system.ingest_event(
    what="Judge granted summary judgment",
    who=["Judge Smith", "Acme Corp", "Beta Inc"],
    when=datetime.now(),
    where="N.D. Cal",
    source_id="pacer",
    raw_text="Order granting motion for summary judgment..."
)

# Patterns are automatically extracted
print(f"Extracted {len(result['patterns_extracted'])} patterns")

# Make a prediction
prediction = system.make_prediction(
    prediction_type="ruling",
    predicted_value="Motion will be granted",
    context={"case_type": "patent"},
    source_patterns=["pat_001"],
    stakes="high"
)

# Check the gate decision
print(f"Gate: {prediction['gate_decision']}")  # "review" for high stakes
print(f"Confidence: {prediction['calibrated_confidence']:.1%}")

# High-stakes items need human review
if prediction['needs_human_review']:
    items = system.get_items_needing_review()
    # Human reviews and approves...
    system.submit_human_review(
        item_id=prediction['prediction_id'],
        reviewer_id="expert_001",
        decision="approve",
        notes="Consistent with recent pattern"
    )
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EPISTEMIC FLOW CONTROL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Events (Ground Truth)                                         │
│      ↓                                                          │
│   Pattern Extraction (LLM) ←─── Human Validation                │
│      ↓                                                          │
│   Pattern Database (Bayesian) ←─── Human Override               │
│      ↓                                                          │
│   Predictions (Calibrated) ←─── Calibration Engine              │
│      ↓                                                          │
│   Review Gate (Thresholds) ←─── Human Review                    │
│      ↓                                                          │
│   Production Output                                             │
│      ↓                                                          │
│   Outcome Recording ───────────→ Training Data                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Statistical Foundation

This isn't just another LLM wrapper. It's built on solid statistical principles:

- **[Wilson Score Intervals](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)** — Conservative confidence bounds that handle small samples correctly. 3 successes out of 5? That's not 60% confidence—Wilson lower bound says ~23%.

- **[Bayesian Updating](https://en.wikipedia.org/wiki/Bayesian_inference)** — Prior beliefs + observations = posterior beliefs. Patterns strengthen with evidence.

- **[Expected Calibration Error](https://arxiv.org/abs/1706.04599)** — The standard metric for prediction calibration. We measure and minimize it.

- **Temporal Decay** — Patterns become stale. A judge's behavior 5 years ago may not predict today. Exponential decay with configurable half-life.

## 📁 Project Structure

```
epistemic-flow-control/
├── core/
│   ├── event_store.py       # Ground truth storage
│   ├── pattern_extractor.py # LLM pattern extraction
│   └── pattern_database.py  # Bayesian weights
├── gates/
│   └── review_gate.py       # Human review flow control
├── validation/
│   └── calibration_engine.py # Accuracy tracking
├── training/
│   └── data_generator.py    # Training data collection
├── normalizers/              # 🆕 Entity normalization
│   ├── judge_normalizer.py  # Clean judge names from URLs
│   └── lawyer_normalizer.py # Validate lawyer entities
├── concentration/            # 🆕 SPOF risk detection
│   ├── hhi_calculator.py    # Herfindahl-Hirschman Index
│   └── spof_detector.py     # Single Point of Failure analysis
├── drift/                    # 🆕 Pattern drift detection
│   ├── embedding_tracker.py # 64-dimensional pattern embeddings
│   └── drift_detector.py    # Statistical drift detection
├── jurisdictions/            # 🆕 Court-specific context
│   ├── base.py              # Abstract jurisdiction classes
│   ├── nd_cal.py            # N.D. California rules
│   └── alsup.py             # Judge Alsup preferences
├── llm/
│   ├── client.py            # LLM integration hub
│   └── providers/           # Provider implementations
├── examples/                 # Compelling demo datasets
├── streamlit_demo/           # Interactive web demo (9 pages)
├── tests/
└── unified_system.py         # Main integration layer
```

## 🆕 New Feature Examples

### Entity Normalization

```python
from normalizers import JudgeNormalizer, LawyerNormalizer

# Clean messy judge names from various sources
normalizer = JudgeNormalizer()

# From CourtListener URL
result = normalizer.normalize("https://courtlistener.com/person/john-g-roberts-jr/")
print(result.normalized_name)  # "John G. Roberts Jr."

# From PACER format
result = normalizer.normalize("ALSUP, WILLIAM H.")
print(result.normalized_name)  # "William H. Alsup"

# Validate lawyer entities (filter out cities, organizations, pro se)
lawyer_normalizer = LawyerNormalizer()
result = lawyer_normalizer.validate("San Francisco")
print(result.is_valid)  # False - geographic location
```

### Concentration Risk Detection

```python
from concentration import HHICalculator, SPOFDetector

# Calculate market concentration using HHI
calc = HHICalculator()
result = calc.from_counts({
    "Judge Gilstrap": 450,
    "Judge Payne": 180,
    "Judge Schroeder": 150,
    "Others": 220
})
print(f"HHI: {result.hhi}")  # ~2800 (highly concentrated)
print(f"Level: {result.level}")  # concentrated

# Detect Single Point of Failure risks
detector = SPOFDetector()
assessment = detector.analyze(case_counts, entity_type="judge", domain="patent")
if assessment.has_critical_spof:
    print(f"SPOF Alert: {assessment.top_spof.entity_id}")
```

### Pattern Drift Detection

```python
from drift import EmbeddingTracker, DriftDetector

# Track pattern changes over time
tracker = EmbeddingTracker()
detector = DriftDetector()

# Set baseline from historical data
baseline = tracker.generate(
    entity_id="judge_alsup",
    pattern_type="summary_judgment",
    metrics={"grant_rate": 0.45, "avg_days": 120}
)
detector.set_baseline(baseline)

# Check for drift in current pattern
current = tracker.generate(...)
drift_event = detector.detect_drift(current)

if drift_event.requires_recalibration:
    print(f"DRIFT ALERT: {drift_event.severity}")
    print(drift_event.recommendation)
```

### Jurisdictional Context

```python
from unified_system import EpistemicFlowControl, SystemConfig

# Configure for a specific judge
config = SystemConfig(
    domain="judicial",
    jurisdiction="nd_cal",
    judge="alsup"
)
system = EpistemicFlowControl(config)

# Get format requirements
requirements = system.get_format_requirements()
# Returns: 14pt Times New Roman, 25-page limit, etc.

# Get procedural rules for a motion type
rules = system.get_procedural_rules("summary_judgment")
```

## 🎭 Example: The Changing Judge

One of our demo stories shows why this matters:

**Judge Rodriguez** had a 78% summary judgment grant rate. Then she became Chief Judge.

With new administrative duties, her grant rate dropped to 42%. A system relying on historical data would be **dangerously wrong**.

Epistemic Flow Control:
1. ⏳ **Temporal decay** reduces confidence in old patterns
2. 📉 **Bayesian updating** adjusts weights with new evidence
3. 🚪 **Review gate** routes uncertain predictions to humans
4. 📈 **Calibration** ensures confidence matches reality

The system doesn't try to be perfect. It **knows when it's uncertain**.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check .
```

## 📚 Documentation

- [Training Data Requirements](TRAINING_DATA_REQUIREMENTS.md) — How to bootstrap the system
- [Validation Package](VALIDATION_PACKAGE.md) — Verification and testing guide
- [LLM Layer Review](LLM_LAYER_REVIEW.md) — Technical deep-dive into LLM integration

## 📜 License

[Apache 2.0](LICENSE) — Use it, modify it, build on it.

## ⭐ Star History

If this project helps you build more reliable AI systems, consider giving it a star!

---

<p align="center">
  <strong>Built for decisions that matter.</strong><br>
  <em>Because "probably right" isn't good enough when stakes are high.</em>
</p>
