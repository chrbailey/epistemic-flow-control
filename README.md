# Epistemic Flow Control

**Human-gated probabilistic intelligence for high-stakes domains.**

## The Water in Sand Metaphor

LLMs generate probabilistic output like water flowing. Humans don't create the water - they control where it flows by opening, closing, and adjusting channels.

```
LLM Output (Water) → Human Gates (Channels) → Production Use (Destination)
```

The human role:
1. **Open channels** - approve high-confidence outputs
2. **Close channels** - block low-confidence or high-risk outputs
3. **Adjust flow** - override weights based on domain expertise
4. **Build new paths** - identify patterns machines miss

## Architecture

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

## Components

| Component | Purpose | Human Gate |
|-----------|---------|------------|
| `EventStore` | Ground truth storage | Event verification |
| `PatternExtractor` | Extract patterns from events | Extraction validation |
| `PatternDatabase` | Store patterns with Bayesian weights | Weight override |
| `ReviewGate` | Control flow to production | Review decisions |
| `CalibrationEngine` | Track prediction accuracy | Outcome recording |
| `TrainingDataGenerator` | Prepare labeled examples | All labeling tasks |

## Quick Start

```python
from unified_system import EpistemicFlowControl, SystemConfig

# Initialize
config = SystemConfig(
    db_dir="./data",
    domain="judicial"
)
system = EpistemicFlowControl(config)

# Register a source
system.register_source(
    source_id="pacer",
    name="PACER",
    source_type="official",
    reliability=0.99
)

# Ingest an event
result = system.ingest_event(
    what="Judge granted summary judgment",
    who=["Judge Smith", "Plaintiff Corp", "Defendant Inc"],
    when=datetime.now(),
    where="N.D. Cal",
    source_id="pacer",
    raw_text="Order granting motion for summary judgment..."
)

# Get patterns for a subject
patterns = system.get_patterns_for_subject("Judge Smith")

# Make a prediction
prediction = system.make_prediction(
    prediction_type="ruling",
    predicted_value="Motion will be granted",
    context={"case_type": "patent", "motion": "summary_judgment"},
    source_patterns=["pat_001", "pat_002"],
    stakes="high"
)

# Check what needs human review
items = system.get_items_needing_review()

# Submit human review
system.submit_human_review(
    item_id=prediction["prediction_id"],
    reviewer_id="expert_001",
    decision="approve",
    notes="Consistent with pattern"
)

# Record outcome (when known)
system.record_prediction_outcome(
    prediction_id=prediction["prediction_id"],
    actual_value="Motion was granted",
    was_correct=True
)
```

## Training Data Requirements

The system needs **small amounts of high-quality data** to bootstrap:

| Data Type | Minimum | Purpose |
|-----------|---------|---------|
| Source Reliability | 50 | Trust calibration |
| Pattern Extractions | 100 | Extraction accuracy |
| Prediction Outcomes | 200 | Confidence calibration |
| Human Overrides | 20 | Override learning |

See `TRAINING_DATA_REQUIREMENTS.md` for detailed collection guide.

## Key Principles

### 1. Events as Ground Truth
Everything traces back to verifiable events. No patterns without events.

### 2. Bayesian Updating
Pattern weights update with new evidence. Confidence grows with data.

### 3. Temporal Decay
Patterns become stale without new confirming events. Old patterns decay.

### 4. Human Gates
Humans control the flow. High-stakes decisions require human approval.

### 5. Calibrated Confidence
Confidence scores match actual accuracy. 80% confidence = right 80% of time.

### 6. Outcome Learning
Every outcome is training data. The system improves from use.

## CLI Interface

```bash
# Check system status
python unified_system.py --domain judicial status

# Get system health metrics
python unified_system.py health

# Run calibration
python unified_system.py calibrate

# Apply temporal decay (run daily)
python unified_system.py decay

# Check training data status
python unified_system.py training
```

## File Structure

```
epistemic-flow-control/
├── core/
│   ├── event_store.py      # Ground truth layer
│   ├── pattern_extractor.py # LLM pattern extraction
│   └── pattern_database.py  # Bayesian weight storage
├── gates/
│   └── review_gate.py      # Human review flow control
├── validation/
│   └── calibration_engine.py # Accuracy tracking
├── training/
│   └── data_generator.py   # Training data collection
├── tests/
├── unified_system.py       # Integration layer
├── TRAINING_DATA_REQUIREMENTS.md
└── README.md
```

## Statistical Foundation

- **Wilson Score Intervals**: Conservative confidence bounds for small samples
- **Bayesian Updating**: Prior beliefs + observations = posterior beliefs
- **Temporal Decay**: Exponential decay with configurable half-life
- **Expected Calibration Error (ECE)**: Measure of confidence calibration

## Design Philosophy

This system is built on the insight that:

> "LLM output is probabilistically reliable but not deterministically correct.
> Human oversight adds irreplaceable value at specific gate points."

The competitive advantage is not better LLM prompts - it's the **human oversight infrastructure** that makes probabilistic output reliable for high-stakes use.

## License

Proprietary. Copyright 2025.
