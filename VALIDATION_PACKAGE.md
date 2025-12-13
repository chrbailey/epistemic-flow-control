# Epistemic Flow Control - Validation Package

**For independent LLM verification**

## Location
```
/Users/christopherbailey/Desktop/AI-WORK-HERE/epistemic-flow-control
```

## What Was Built

A human-gated probabilistic intelligence system with:
1. **Event Store** - Ground truth storage with source reliability
2. **Pattern Extractor** - Extracts patterns from events (rule-based demo mode)
3. **Pattern Database** - Bayesian weight updates + temporal decay
4. **Review Gate** - Human approval workflow for uncertain predictions
5. **Calibration Engine** - Learns from outcomes to correct overconfidence
6. **Unified System** - Integration layer with CLI

## File Structure
```
epistemic-flow-control/
├── core/
│   ├── event_store.py        # Event storage, source reliability
│   ├── pattern_extractor.py  # Pattern extraction from events
│   └── pattern_database.py   # Bayesian weights, temporal decay
├── gates/
│   └── review_gate.py        # Human review workflow
├── validation/
│   └── calibration_engine.py # Calibration from outcomes
├── training/
│   └── data_generator.py     # Training data collection
├── unified_system.py         # Integration layer
├── build_samples.py          # Test data generator
├── live_test.py              # Demo script
└── live_test_data/           # SQLite databases with test data
    ├── events.db
    ├── patterns.db
    └── calibration.db
```

## Core Algorithms

### 1. Bayesian Weight Update (pattern_database.py)
```python
def bayesian_update(prior_mean, prior_strength, observed_successes, observed_total):
    prior_alpha = prior_mean * prior_strength
    prior_beta = (1 - prior_mean) * prior_strength
    posterior_alpha = prior_alpha + observed_successes
    posterior_beta = prior_beta + (observed_total - observed_successes)
    posterior_strength = posterior_alpha + posterior_beta
    posterior_mean = posterior_alpha / posterior_strength
    return posterior_mean, posterior_strength
```

**Default prior**: mean=0.5, strength=2

**Example**: 12 observations
- (0.5×2 + 12) / (2 + 12) = 13/14 = 0.929

### 2. Confidence Computation (unified_system.py)
```python
# Geometric mean of pattern weights
raw_confidence = math.exp(sum(math.log(w) for w in pattern_weights) / len(pattern_weights))

# Apply calibration
calibrated_confidence = raw_confidence * calibration_factor
```

### 3. Calibration Factor (calibration_engine.py)
```python
# Factor = actual_accuracy / stated_confidence
# If system says 90% but is right 64% of time:
# factor = 0.64 / 0.90 = 0.71
```

## Verification Commands

### Step 1: Verify Events in Database
```bash
cd /Users/christopherbailey/Desktop/AI-WORK-HERE/epistemic-flow-control

sqlite3 live_test_data/events.db "SELECT COUNT(*) FROM events"
# Expected: 180

sqlite3 live_test_data/events.db "
SELECT
  CASE
    WHEN who LIKE '%Gilstrap%' THEN 'Gilstrap'
    WHEN who LIKE '%Albright%' THEN 'Albright'
    WHEN who LIKE '%Alsup%' THEN 'Alsup'
    WHEN who LIKE '%Koh%' THEN 'Koh'
    WHEN who LIKE '%O''Malley%' THEN 'OMalley'
  END as judge,
  COUNT(*) as events,
  SUM(CASE WHEN what LIKE '%granted%' THEN 1 ELSE 0 END) as grants,
  SUM(CASE WHEN what LIKE '%denied%' THEN 1 ELSE 0 END) as denies
FROM events
GROUP BY judge"
```

### Step 2: Verify Patterns Accumulated
```bash
sqlite3 live_test_data/patterns.db "SELECT COUNT(*) FROM patterns"
# Expected: 30 (6 patterns per judge × 5 judges)

sqlite3 live_test_data/patterns.db "
SELECT pattern_key, raw_weight, supporting_events
FROM patterns
WHERE subject = 'Judge Rodney Gilstrap'"
```

### Step 3: Verify Bayesian Calculation
```bash
sqlite3 live_test_data/patterns.db "
SELECT supporting_events, raw_weight,
  ROUND((0.5 * 2 + supporting_events) / (2.0 + supporting_events), 6) as expected_weight
FROM patterns
LIMIT 5"
# raw_weight should equal expected_weight
```

### Step 4: Verify Predictions
```bash
sqlite3 live_test_data/calibration.db "SELECT COUNT(*) FROM predictions WHERE outcome_recorded = 1"
# Expected: 250

sqlite3 live_test_data/calibration.db "
SELECT
  ROUND(confidence, 2) as conf,
  COUNT(*) as total,
  SUM(was_correct) as correct,
  ROUND(100.0 * SUM(was_correct) / COUNT(*), 1) as accuracy
FROM predictions
WHERE outcome_recorded = 1
GROUP BY ROUND(confidence, 2)"
```

### Step 5: Verify Calibration Factor
```bash
sqlite3 live_test_data/calibration.db "SELECT factor, sample_size FROM calibration_factors"
# Expected: factor ≈ 0.71-0.72, sample_size = 250
```

### Step 6: Run Full Verification Script
```python
#!/usr/bin/env python3
"""Run this to verify the system end-to-end"""
import sqlite3
import math

def verify():
    print("=" * 60)
    print("EPISTEMIC FLOW CONTROL - VERIFICATION")
    print("=" * 60)

    # 1. Check events
    conn = sqlite3.connect("live_test_data/events.db")
    events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    print(f"\n1. Events: {events}")
    assert events > 0, "No events found!"

    # 2. Check patterns
    conn = sqlite3.connect("live_test_data/patterns.db")
    patterns = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]

    # Verify Bayesian calculation
    row = conn.execute("""
        SELECT supporting_events, raw_weight
        FROM patterns LIMIT 1
    """).fetchone()
    obs, weight = row
    expected = (0.5 * 2 + obs) / (2 + obs)
    conn.close()

    print(f"2. Patterns: {patterns}")
    print(f"   Bayesian check: {obs} obs → weight {weight:.4f} (expected {expected:.4f})")
    assert abs(weight - expected) < 0.001, "Bayesian calculation mismatch!"

    # 3. Check predictions
    conn = sqlite3.connect("live_test_data/calibration.db")
    preds = conn.execute("""
        SELECT COUNT(*), SUM(was_correct), AVG(confidence)
        FROM predictions WHERE outcome_recorded = 1
    """).fetchone()
    total, correct, avg_conf = preds
    accuracy = correct / total if total > 0 else 0

    # Check calibration factor
    factor_row = conn.execute("SELECT factor FROM calibration_factors").fetchone()
    factor = factor_row[0] if factor_row else 1.0
    conn.close()

    print(f"3. Predictions: {total}")
    print(f"   Correct: {correct} ({accuracy:.1%})")
    print(f"   Avg confidence: {avg_conf:.1%}")
    print(f"   Calibration factor: {factor:.3f}")

    # Verify calibration logic
    expected_factor = accuracy / avg_conf
    print(f"   Expected factor: {expected_factor:.3f}")
    assert abs(factor - expected_factor) < 0.05, "Calibration factor mismatch!"

    # 4. Test new prediction with calibration
    from unified_system import EpistemicFlowControl, SystemConfig
    config = SystemConfig(db_dir="./live_test_data", domain="judicial")
    system = EpistemicFlowControl(config)

    patterns = system.get_patterns_for_subject("Judge Rodney Gilstrap", min_confidence=0.1)
    if patterns:
        prediction = system.make_prediction(
            prediction_type="ruling",
            predicted_value="Test prediction",
            context={"test": True},
            source_patterns=[p["pattern_id"] for p in patterns[:3]],
            stakes="high"
        )
        print(f"\n4. New prediction test:")
        print(f"   Raw confidence: {prediction['raw_confidence']:.1%}")
        print(f"   Calibrated: {prediction['calibrated_confidence']:.1%}")
        print(f"   Gate decision: {prediction['gate_decision']}")

        # Verify calibration applied
        expected_calibrated = prediction['raw_confidence'] * factor
        assert abs(prediction['calibrated_confidence'] - expected_calibrated) < 0.01, \
            "Calibration not applied correctly!"

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    verify()
```

## Expected Database Schema

### events.db
```sql
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    who TEXT NOT NULL,
    what TEXT NOT NULL,
    when_occurred TEXT NOT NULL,
    where_location TEXT,
    why TEXT,
    source_id TEXT NOT NULL,
    raw_text TEXT,
    verification_status TEXT,
    domain TEXT,
    event_type TEXT
);
```

### patterns.db
```sql
CREATE TABLE patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_key TEXT NOT NULL,
    subject TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    description TEXT,
    raw_weight REAL DEFAULT 0.5,
    decayed_weight REAL DEFAULT 0.5,
    supporting_events INTEGER DEFAULT 0,
    total_observations INTEGER DEFAULT 0
);
```

### calibration.db
```sql
CREATE TABLE predictions (
    prediction_id TEXT PRIMARY KEY,
    prediction_type TEXT,
    predicted_value TEXT,
    confidence REAL,
    context TEXT,
    was_correct INTEGER,
    outcome_recorded INTEGER DEFAULT 0
);

CREATE TABLE calibration_factors (
    domain TEXT PRIMARY KEY,
    factor REAL DEFAULT 1.0,
    computed_at TEXT,
    sample_size INTEGER
);
```

## Key Assertions to Verify

1. **Events exist**: `SELECT COUNT(*) FROM events` > 0
2. **Patterns accumulated**: `SELECT COUNT(*) FROM patterns` = 30
3. **Bayesian math correct**: `raw_weight = (0.5*2 + n) / (2 + n)` for n observations
4. **Predictions recorded**: 250 predictions with outcomes
5. **Confidence varies**: Multiple distinct confidence values (not all 0.5)
6. **Calibration computed**: Factor ≈ 0.71-0.72 (not 1.0)
7. **Calibration applied**: New predictions have `calibrated = raw * factor`

## What Would Indicate Failure

- All confidence values = 0.5 (hardcoded default)
- Calibration factor = 1.0 (never computed)
- 0 patterns in database (extraction not working)
- Bayesian weights don't match formula
- Raw confidence equals calibrated confidence

## Quick Verification One-Liner
```bash
cd /Users/christopherbailey/Desktop/AI-WORK-HERE/epistemic-flow-control && \
sqlite3 live_test_data/calibration.db "SELECT 'Predictions:', COUNT(*), 'Calibration:', (SELECT ROUND(factor,3) FROM calibration_factors) FROM predictions WHERE outcome_recorded=1" && \
sqlite3 live_test_data/patterns.db "SELECT 'Patterns:', COUNT(*), 'Avg weight:', ROUND(AVG(raw_weight),3) FROM patterns" && \
sqlite3 live_test_data/events.db "SELECT 'Events:', COUNT(*) FROM events"
```

Expected output:
```
Predictions:|250|Calibration:|0.719
Patterns:|30|Avg weight:|0.864
Events:|180
```
