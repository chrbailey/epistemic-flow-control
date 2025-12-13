# Training Data Requirements

## The "Aunt/Uncle" Principle

> "A little bit of accurate data goes a long way, no different than your parent's siblings who fill in missing details in stories and expand your complete understanding."

This system is designed to be bootstrapped with **small amounts of high-quality data**. Expert labels are extremely valuable. The system learns from usage over time.

---

## Minimum Viable Training Data

To bootstrap the system, you need:

| Data Type | Minimum | Purpose |
|-----------|---------|---------|
| Source Reliability Labels | 50 | Calibrate how much to trust different sources |
| Pattern Extraction Examples | 100 | Teach pattern extraction accuracy |
| Prediction Outcomes | 200 | Calibrate confidence scores |
| Human Override Outcomes | 20 | Learn when to trust human judgment |
| Domain Priors | 30 | Anchor Bayesian system |

**Total effort estimate: 20-40 hours of expert labeling**

---

## Detailed Requirements by Component

### 1. Event Store

#### Source Reliability Labels

**What you need:**
```json
{
  "source_id": "pacer",
  "name": "PACER - Federal Court Records",
  "reliability_score": 99,
  "reasoning": "Official government court records, highest reliability",
  "caveats": "Data entry errors possible but rare"
}
```

**How to collect:**
1. List all information sources you use (50-100 sources)
2. Have domain experts rate each on 0-100 scale
3. Require reasoning for each rating
4. Review and reconcile disagreements

**Collection template:**
```
SOURCE RELIABILITY TASK

For each source, provide:
1. Reliability score (0-100)
   - 90-100: Official records, government sources
   - 70-89: Major journalism, peer-reviewed
   - 50-69: Reputable but may have bias
   - 30-49: Variable quality
   - 0-29: Unreliable

2. Why this score?
3. Any situations where reliability changes?
```

#### Event Verification Labels

**What you need:**
```json
{
  "event_id": "evt_001",
  "verification_status": "verified",
  "verification_method": "Cross-referenced with PACER and Law360",
  "confidence": 0.95,
  "notes": "Multiple independent sources confirm"
}
```

**How to collect:**
1. Sample 100 events from the system
2. Have experts verify against primary sources
3. Record verification status and method

---

### 2. Pattern Extractor

#### Pattern Extraction Examples (Gold Standard)

**What you need:**
```json
{
  "event_text": "Judge Gilstrap granted summary judgment in 3 of 47 patent cases in 2023",
  "correct_extraction": {
    "pattern_type": "outcome_pattern",
    "subject": "Judge Gilstrap",
    "description": "Low summary judgment grant rate in patent cases (6.4%)",
    "structured_pattern": {
      "metric": "summary_judgment_grant_rate",
      "value": 0.064,
      "sample_size": 47,
      "time_period": "2023"
    },
    "confidence": 0.92,
    "reasoning": "Large sample, clear quantitative pattern"
  }
}
```

**How to collect:**
1. Run LLM extraction on 100 events
2. Have experts review each extraction
3. Mark as CORRECT, PARTIAL, or INCORRECT
4. For PARTIAL/INCORRECT, provide correction
5. Corrections are especially valuable

**Collection template:**
```
PATTERN EXTRACTION VALIDATION

Event: [event text]

LLM Extracted:
- Pattern type: [type]
- Description: [description]
- Confidence: [confidence]

Your evaluation:
[ ] CORRECT - Extraction is accurate
[ ] PARTIAL - Partially right, needs refinement
[ ] INCORRECT - Fundamentally wrong

If not CORRECT, what is the correct extraction?
[Provide corrected pattern]

What did the LLM miss or misunderstand?
[Explanation]
```

---

### 3. Pattern Database

#### Domain Priors

**What you need:**
```json
{
  "pattern_key": "summary_judgment_grant_rate|judicial",
  "prior_mean": 0.20,
  "prior_strength": 10,
  "reasoning": "National average SJ grant rate is approximately 20%",
  "uncertainty_notes": "Varies significantly by case type and jurisdiction"
}
```

**How to collect:**
1. List key pattern types for your domain
2. Ask experts: "Before seeing any data, what's your estimate?"
3. Ask: "How confident? How many observations to change your mind?"

**Collection template:**
```
PRIOR ELICITATION

Domain: [domain]
Pattern: [pattern description]

Question 1: What is your prior estimate?
Before seeing any data, what's your best guess for the base rate?
Example: "What fraction of judges grant summary judgment?"

Your estimate: _____ (0 to 1)

Question 2: How confident are you?
If you saw 10 observations that contradicted your estimate,
would you change your mind?
- 1-10: Very uncertain, easily swayed
- 10-30: Moderate confidence
- 30-100: Very confident, need lots of data

Your confidence: _____

Question 3: What's your reasoning?
Why do you believe this? What's it based on?

Reasoning: _____
```

---

### 4. Review Gate

#### Gate Decision Outcomes

**What you need:**
```json
{
  "item_id": "pred_001",
  "gate_decision": "review_required",
  "human_decision": "approve",
  "was_correct": true,
  "notes": "Prediction was accurate, human approval appropriate"
}
```

**How to collect:**
1. Track all items that go through the gate
2. Record gate decision and human decision
3. Later, record actual outcome
4. Analyze: Did the gate make good decisions?

---

### 5. Calibration Engine

#### Prediction Outcomes

**What you need:**
```json
{
  "prediction_id": "pred_001",
  "predicted_value": "Judge will grant motion to dismiss",
  "confidence": 0.75,
  "actual_outcome": "Motion to dismiss denied",
  "was_correct": false,
  "notes": "Judge cited genuine issues of fact not addressed in motion"
}
```

**How to collect:**
1. Record every prediction the system makes
2. Set deadline for when outcome will be known
3. At deadline, record actual outcome
4. Mark as correct or incorrect
5. Analyze accuracy by confidence bucket

**Collection template:**
```
PREDICTION OUTCOME RECORDING

Prediction: [prediction description]
Confidence: [confidence]%
Made on: [date]

What actually happened?
[Actual outcome]

Was the prediction correct?
[ ] CORRECT
[ ] INCORRECT

If incorrect, why?
[Explanation]

Was the confidence level appropriate?
[ ] Yes, confidence matched outcome likelihood
[ ] Too high - we were overconfident
[ ] Too low - we were underconfident
```

---

### 6. Human Override Tracking

#### Override Outcomes

**What you need:**
```json
{
  "pattern_id": "pat_001",
  "model_weight": 0.65,
  "human_override_weight": 0.85,
  "override_reason": "Recent events suggest pattern is strengthening",
  "outcome_was_correct": true,
  "outcome_notes": "Human was right, pattern did strengthen"
}
```

**How to collect:**
1. Record when humans override model weights
2. Record their reasoning
3. Later, verify if override was correct
4. Learn when to trust human overrides

---

## Collection Strategy

### Phase 1: Bootstrap (Week 1-2)

Focus on minimum viable data:
1. **Day 1-2**: Label 50 sources for reliability
2. **Day 3-5**: Extract patterns from 50 events, validate
3. **Day 6-7**: Elicit priors from domain experts

### Phase 2: Calibration (Weeks 3-8)

Collect outcome data:
1. Make predictions, record outcomes
2. Track gate decisions, verify correctness
3. Record human overrides and verify

### Phase 3: Continuous (Ongoing)

The system learns from usage:
1. Every prediction with recorded outcome improves calibration
2. Every human review creates training data
3. Every override teaches when to trust humans

---

## Data Quality Guidelines

### For Labelers

1. **Be honest, not optimistic**
   - If prediction was partially correct, mark INCORRECT
   - We need accurate calibration, not inflated accuracy

2. **Explain your reasoning**
   - Your explanations teach the system
   - Future labelers learn from your notes

3. **Flag uncertainty**
   - If you're not sure, say so
   - Uncertain labels can be reviewed by experts

### For Expert Reviewers

1. **Prioritize corrections over confirmations**
   - Corrections are 10x more valuable than confirmations
   - Seek out cases where the system is wrong

2. **Look for systematic errors**
   - Is the system consistently wrong in certain cases?
   - Document patterns of errors

3. **Calibrate confidence appropriately**
   - High confidence should be rare
   - Most predictions should be 60-80% confident

---

## Tools for Data Collection

### Web Interface (Recommended)

Build a simple web UI for labelers:
- Present one task at a time
- Capture structured responses
- Track labeler progress
- Enable expert review

### Spreadsheet (Quick Start)

For bootstrapping:
- Create Google Sheet with task template
- Share with labelers
- Import completed labels

### API Integration

For ongoing collection:
- Hook outcome recording into your workflow
- Auto-create prediction records
- Track outcomes as they become known

---

## Measuring Data Quality

### Inter-Rater Reliability

Have multiple labelers label the same items:
- Source reliability: Expect 80%+ agreement within ±10 points
- Pattern extraction: Expect 70%+ agreement on correctness
- Outcome recording: Expect 95%+ agreement (factual)

### Expert Review Coverage

Track what percentage of labels are expert-reviewed:
- Target: 100% of corrections reviewed
- Target: 20% random sample of confirmations reviewed

### Calibration Quality

Measure if collected data improves calibration:
- ECE (Expected Calibration Error) should decrease over time
- Accuracy at each confidence level should converge to confidence

---

## Example: Bootstrapping Judicial Domain

### Week 1: Sources

Label these judicial sources:
1. PACER (federal court records)
2. Law360 (legal journalism)
3. Bloomberg Law
4. Lex Machina (analytics)
5. The Texas Lawbook
6. IP Watchdog
7. Judge-specific blogs
8. Twitter/X legal commentary
...etc (50 total)

### Week 1: Priors

Elicit priors for:
1. Summary judgment grant rate
2. Motion to dismiss grant rate
3. Case duration (filing to resolution)
4. Jury trial rate
5. Settlement rate
6. Appeal rate
...etc (30 total pattern types)

### Week 2: Pattern Extraction

Validate extractions from:
1. 20 recent jury verdicts
2. 20 motion rulings
3. 20 opinion/order filings
4. 20 procedural events
5. 20 settlement announcements
...etc (100 total events)

### Weeks 3-8: Outcomes

Record outcomes for:
1. All predictions made
2. All gate decisions
3. All human overrides

Target: 200+ predictions with outcomes by week 8

---

## Summary: What You Need to Collect

| Priority | Data Type | Min Samples | Collection Method |
|----------|-----------|-------------|-------------------|
| HIGH | Source reliability | 50 | Expert survey |
| HIGH | Pattern extractions | 100 | LLM + human validation |
| HIGH | Prediction outcomes | 200 | Outcome tracking |
| MEDIUM | Domain priors | 30 | Expert elicitation |
| MEDIUM | Human overrides | 20 | Override tracking |
| LOW | Gate threshold calibration | 50 | Threshold search |

Start with HIGH priority. The system works with partial data but improves with more.

---

## Files Reference

Training data collection is managed by:
- `training/data_generator.py` - Task generation and management
- `validation/calibration_engine.py` - Outcome tracking
- `gates/review_gate.py` - Human review tracking

Run `python unified_system.py training` to see current status.
