#!/usr/bin/env python3
"""
Live Test - Demonstrates the full epistemic flow control system.
Run this to see everything working together.
"""

from datetime import datetime
from unified_system import EpistemicFlowControl, SystemConfig

def main():
    print("=" * 70)
    print("EPISTEMIC FLOW CONTROL - LIVE TEST")
    print("=" * 70)
    print()

    # Initialize system
    config = SystemConfig(
        db_dir="./live_test_data",
        domain="judicial"
    )
    system = EpistemicFlowControl(config)

    # Step 1: Register sources
    print("STEP 1: Registering information sources...")
    print("-" * 70)

    system.register_source(
        source_id="pacer",
        name="PACER - Federal Court Records",
        source_type="official",
        reliability=0.99,
        notes="Official government records"
    )
    print("  ✓ Registered PACER (reliability: 99%)")

    system.register_source(
        source_id="law360",
        name="Law360 - Legal News",
        source_type="journalism",
        reliability=0.85,
        notes="Reputable legal journalism"
    )
    print("  ✓ Registered Law360 (reliability: 85%)")

    # Step 2: Register a human reviewer
    print()
    print("STEP 2: Registering human reviewer...")
    print("-" * 70)

    system.register_reviewer(
        reviewer_id="expert_001",
        name="Domain Expert",
        role="expert",
        domains=["judicial"]
    )
    print("  ✓ Registered expert_001 (Domain Expert)")

    # Step 3: Ingest events
    print()
    print("STEP 3: Ingesting events (ground truth)...")
    print("-" * 70)

    events = [
        {
            "what": "Judge Gilstrap granted summary judgment in patent case",
            "who": ["Judge Rodney Gilstrap", "TechCorp Inc", "PatentHold LLC"],
            "when": datetime(2024, 6, 15),
            "where": "E.D. Texas, Marshall Division",
            "source_id": "pacer",
            "raw_text": "ORDER: Defendant's motion for summary judgment is GRANTED...",
            "why": "No genuine issue of material fact on infringement",
            "event_type": "summary_judgment"
        },
        {
            "what": "Judge Gilstrap denied summary judgment in patent case",
            "who": ["Judge Rodney Gilstrap", "InnovateCo", "BigTech Inc"],
            "when": datetime(2024, 7, 20),
            "where": "E.D. Texas, Marshall Division",
            "source_id": "pacer",
            "raw_text": "ORDER: Plaintiff's motion for summary judgment is DENIED...",
            "why": "Genuine issues of fact remain on validity",
            "event_type": "summary_judgment"
        },
        {
            "what": "Judge Gilstrap scheduled Markman hearing for 4 hours",
            "who": ["Judge Rodney Gilstrap", "ChipMaker Corp", "DesignLabs"],
            "when": datetime(2024, 8, 5),
            "where": "E.D. Texas, Marshall Division",
            "source_id": "law360",
            "raw_text": "Markman hearing set for September 15, expected duration 4 hours...",
            "event_type": "scheduling"
        }
    ]

    for event in events:
        result = system.ingest_event(**event)
        if result["success"]:
            print(f"  ✓ Ingested: {event['what'][:50]}...")
            print(f"    Verification: {result['verification_status']}")
            if result['patterns_extracted']:
                for p in result['patterns_extracted']:
                    print(f"    Pattern: {p['description'][:40]}... (conf: {p['confidence']:.0%})")
        else:
            print(f"  ✗ Failed: {result['message']}")

    # Step 4: Query patterns
    print()
    print("STEP 4: Querying patterns for Judge Gilstrap...")
    print("-" * 70)

    patterns = system.get_patterns_for_subject("Judge Gilstrap", min_confidence=0.3)
    if patterns:
        for p in patterns:
            print(f"  Pattern: {p['description']}")
            print(f"    Type: {p['pattern_type']}")
            print(f"    Weight: {p['weight']:.0%}")
            print(f"    Based on: {p['supporting_events']} events")
            print()
    else:
        print("  (No patterns extracted yet - need more events or LLM extraction)")

    # Step 5: Make a prediction
    print()
    print("STEP 5: Making a prediction...")
    print("-" * 70)

    prediction = system.make_prediction(
        prediction_type="ruling",
        predicted_value="Judge will deny motion for summary judgment",
        context={
            "judge": "Judge Gilstrap",
            "motion_type": "summary_judgment",
            "case_type": "patent"
        },
        source_patterns=[],  # Would include pattern IDs in real use
        stakes="medium"
    )

    print(f"  Prediction: {prediction['predicted_value']}")
    print(f"  Raw confidence: {prediction['raw_confidence']:.0%}")
    print(f"  Calibrated confidence: {prediction['calibrated_confidence']:.0%}")
    print(f"  Gate decision: {prediction['gate_decision']}")
    print(f"  Needs review: {prediction['needs_human_review']}")

    # Step 6: Check review queue
    print()
    print("STEP 6: Checking human review queue...")
    print("-" * 70)

    items = system.get_items_needing_review()
    print(f"  Items needing review: {len(items)}")

    if items:
        item = items[0]
        print(f"  First item: {item['item_id']}")
        print(f"  Type: {item['item_type']}")
        print(f"  Confidence: {item['confidence']:.0%}")

        # Submit review
        print()
        print("  Submitting human review (approving)...")
        system.submit_human_review(
            item_id=item['item_id'],
            reviewer_id="expert_001",
            decision="approve",
            notes="Pattern consistent with historical behavior"
        )
        print("  ✓ Review submitted")

    # Step 7: Record outcome
    print()
    print("STEP 7: Recording prediction outcome (simulating future)...")
    print("-" * 70)

    system.record_prediction_outcome(
        prediction_id=prediction['prediction_id'],
        actual_value="Motion was denied",
        was_correct=True,
        notes="Prediction matched actual ruling"
    )
    print("  ✓ Outcome recorded: CORRECT")

    # Step 8: Check calibration
    print()
    print("STEP 8: Checking calibration status...")
    print("-" * 70)

    calibration = system.get_calibration_status()
    print(f"  Current calibration factor: {calibration['current_factor']:.2f}")
    print(f"  Recommendations: {calibration['recommendations']}")

    # Step 9: Training data status
    print()
    print("STEP 9: Training data requirements...")
    print("-" * 70)

    training = system.get_training_data_status()
    print(f"  Requirements satisfied: {training['stats']['requirements_satisfied']}")
    print()
    print("  What you need to collect:")

    for req in training['requirements']:
        if not req['is_satisfied']:
            print(f"    [{req['priority'].upper()}] {req['data_type']}: {req['progress']}")

    # Step 10: System health
    print()
    print("STEP 10: Final system health...")
    print("-" * 70)

    health = system.get_system_health()
    print(f"  Events stored: {health['event_store']['total_events']}")
    print(f"  Patterns stored: {health['pattern_database']['total_patterns']}")
    print(f"  Items processed: {health['review_gate']['total_items_processed']}")
    print(f"  Predictions made: {health['calibration']['total_predictions']}")

    print()
    print("=" * 70)
    print("LIVE TEST COMPLETE")
    print("=" * 70)
    print()
    print("The system is working. Next steps:")
    print("1. Collect training data (see TRAINING_DATA_REQUIREMENTS.md)")
    print("2. Ingest real events from your domain")
    print("3. Make predictions and record outcomes")
    print("4. Let the system learn from usage")
    print()


if __name__ == "__main__":
    main()
