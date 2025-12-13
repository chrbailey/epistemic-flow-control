#!/usr/bin/env python3
"""
STANDALONE VERIFICATION SCRIPT
Run this to verify the epistemic flow control system is computing real values.
"""
import sqlite3
import math
import os
import sys

def main():
    print("=" * 70)
    print("EPISTEMIC FLOW CONTROL - INDEPENDENT VERIFICATION")
    print("=" * 70)

    db_dir = "./live_test_data"
    if not os.path.exists(db_dir):
        print(f"ERROR: {db_dir} not found. Run build_samples.py first.")
        sys.exit(1)

    errors = []

    # =========================================================================
    # TEST 1: Events exist and have variety
    # =========================================================================
    print("\n[TEST 1] Events Database")
    print("-" * 40)

    conn = sqlite3.connect(f"{db_dir}/events.db")
    cursor = conn.cursor()

    # Count events
    cursor.execute("SELECT COUNT(*) FROM events")
    event_count = cursor.fetchone()[0]
    print(f"  Total events: {event_count}")

    if event_count == 0:
        errors.append("No events in database")
    elif event_count < 100:
        errors.append(f"Only {event_count} events (expected ~180)")

    # Check variety in events
    cursor.execute("""
        SELECT COUNT(DISTINCT source_id) as sources,
               COUNT(DISTINCT verification_status) as statuses,
               SUM(CASE WHEN what LIKE '%granted%' THEN 1 ELSE 0 END) as grants,
               SUM(CASE WHEN what LIKE '%denied%' THEN 1 ELSE 0 END) as denies
        FROM events
    """)
    sources, statuses, grants, denies = cursor.fetchone()
    print(f"  Distinct sources: {sources}")
    print(f"  Distinct statuses: {statuses}")
    print(f"  Grants: {grants}, Denies: {denies}")

    if grants == 0 or denies == 0:
        errors.append("Events have no variety (all grants or all denies)")

    conn.close()
    print(f"  RESULT: {'PASS' if event_count >= 100 else 'FAIL'}")

    # =========================================================================
    # TEST 2: Patterns accumulated with Bayesian weights
    # =========================================================================
    print("\n[TEST 2] Patterns Database")
    print("-" * 40)

    conn = sqlite3.connect(f"{db_dir}/patterns.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM patterns")
    pattern_count = cursor.fetchone()[0]
    print(f"  Total patterns: {pattern_count}")

    if pattern_count == 0:
        errors.append("No patterns in database")

    # Verify Bayesian calculation
    cursor.execute("""
        SELECT supporting_events, raw_weight
        FROM patterns
        WHERE supporting_events > 1
        LIMIT 10
    """)
    bayesian_correct = True
    print("\n  Bayesian weight verification:")
    print("  Observations | Actual | Expected | Match")
    print("  " + "-" * 45)

    for obs, actual_weight in cursor.fetchall():
        # Formula: (prior_mean * prior_strength + obs) / (prior_strength + obs)
        # Default: prior_mean=0.5, prior_strength=2
        expected_weight = (0.5 * 2 + obs) / (2 + obs)
        match = abs(actual_weight - expected_weight) < 0.0001
        if not match:
            bayesian_correct = False
        print(f"  {obs:11} | {actual_weight:.4f} | {expected_weight:.4f} | {'✓' if match else '✗'}")

    if not bayesian_correct:
        errors.append("Bayesian weight calculation incorrect")

    # Check weight variety
    cursor.execute("SELECT COUNT(DISTINCT ROUND(raw_weight, 2)) FROM patterns")
    distinct_weights = cursor.fetchone()[0]
    print(f"\n  Distinct weight values: {distinct_weights}")

    if distinct_weights <= 1:
        errors.append("All patterns have same weight (likely hardcoded)")

    conn.close()
    print(f"  RESULT: {'PASS' if bayesian_correct and distinct_weights > 1 else 'FAIL'}")

    # =========================================================================
    # TEST 3: Predictions with varied confidence
    # =========================================================================
    print("\n[TEST 3] Predictions Database")
    print("-" * 40)

    conn = sqlite3.connect(f"{db_dir}/calibration.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*),
               COUNT(DISTINCT ROUND(confidence, 2)),
               AVG(confidence),
               SUM(was_correct),
               AVG(was_correct)
        FROM predictions
        WHERE outcome_recorded = 1
    """)
    pred_count, distinct_conf, avg_conf, correct, accuracy = cursor.fetchone()

    print(f"  Total predictions: {pred_count}")
    print(f"  Distinct confidence values: {distinct_conf}")
    print(f"  Average confidence: {avg_conf:.3f}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.1%}")

    if pred_count == 0:
        errors.append("No predictions with outcomes")
    if distinct_conf <= 1:
        errors.append("All predictions have same confidence (likely hardcoded)")
    if avg_conf == 0.5:
        errors.append("Average confidence is exactly 0.5 (default value)")

    # Check confidence distribution
    cursor.execute("""
        SELECT ROUND(confidence, 2) as conf,
               COUNT(*) as n,
               SUM(was_correct) as correct
        FROM predictions
        WHERE outcome_recorded = 1
        GROUP BY ROUND(confidence, 2)
    """)
    print("\n  Confidence distribution:")
    print("  Conf  | Count | Correct | Accuracy")
    print("  " + "-" * 40)
    for conf, n, correct in cursor.fetchall():
        acc = correct / n if n > 0 else 0
        print(f"  {conf:.2f}  | {n:5} | {correct:7} | {acc:.0%}")

    print(f"  RESULT: {'PASS' if pred_count > 0 and distinct_conf > 1 else 'FAIL'}")

    # =========================================================================
    # TEST 4: Calibration factor computed
    # =========================================================================
    print("\n[TEST 4] Calibration Factor")
    print("-" * 40)

    cursor.execute("SELECT factor, sample_size FROM calibration_factors WHERE domain = 'judicial'")
    row = cursor.fetchone()

    if row:
        factor, sample_size = row
        print(f"  Stored factor: {factor:.4f}")
        print(f"  Sample size: {sample_size}")

        # Verify factor calculation
        expected_factor = accuracy / avg_conf if avg_conf > 0 else 1.0
        print(f"  Expected (accuracy/confidence): {expected_factor:.4f}")

        if abs(factor - 1.0) < 0.001:
            errors.append("Calibration factor is exactly 1.0 (never computed)")
        elif abs(factor - expected_factor) > 0.1:
            errors.append(f"Calibration factor mismatch: {factor:.3f} vs {expected_factor:.3f}")
    else:
        errors.append("No calibration factor in database")
        factor = None

    conn.close()
    print(f"  RESULT: {'PASS' if row and abs(factor - 1.0) > 0.001 else 'FAIL'}")

    # =========================================================================
    # TEST 5: System integration - new prediction uses calibration
    # =========================================================================
    print("\n[TEST 5] System Integration")
    print("-" * 40)

    try:
        from unified_system import EpistemicFlowControl, SystemConfig

        config = SystemConfig(db_dir=db_dir, domain="judicial")
        system = EpistemicFlowControl(config)

        # Get stored calibration factor
        stored_factor = system.calibration.get_calibration_factor("judicial")
        print(f"  Stored calibration factor: {stored_factor:.4f}")

        # Get patterns
        patterns = system.get_patterns_for_subject("Judge Rodney Gilstrap", min_confidence=0.1)
        print(f"  Patterns for Gilstrap: {len(patterns)}")

        if patterns:
            pattern_ids = [p["pattern_id"] for p in patterns[:3]]

            # Make prediction
            pred = system.make_prediction(
                prediction_type="ruling",
                predicted_value="Test verification prediction",
                context={"test": True},
                source_patterns=pattern_ids,
                stakes="high"
            )

            print(f"  Raw confidence: {pred['raw_confidence']:.4f}")
            print(f"  Calibrated confidence: {pred['calibrated_confidence']:.4f}")
            print(f"  Gate decision: {pred['gate_decision']}")

            # Verify calibration applied
            expected_calibrated = pred['raw_confidence'] * stored_factor
            calibration_applied = abs(pred['calibrated_confidence'] - expected_calibrated) < 0.01

            print(f"  Expected calibrated: {expected_calibrated:.4f}")
            print(f"  Calibration applied: {'YES' if calibration_applied else 'NO'}")

            if not calibration_applied:
                errors.append("Calibration factor not applied to new predictions")

            # Verify gate decision makes sense
            # Below 0.7 should be blocked or review, not auto_pass
            if pred['calibrated_confidence'] < 0.7 and pred['gate_decision'] == 'auto_pass':
                errors.append("Gate decision doesn't match low confidence")

        print(f"  RESULT: {'PASS' if patterns and calibration_applied else 'FAIL'}")

    except Exception as e:
        errors.append(f"System integration error: {e}")
        print(f"  RESULT: FAIL ({e})")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    if errors:
        print(f"\nFAILED - {len(errors)} error(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        sys.exit(1)
    else:
        print("\nPASSED - All verifications successful")
        print("\nThe system is computing real values:")
        print("  - Events stored with variety")
        print("  - Patterns accumulated with correct Bayesian weights")
        print("  - Predictions have varied confidence (not hardcoded)")
        print("  - Calibration factor computed from outcomes")
        print("  - New predictions apply calibration correctly")
        sys.exit(0)


if __name__ == "__main__":
    main()
