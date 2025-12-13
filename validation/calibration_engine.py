"""
Validation Engine - Calibration and Accuracy Tracking

This component answers the key question:
"When we say we're 80% confident, are we right 80% of the time?"

CALIBRATION EXPLAINED:
----------------------
A well-calibrated system has confidence scores that match actual accuracy.
- Predictions at 90% confidence should be correct 90% of the time
- Predictions at 60% confidence should be correct 60% of the time

If they don't match, we need to RECALIBRATE.

TRAINING DATA REQUIREMENTS:
---------------------------
1. PREDICTION_OUTCOMES: Need predictions with actual outcomes
   - What did we predict?
   - What confidence did we assign?
   - Was it actually correct?

   COLLECTION: Record every prediction and later record its outcome

2. CALIBRATION_CURVES: Need enough data to measure calibration
   - At least 50 predictions per confidence bucket
   - Cover the full range of confidence levels

   COLLECTION: Systematic outcome recording over time

3. DOMAIN_SPECIFIC_CALIBRATION: Different domains may need different calibration
   - Legal predictions vs market predictions vs medical predictions

   COLLECTION: Segment predictions by domain, calibrate separately
"""

import json
import sqlite3
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics


@dataclass
class Prediction:
    """
    A prediction made by the system with tracking for validation.
    """
    prediction_id: str
    prediction_type: str            # "pattern", "outcome", "recommendation"
    domain: str

    # What we predicted
    predicted_value: Any            # The actual prediction
    confidence: float               # Our stated confidence (0-1)

    # Context
    context: Dict[str, Any]         # Information available at prediction time
    model_version: str              # Which version of the model made this
    source_patterns: List[str]      # Pattern IDs used for this prediction

    # Timing
    predicted_at: datetime = field(default_factory=datetime.now)
    outcome_deadline: Optional[datetime] = None  # When we can measure outcome

    # Outcome (filled in later)
    outcome_recorded: bool = False
    actual_value: Any = None
    was_correct: Optional[bool] = None
    outcome_recorded_at: Optional[datetime] = None
    outcome_notes: str = ""


@dataclass
class CalibrationBucket:
    """
    A bucket for measuring calibration at a confidence level.
    """
    confidence_lower: float
    confidence_upper: float
    predictions: int = 0
    correct: int = 0

    def accuracy(self) -> Optional[float]:
        """Actual accuracy in this bucket."""
        if self.predictions == 0:
            return None
        return self.correct / self.predictions

    def expected_accuracy(self) -> float:
        """Expected accuracy (midpoint of bucket)."""
        return (self.confidence_lower + self.confidence_upper) / 2

    def calibration_error(self) -> Optional[float]:
        """Difference between actual and expected accuracy."""
        actual = self.accuracy()
        if actual is None:
            return None
        return actual - self.expected_accuracy()


class CalibrationEngine:
    """
    Engine for tracking prediction accuracy and calibration.

    This is the "truth verification" layer of the system.
    """

    def __init__(self, db_path: str = "calibration.db"):
        self.db_path = db_path
        self._init_db()

        # Calibration factors by domain (learned from data)
        self.calibration_factors: Dict[str, float] = {}

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                prediction_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                predicted_value TEXT NOT NULL,
                confidence REAL NOT NULL,
                context TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source_patterns TEXT NOT NULL,
                predicted_at TEXT NOT NULL,
                outcome_deadline TEXT,
                outcome_recorded INTEGER DEFAULT 0,
                actual_value TEXT,
                was_correct INTEGER,
                outcome_recorded_at TEXT,
                outcome_notes TEXT
            )
        """)

        # Calibration history (periodic snapshots)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                snapshot_at TEXT NOT NULL,
                bucket_data TEXT NOT NULL,
                overall_calibration_error REAL,
                sample_size INTEGER,
                notes TEXT
            )
        """)

        # Calibration factors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_factors (
                domain TEXT PRIMARY KEY,
                factor REAL NOT NULL,
                computed_at TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                notes TEXT
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_domain ON predictions(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_outcome ON predictions(outcome_recorded)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_confidence ON predictions(confidence)")

        conn.commit()
        conn.close()

    # ========== PREDICTION RECORDING ==========

    def record_prediction(self, prediction: Prediction) -> bool:
        """
        Record a new prediction for future validation.

        Every prediction we make gets recorded so we can measure accuracy.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO predictions
                (prediction_id, prediction_type, domain, predicted_value, confidence,
                 context, model_version, source_patterns, predicted_at, outcome_deadline,
                 outcome_recorded, actual_value, was_correct, outcome_recorded_at, outcome_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.prediction_id,
                prediction.prediction_type,
                prediction.domain,
                json.dumps(prediction.predicted_value),
                prediction.confidence,
                json.dumps(prediction.context),
                prediction.model_version,
                json.dumps(prediction.source_patterns),
                prediction.predicted_at.isoformat(),
                prediction.outcome_deadline.isoformat() if prediction.outcome_deadline else None,
                0, None, None, None, ""
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def record_outcome(
        self,
        prediction_id: str,
        actual_value: Any,
        was_correct: bool,
        notes: str = ""
    ) -> bool:
        """
        Record the actual outcome of a prediction.

        THIS IS THE KEY TRAINING DATA:
        Comparing predicted vs actual lets us measure calibration.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()

        cursor.execute("""
            UPDATE predictions
            SET outcome_recorded = 1,
                actual_value = ?,
                was_correct = ?,
                outcome_recorded_at = ?,
                outcome_notes = ?
            WHERE prediction_id = ?
        """, (
            json.dumps(actual_value),
            1 if was_correct else 0,
            now.isoformat(),
            notes,
            prediction_id
        ))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    # ========== CALIBRATION ANALYSIS ==========

    def compute_calibration(
        self,
        domain: Optional[str] = None,
        num_buckets: int = 10,
        min_samples_per_bucket: int = 5
    ) -> Dict[str, Any]:
        """
        Compute calibration metrics.

        Returns:
        - Calibration curve (expected vs actual accuracy per bucket)
        - Expected Calibration Error (ECE)
        - Calibration factor (multiply confidence by this)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT confidence, was_correct
            FROM predictions
            WHERE outcome_recorded = 1
        """
        params = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < min_samples_per_bucket * num_buckets:
            return {
                "error": f"Insufficient data: {len(rows)} samples, need {min_samples_per_bucket * num_buckets}",
                "sample_size": len(rows)
            }

        # Create buckets
        bucket_size = 1.0 / num_buckets
        buckets: Dict[int, CalibrationBucket] = {}

        for i in range(num_buckets):
            lower = i * bucket_size
            upper = (i + 1) * bucket_size
            buckets[i] = CalibrationBucket(
                confidence_lower=lower,
                confidence_upper=upper
            )

        # Assign predictions to buckets
        for confidence, was_correct in rows:
            bucket_idx = min(int(confidence / bucket_size), num_buckets - 1)
            buckets[bucket_idx].predictions += 1
            if was_correct:
                buckets[bucket_idx].correct += 1

        # Compute calibration curve
        calibration_curve = []
        total_weighted_error = 0.0
        total_samples = 0

        for i, bucket in buckets.items():
            if bucket.predictions >= min_samples_per_bucket:
                expected = bucket.expected_accuracy()
                actual = bucket.accuracy()
                error = bucket.calibration_error()

                calibration_curve.append({
                    "confidence_range": [bucket.confidence_lower, bucket.confidence_upper],
                    "expected_accuracy": expected,
                    "actual_accuracy": actual,
                    "calibration_error": error,
                    "sample_size": bucket.predictions
                })

                # Weighted contribution to ECE
                total_weighted_error += abs(error) * bucket.predictions
                total_samples += bucket.predictions

        # Expected Calibration Error (ECE)
        ece = total_weighted_error / total_samples if total_samples > 0 else None

        # Compute calibration factor
        # If actual accuracy is consistently lower than confidence, factor < 1
        if calibration_curve:
            total_confidence = sum(b["expected_accuracy"] * b["sample_size"] for b in calibration_curve)
            total_actual = sum(b["actual_accuracy"] * b["sample_size"] for b in calibration_curve)
            calibration_factor = total_actual / total_confidence if total_confidence > 0 else 1.0
        else:
            calibration_factor = 1.0

        return {
            "domain": domain or "all",
            "sample_size": len(rows),
            "calibration_curve": calibration_curve,
            "expected_calibration_error": ece,
            "calibration_factor": calibration_factor,
            "interpretation": self._interpret_calibration(ece, calibration_factor)
        }

    def _interpret_calibration(self, ece: Optional[float], factor: float) -> str:
        """Interpret calibration results for humans."""
        if ece is None:
            return "Insufficient data for calibration analysis"

        interpretation = []

        # ECE interpretation
        if ece < 0.02:
            interpretation.append("Excellent calibration (ECE < 2%)")
        elif ece < 0.05:
            interpretation.append("Good calibration (ECE < 5%)")
        elif ece < 0.10:
            interpretation.append("Moderate calibration issues (ECE 5-10%)")
        else:
            interpretation.append("Significant calibration problems (ECE > 10%)")

        # Factor interpretation
        if factor < 0.8:
            interpretation.append("System is OVERCONFIDENT - multiply confidence by {:.2f}".format(factor))
        elif factor > 1.2:
            interpretation.append("System is UNDERCONFIDENT - multiply confidence by {:.2f}".format(factor))
        else:
            interpretation.append("Confidence levels are reasonable (factor ~1.0)")

        return "; ".join(interpretation)

    def save_calibration_snapshot(self, domain: str, notes: str = "") -> bool:
        """
        Save a calibration snapshot for historical tracking.

        Run this periodically to track calibration over time.
        """
        calibration = self.compute_calibration(domain)

        if "error" in calibration:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO calibration_snapshots
            (domain, snapshot_at, bucket_data, overall_calibration_error, sample_size, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            domain,
            datetime.now().isoformat(),
            json.dumps(calibration["calibration_curve"]),
            calibration["expected_calibration_error"],
            calibration["sample_size"],
            notes
        ))

        # Update calibration factor
        cursor.execute("""
            INSERT OR REPLACE INTO calibration_factors
            (domain, factor, computed_at, sample_size, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (
            domain,
            calibration["calibration_factor"],
            datetime.now().isoformat(),
            calibration["sample_size"],
            calibration["interpretation"]
        ))

        conn.commit()
        conn.close()

        # Update in-memory factor
        self.calibration_factors[domain] = calibration["calibration_factor"]

        return True

    def get_calibration_factor(self, domain: str) -> float:
        """Get the calibration factor for a domain."""
        if domain in self.calibration_factors:
            return self.calibration_factors[domain]

        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT factor FROM calibration_factors WHERE domain = ?",
            (domain,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            self.calibration_factors[domain] = row[0]
            return row[0]

        return 1.0  # Default: no calibration adjustment

    def apply_calibration(self, raw_confidence: float, domain: str) -> float:
        """
        Apply calibration factor to a raw confidence score.

        Use this when making predictions to get calibrated confidence.
        """
        factor = self.get_calibration_factor(domain)
        return min(1.0, max(0.0, raw_confidence * factor))

    # ========== ACCURACY ANALYSIS ==========

    def compute_accuracy_by_type(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Compute accuracy broken down by prediction type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT prediction_type, COUNT(*) as total,
                   SUM(was_correct) as correct
            FROM predictions
            WHERE outcome_recorded = 1
        """
        params = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " GROUP BY prediction_type"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = {}
        for pred_type, total, correct in rows:
            results[pred_type] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0
            }

        return results

    def compute_accuracy_over_time(
        self,
        domain: Optional[str] = None,
        bucket_days: int = 30
    ) -> List[Dict]:
        """Compute accuracy trend over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT predicted_at, was_correct
            FROM predictions
            WHERE outcome_recorded = 1
        """
        params = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY predicted_at"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Bucket by time period
        results = []
        current_bucket_start = None
        current_bucket = {"total": 0, "correct": 0}

        for predicted_at_str, was_correct in rows:
            predicted_at = datetime.fromisoformat(predicted_at_str)

            if current_bucket_start is None:
                current_bucket_start = predicted_at

            # Check if we need a new bucket
            if (predicted_at - current_bucket_start).days >= bucket_days:
                if current_bucket["total"] > 0:
                    results.append({
                        "period_start": current_bucket_start.isoformat(),
                        "total": current_bucket["total"],
                        "correct": current_bucket["correct"],
                        "accuracy": current_bucket["correct"] / current_bucket["total"]
                    })

                current_bucket_start = predicted_at
                current_bucket = {"total": 0, "correct": 0}

            current_bucket["total"] += 1
            if was_correct:
                current_bucket["correct"] += 1

        # Add final bucket
        if current_bucket["total"] > 0:
            results.append({
                "period_start": current_bucket_start.isoformat(),
                "total": current_bucket["total"],
                "correct": current_bucket["correct"],
                "accuracy": current_bucket["correct"] / current_bucket["total"]
            })

        return results

    # ========== PENDING OUTCOME TRACKING ==========

    def get_predictions_needing_outcomes(
        self,
        domain: Optional[str] = None,
        max_items: int = 50
    ) -> List[Dict]:
        """
        Get predictions that need outcome recording.

        HUMAN TASK: Record what actually happened.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT prediction_id, prediction_type, domain, predicted_value,
                   confidence, context, predicted_at, outcome_deadline
            FROM predictions
            WHERE outcome_recorded = 0
        """
        params = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY predicted_at ASC LIMIT ?"
        params.append(max_items)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "prediction_id": row[0],
                "prediction_type": row[1],
                "domain": row[2],
                "predicted_value": json.loads(row[3]),
                "confidence": row[4],
                "context": json.loads(row[5]),
                "predicted_at": row[6],
                "outcome_deadline": row[7],
                "days_since_prediction": (datetime.now() - datetime.fromisoformat(row[6])).days
            }
            for row in rows
        ]

    # ========== STATISTICS ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get overall validation statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE outcome_recorded = 1")
        with_outcomes = cursor.fetchone()[0]

        cursor.execute("""
            SELECT domain, COUNT(*) as total, SUM(was_correct) as correct
            FROM predictions
            WHERE outcome_recorded = 1
            GROUP BY domain
        """)
        by_domain = {
            row[0]: {"total": row[1], "correct": row[2], "accuracy": row[2]/row[1] if row[1] > 0 else 0}
            for row in cursor.fetchall()
        }

        cursor.execute("""
            SELECT AVG(confidence) FROM predictions WHERE outcome_recorded = 1 AND was_correct = 1
        """)
        avg_confidence_correct = cursor.fetchone()[0]

        cursor.execute("""
            SELECT AVG(confidence) FROM predictions WHERE outcome_recorded = 1 AND was_correct = 0
        """)
        avg_confidence_incorrect = cursor.fetchone()[0]

        conn.close()

        return {
            "total_predictions": total_predictions,
            "with_outcomes": with_outcomes,
            "pending_outcomes": total_predictions - with_outcomes,
            "by_domain": by_domain,
            "avg_confidence_when_correct": avg_confidence_correct,
            "avg_confidence_when_incorrect": avg_confidence_incorrect
        }


# ========== TRAINING DATA COLLECTION ==========

def create_outcome_recording_task(predictions: List[Dict]) -> Dict:
    """
    Create a task for recording prediction outcomes.

    HUMAN TASK: Look at what we predicted and record what actually happened.
    """
    return {
        "task_type": "outcome_recording",
        "instructions": """
            For each prediction, determine the actual outcome:

            1. Look up what actually happened
            2. Determine if the prediction was CORRECT or INCORRECT
            3. If partially correct, lean toward INCORRECT (we want conservative calibration)
            4. Add any notes that would help understand why it was right/wrong

            IMPORTANT: Be honest about outcomes. Incorrect predictions are valuable
            data for improving calibration.
        """,
        "predictions": predictions,
        "output_format": {
            "prediction_id": "str",
            "actual_value": "any",
            "was_correct": "bool",
            "notes": "str",
            "confidence_was_appropriate": "yes | too_high | too_low"
        }
    }


def create_calibration_review_task(calibration_data: Dict) -> Dict:
    """
    Create a task for human review of calibration analysis.

    HUMAN TASK: Review calibration and recommend adjustments.
    """
    return {
        "task_type": "calibration_review",
        "instructions": """
            Review the calibration analysis for this domain.

            Questions to answer:
            1. Is the calibration factor reasonable?
            2. Are there specific confidence ranges that are especially problematic?
            3. Should we adjust thresholds based on this data?
            4. Are there prediction types that need separate calibration?

            Recommendations should be actionable.
        """,
        "calibration_data": calibration_data,
        "output_format": {
            "recommended_factor": "float",
            "problematic_ranges": [{"lower": "float", "upper": "float", "issue": "str"}],
            "threshold_recommendations": {
                "auto_pass": "float",
                "review": "float",
                "block": "float"
            },
            "other_recommendations": "str"
        }
    }


if __name__ == "__main__":
    # Demo
    engine = CalibrationEngine("demo_calibration.db")

    # Record some predictions
    for i in range(100):
        confidence = 0.5 + (i % 50) / 100  # Range from 0.5 to 1.0

        # Simulate: higher confidence = more likely correct
        # But with some noise to make it interesting
        import random
        random.seed(i)
        was_correct = random.random() < (confidence * 0.9)  # Slightly overconfident

        pred = Prediction(
            prediction_id=f"pred_{i:03d}",
            prediction_type="pattern",
            domain="judicial",
            predicted_value={"pattern": f"test_pattern_{i}"},
            confidence=confidence,
            context={"test": True},
            model_version="v1.0",
            source_patterns=[]
        )

        engine.record_prediction(pred)
        engine.record_outcome(
            prediction_id=pred.prediction_id,
            actual_value={"result": was_correct},
            was_correct=was_correct,
            notes="Demo outcome"
        )

    # Compute calibration
    print("Calibration Analysis:")
    calibration = engine.compute_calibration(domain="judicial")
    print(f"  Sample size: {calibration['sample_size']}")
    print(f"  ECE: {calibration['expected_calibration_error']:.3f}")
    print(f"  Calibration factor: {calibration['calibration_factor']:.3f}")
    print(f"  Interpretation: {calibration['interpretation']}")

    print("\nCalibration Curve:")
    for bucket in calibration.get("calibration_curve", []):
        print(f"  Confidence {bucket['confidence_range'][0]:.1f}-{bucket['confidence_range'][1]:.1f}: "
              f"Expected {bucket['expected_accuracy']:.1%}, Actual {bucket['actual_accuracy']:.1%}")

    print(f"\nOverall stats: {engine.get_stats()}")
