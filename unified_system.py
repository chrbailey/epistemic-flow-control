"""
Unified Epistemic Flow Control System

This integrates all components into a coherent system where:
1. Events flow in as ground truth
2. Patterns are extracted (LLM + human validation)
3. Patterns are stored with Bayesian weights
4. Predictions flow through human gates
5. Outcomes validate calibration
6. Training data improves the system

THE WATER IN SAND METAPHOR:
---------------------------
- WATER: LLM-generated probabilistic output (patterns, predictions)
- SAND: Structured domain (events, databases, records)
- CHANNELS: Human gates that control flow
- HUMAN ROLE: Open/close channels, adjust flow, build new paths

The human doesn't create the water. The human controls where it flows.
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Import all components
from core.event_store import (
    EventStore, Event, Source, SourceType, VerificationStatus
)
from core.pattern_extractor import (
    PatternExtractor, ExtractedPattern, PatternType, ExtractionSource
)
from core.pattern_database import (
    PatternDatabase, StoredPattern, PatternPrior,
    wilson_score_lower, bayesian_update, temporal_decay
)
from gates.review_gate import (
    ReviewGate, GatedItem, GateDecision, Reviewer, ReviewerRole
)
from validation.calibration_engine import (
    CalibrationEngine, Prediction
)
from training.data_generator import (
    TrainingDataGenerator, LabelingTask, TaskType
)

# Import new modules
from normalizers import JudgeNormalizer, LawyerNormalizer
from concentration import HHICalculator, SPOFDetector
from drift import EmbeddingTracker, DriftDetector
from jurisdictions import JurisdictionalContext, NDCalContext, AlsupContext


@dataclass
class SystemConfig:
    """Configuration for the unified system."""
    db_dir: str = "./data"
    domain: str = "default"

    # NEW: Jurisdiction context
    jurisdiction: Optional[str] = None  # e.g., "nd_cal"
    judge: Optional[str] = None         # e.g., "alsup"

    # Gate thresholds (can be calibrated)
    auto_pass_threshold: float = 0.92
    review_threshold: float = 0.70
    block_threshold: float = 0.50

    # Decay settings
    pattern_half_life_days: float = 180.0

    # Calibration settings
    min_calibration_samples: int = 50


class EpistemicFlowControl:
    """
    The unified system for human-gated probabilistic intelligence.

    ARCHITECTURE:
    -------------
    Events → Pattern Extraction → Pattern Database → Predictions
                    ↓                    ↓                ↓
            Human Validation      Human Override    Human Review Gate
                    ↓                    ↓                ↓
            Calibration ←←←←←←←←←←←← Outcomes
    """

    def __init__(self, config: SystemConfig):
        self.config = config

        # Ensure data directory exists
        Path(config.db_dir).mkdir(parents=True, exist_ok=True)

        # Initialize all components
        self.event_store = EventStore(
            db_path=f"{config.db_dir}/events.db"
        )

        self.pattern_extractor = PatternExtractor(
            event_store=self.event_store
        )

        self.pattern_db = PatternDatabase(
            db_path=f"{config.db_dir}/patterns.db",
            default_half_life_days=config.pattern_half_life_days
        )

        self.review_gate = ReviewGate(
            db_path=f"{config.db_dir}/gate.db"
        )

        self.calibration = CalibrationEngine(
            db_path=f"{config.db_dir}/calibration.db"
        )

        self.training_data = TrainingDataGenerator(
            db_path=f"{config.db_dir}/training.db"
        )

        # NEW: Initialize new components
        self.judge_normalizer = JudgeNormalizer()
        self.lawyer_normalizer = LawyerNormalizer()
        self.hhi_calculator = HHICalculator()
        self.spof_detector = SPOFDetector()
        self.embedding_tracker = EmbeddingTracker()
        self.drift_detector = DriftDetector()

        # NEW: Load jurisdiction context
        self.jurisdiction_context: Optional[JurisdictionalContext] = None
        self._load_jurisdiction_context()

    def _load_jurisdiction_context(self):
        """Load the appropriate jurisdictional context based on config."""
        if self.config.judge == "alsup":
            self.jurisdiction_context = AlsupContext()
        elif self.config.jurisdiction == "nd_cal":
            self.jurisdiction_context = NDCalContext()
        # Add more jurisdictions as needed

    # ========== JURISDICTION CONTEXT ==========

    def get_jurisdiction_context(self) -> Optional[Dict]:
        """Get the current jurisdictional context summary."""
        if not self.jurisdiction_context:
            return None
        return self.jurisdiction_context.get_context_summary()

    def get_format_requirements(self) -> List[Dict]:
        """Get formatting requirements for the current jurisdiction."""
        if not self.jurisdiction_context:
            return []
        return [
            {"name": r.name, "value": r.value, "mandatory": r.is_mandatory, "notes": r.notes}
            for r in self.jurisdiction_context.get_format_requirements()
        ]

    def get_procedural_rules(self, motion_type: Optional[str] = None) -> List[Dict]:
        """Get procedural rules for the current jurisdiction."""
        if not self.jurisdiction_context:
            return []

        from jurisdictions.base import MotionType

        if motion_type:
            try:
                mt = MotionType(motion_type)
                rules = self.jurisdiction_context.get_rules_for_motion(mt)
            except ValueError:
                rules = self.jurisdiction_context.get_procedural_rules()
        else:
            rules = self.jurisdiction_context.get_procedural_rules()

        return [
            {
                "rule_id": r.rule_id,
                "title": r.title,
                "description": r.description,
                "source": r.source,
                "mandatory": r.is_mandatory
            }
            for r in rules
        ]

    # ========== ENTITY NORMALIZATION ==========

    def normalize_judge(self, raw_input: str) -> Dict:
        """Normalize a judge name from various formats."""
        result = self.judge_normalizer.normalize(raw_input)
        return {
            "raw_input": result.raw_input,
            "normalized_name": result.normalized_name,
            "source_type": result.source_type.value,
            "confidence": result.confidence,
            "first_name": result.first_name,
            "last_name": result.last_name,
            "suffix": result.suffix
        }

    def validate_lawyer(self, raw_input: str) -> Dict:
        """Validate and normalize a lawyer entity name."""
        result = self.lawyer_normalizer.validate(raw_input)
        return {
            "raw_input": result.raw_input,
            "normalized_name": result.normalized_name,
            "is_valid": result.is_valid,
            "rejection_reason": result.rejection_reason.value if result.rejection_reason else None,
            "confidence": result.confidence
        }

    # ========== CONCENTRATION ANALYSIS ==========

    def analyze_concentration(
        self,
        entity_counts: Dict[str, int],
        entity_type: str = "entity"
    ) -> Dict:
        """
        Analyze concentration risk in entity distribution.

        Returns HHI metrics and SPOF risk assessment.
        """
        hhi_result = self.hhi_calculator.from_counts(entity_counts)
        spof_assessment = self.spof_detector.analyze(
            entity_counts,
            entity_type=entity_type,
            domain=self.config.domain
        )

        return {
            "hhi": hhi_result.hhi,
            "concentration_level": hhi_result.level.value,
            "top_entity": hhi_result.top_entity,
            "top_share": hhi_result.top_share,
            "equivalent_firms": hhi_result.equivalent_firms,
            "is_healthy": hhi_result.is_healthy,
            "spof_risks": [
                {
                    "entity_id": r.entity_id,
                    "share": r.share,
                    "risk_level": r.risk_level.value,
                    "is_spof": r.is_spof,
                    "recommendation": r.recommendation
                }
                for r in spof_assessment.spof_risks[:5]  # Top 5 risks
            ],
            "overall_health": spof_assessment.overall_health,
            "has_critical_spof": spof_assessment.has_critical_spof
        }

    # ========== DRIFT DETECTION ==========

    def check_pattern_drift(
        self,
        entity_id: str,
        pattern_type: str,
        current_metrics: Dict
    ) -> Dict:
        """
        Check for drift in a pattern compared to baseline.

        Args:
            entity_id: ID of the entity (judge, lawyer, etc.)
            pattern_type: Type of pattern to check
            current_metrics: Current metric values to compare against baseline

        Returns:
            Drift analysis including severity and recommendations
        """
        # Generate embedding from current metrics
        current_embedding = self.embedding_tracker.generate(
            entity_id=entity_id,
            pattern_type=pattern_type,
            metrics=current_metrics,
            sample_count=current_metrics.get("sample_count", 0)
        )

        # Check for drift
        drift_event = self.drift_detector.detect_drift(current_embedding)

        return {
            "entity_id": drift_event.entity_id,
            "pattern_type": drift_event.pattern_type,
            "drift_type": drift_event.drift_type.value,
            "severity": drift_event.severity.value,
            "baseline_similarity": drift_event.baseline_similarity,
            "drift_percentage": drift_event.drift_percentage,
            "confidence_impact": drift_event.confidence_impact,
            "requires_recalibration": drift_event.requires_recalibration,
            "recommendation": drift_event.recommendation,
            "top_changed_dimensions": drift_event.top_changed_dimensions
        }

    def set_pattern_baseline(
        self,
        entity_id: str,
        pattern_type: str,
        metrics: Dict,
        sample_count: int = 0
    ) -> bool:
        """Set a baseline for drift comparison."""
        embedding = self.embedding_tracker.generate(
            entity_id=entity_id,
            pattern_type=pattern_type,
            metrics=metrics,
            sample_count=sample_count
        )
        self.drift_detector.set_baseline(embedding)
        return True

    # ========== EVENT INGESTION ==========

    def ingest_event(
        self,
        what: str,
        who: List[str],
        when: datetime,
        where: str,
        source_id: str,
        raw_text: str,
        why: Optional[str] = None,
        how: Optional[str] = None,
        event_type: str = "general",
        auto_extract_patterns: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a new event into the system.

        This is the primary entry point for new information.

        Returns:
        - event_id: The created event ID
        - patterns_extracted: Any patterns automatically extracted
        - needs_human_review: Whether human review is needed
        """
        # Generate unique event ID
        import random as _rand
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand.randint(0, 999999):06d}"

        # Get source reliability to set verification status
        source = self.event_store.get_source(source_id)
        if source:
            reliability = source.effective_reliability()
            if reliability >= 0.95:
                verification = VerificationStatus.VERIFIED
            elif reliability >= 0.80:
                verification = VerificationStatus.HIGH_CONFIDENCE
            elif reliability >= 0.60:
                verification = VerificationStatus.MEDIUM_CONFIDENCE
            else:
                verification = VerificationStatus.LOW_CONFIDENCE
        else:
            verification = VerificationStatus.UNVERIFIED

        # Create event
        event = Event(
            event_id=event_id,
            who=who,
            what=what,
            when=when,
            where=where,
            why=why,
            how=how,
            source_id=source_id,
            source_url=None,
            raw_text=raw_text,
            verification_status=verification,
            domain=self.config.domain,
            event_type=event_type
        )

        # Store event
        success, msg = self.event_store.add_event(event)

        result = {
            "success": success,
            "event_id": event_id if success else None,
            "message": msg,
            "verification_status": verification.value,
            "patterns_extracted": [],
            "needs_human_review": verification.value in ["unverified", "low"]
        }

        # Auto-extract patterns if enabled
        if success and auto_extract_patterns:
            patterns = self.pattern_extractor.extract_patterns([event])
            for pattern in patterns:
                # Store pattern
                stored_id = self.pattern_db.store_pattern(pattern)
                result["patterns_extracted"].append({
                    "pattern_id": stored_id,
                    "description": pattern.description,
                    "confidence": pattern.effective_confidence,
                    "needs_validation": not pattern.human_validated
                })

        return result

    # ========== PATTERN MANAGEMENT ==========

    def get_patterns_for_subject(
        self,
        subject: str,
        min_confidence: float = 0.5,
        include_stale: bool = False
    ) -> List[Dict]:
        """
        Get all patterns for a subject (e.g., a judge, company, etc.)

        Returns patterns with their current weights and confidence.
        """
        patterns = self.pattern_db.query_patterns(
            subject=subject,
            domain=self.config.domain,
            min_weight=min_confidence,
            include_stale=include_stale
        )

        return [
            {
                "pattern_id": p.pattern_id,
                "subject": p.subject,
                "description": p.description,
                "pattern_type": p.pattern_type.value,
                "weight": p.effective_weight(),
                "raw_weight": p.raw_weight,
                "supporting_events": p.supporting_events,
                "total_observations": p.total_observations,
                "last_observed": p.last_observed.isoformat(),
                "human_override": p.human_override,
                "confidence_interval": [
                    wilson_score_lower(p.supporting_events, p.total_observations),
                    p.raw_weight  # Upper bound approximation
                ]
            }
            for p in patterns
        ]

    def human_override_pattern(
        self,
        pattern_id: str,
        new_weight: float,
        reason: str,
        overrider: str
    ) -> bool:
        """
        Human override of a pattern weight.

        THE KEY GATE: This is where human judgment supersedes the model.
        """
        return self.pattern_db.human_override_weight(
            pattern_id=pattern_id,
            new_weight=new_weight,
            reason=reason,
            overrider=overrider
        )

    # ========== PREDICTIONS ==========

    def make_prediction(
        self,
        prediction_type: str,
        predicted_value: Any,
        context: Dict[str, Any],
        source_patterns: List[str],
        stakes: str = "medium"
    ) -> Dict[str, Any]:
        """
        Make a prediction and route it through the review gate.

        Returns the gate decision and prediction ID.
        """
        # Calculate confidence from source patterns
        patterns = [self.pattern_db.get_pattern(pid) for pid in source_patterns]
        patterns = [p for p in patterns if p is not None]

        if patterns:
            # Combine pattern weights (geometric mean)
            import math
            raw_confidence = math.exp(
                sum(math.log(max(0.01, p.effective_weight())) for p in patterns)
                / len(patterns)
            )
        else:
            raw_confidence = 0.5  # No patterns, neutral confidence

        # Apply calibration
        calibrated_confidence = self.calibration.apply_calibration(
            raw_confidence, self.config.domain
        )

        # Create prediction with unique ID
        import random as _rand
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand.randint(0, 999999):06d}"

        prediction = Prediction(
            prediction_id=prediction_id,
            prediction_type=prediction_type,
            domain=self.config.domain,
            predicted_value=predicted_value,
            confidence=calibrated_confidence,
            context=context,
            model_version="v1.0",
            source_patterns=source_patterns
        )

        # Record prediction
        self.calibration.record_prediction(prediction)

        # Create gated item
        gated_item = GatedItem(
            item_id=prediction_id,
            item_type="prediction",
            content={
                "predicted_value": predicted_value,
                "context": context,
                "source_patterns": source_patterns
            },
            confidence=calibrated_confidence,
            domain=self.config.domain,
            stakes=stakes
        )

        # Process through gate
        gate_decision = self.review_gate.process(gated_item)

        return {
            "prediction_id": prediction_id,
            "predicted_value": predicted_value,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "gate_decision": gate_decision.value,
            "gate_reasoning": gated_item.gate_reasoning,
            "needs_human_review": gate_decision in [
                GateDecision.REVIEW_REQUIRED,
                GateDecision.BLOCKED,
                GateDecision.ESCALATED
            ]
        }

    def record_prediction_outcome(
        self,
        prediction_id: str,
        actual_value: Any,
        was_correct: bool,
        notes: str = ""
    ) -> bool:
        """
        Record the actual outcome of a prediction.

        TRAINING DATA: This is how we learn if our predictions are calibrated.
        """
        # Record in calibration engine
        self.calibration.record_outcome(
            prediction_id=prediction_id,
            actual_value=actual_value,
            was_correct=was_correct,
            notes=notes
        )

        # Record in review gate
        self.review_gate.record_outcome(
            item_id=prediction_id,
            was_correct=was_correct,
            notes=notes
        )

        return True

    # ========== HUMAN REVIEW INTERFACE ==========

    def get_items_needing_review(
        self,
        reviewer_id: Optional[str] = None,
        max_items: int = 20
    ) -> List[Dict]:
        """
        Get items that need human review.

        THE REVIEW QUEUE: This is where humans control the flow.
        """
        items = self.review_gate.get_review_queue(
            reviewer_id=reviewer_id,
            domain=self.config.domain,
            max_items=max_items
        )

        return [
            {
                "item_id": item.item_id,
                "item_type": item.item_type,
                "content": item.content,
                "confidence": item.confidence,
                "stakes": item.stakes,
                "gate_decision": item.gate_decision.value if item.gate_decision else None,
                "reasoning": item.gate_reasoning
            }
            for item in items
        ]

    def submit_human_review(
        self,
        item_id: str,
        reviewer_id: str,
        decision: str,  # "approve", "reject", "modify"
        notes: str,
        modified_content: Optional[Dict] = None
    ) -> bool:
        """
        Submit a human review decision.

        THE GATE OPERATION: Human opens or closes the channel.
        """
        return self.review_gate.submit_review(
            item_id=item_id,
            reviewer_id=reviewer_id,
            decision=decision,
            notes=notes,
            modified_content=modified_content
        )

    # ========== CALIBRATION ==========

    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status and recommendations."""
        calibration = self.calibration.compute_calibration(domain=self.config.domain)

        return {
            "domain": self.config.domain,
            "calibration_data": calibration,
            "current_factor": self.calibration.get_calibration_factor(self.config.domain),
            "recommendations": self._get_calibration_recommendations(calibration)
        }

    def _get_calibration_recommendations(self, calibration: Dict) -> List[str]:
        """Generate recommendations based on calibration data."""
        recommendations = []

        if "error" in calibration:
            recommendations.append(
                f"Need more data: {calibration.get('sample_size', 0)} samples, "
                f"need {self.config.min_calibration_samples}"
            )
            return recommendations

        factor = calibration.get("calibration_factor", 1.0)
        ece = calibration.get("expected_calibration_error", 0)

        if factor < 0.85:
            recommendations.append(
                f"System is overconfident. Apply calibration factor of {factor:.2f}"
            )
        elif factor > 1.15:
            recommendations.append(
                f"System is underconfident. Consider raising confidence scores"
            )

        if ece > 0.10:
            recommendations.append(
                f"High calibration error ({ece:.2%}). Review extraction methodology"
            )

        # Check specific buckets
        for bucket in calibration.get("calibration_curve", []):
            error = bucket.get("calibration_error", 0)
            if abs(error) > 0.15:
                range_str = f"{bucket['confidence_range'][0]:.1f}-{bucket['confidence_range'][1]:.1f}"
                recommendations.append(
                    f"Confidence range {range_str} is miscalibrated by {error:.1%}"
                )

        return recommendations if recommendations else ["Calibration looks good"]

    def recalibrate(self) -> bool:
        """
        Recalibrate the system based on recorded outcomes.

        Run this periodically (e.g., weekly) to keep calibration current.
        """
        return self.calibration.save_calibration_snapshot(
            domain=self.config.domain,
            notes=f"Periodic recalibration at {datetime.now().isoformat()}"
        )

    # ========== TRAINING DATA ==========

    def get_training_data_status(self) -> Dict[str, Any]:
        """Get status of training data requirements."""
        requirements = self.training_data.get_requirements_status()
        next_priority = self.training_data.get_next_priority_task()

        return {
            "requirements": requirements,
            "next_priority": next_priority,
            "stats": self.training_data.get_stats()
        }

    def generate_labeling_task(
        self,
        task_type: str,
        items: List[Dict]
    ) -> LabelingTask:
        """
        Generate a labeling task for human labelers.

        TRAINING DATA COLLECTION: This is how we get human labels.
        """
        if task_type == "source_reliability":
            return self.training_data.generate_source_reliability_task(items)
        elif task_type == "pattern_extraction":
            return self.training_data.generate_pattern_extraction_task(items)
        elif task_type == "prediction_outcome":
            return self.training_data.generate_prediction_outcome_task(items)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    # ========== SYSTEM MAINTENANCE ==========

    def apply_temporal_decay(self) -> int:
        """Apply temporal decay to all patterns. Run daily."""
        return self.pattern_db.apply_decay_all()

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        return {
            "event_store": self.event_store.get_stats(),
            "pattern_database": self.pattern_db.get_stats(),
            "review_gate": self.review_gate.get_stats(),
            "calibration": self.calibration.get_stats(),
            "training_data": self.training_data.get_stats()
        }

    # ========== CONVENIENCE METHODS ==========

    def register_source(
        self,
        source_id: str,
        name: str,
        source_type: str,
        reliability: float,
        notes: str = ""
    ) -> bool:
        """Register a new information source."""
        source = Source(
            source_id=source_id,
            name=name,
            source_type=SourceType(source_type),
            base_reliability=reliability,
            notes=notes
        )
        return self.event_store.register_source(source)

    def register_reviewer(
        self,
        reviewer_id: str,
        name: str,
        role: str,
        domains: List[str]
    ) -> bool:
        """Register a new human reviewer."""
        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            name=name,
            role=ReviewerRole(role),
            domains=domains
        )
        return self.review_gate.register_reviewer(reviewer)


# ========== CLI INTERFACE ==========

def main():
    """Command-line interface for the system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Epistemic Flow Control System"
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="Directory for databases"
    )
    parser.add_argument(
        "--domain", default="judicial",
        help="Domain for this instance"
    )
    parser.add_argument(
        "command",
        choices=["status", "health", "calibrate", "decay", "training"],
        help="Command to run"
    )

    args = parser.parse_args()

    config = SystemConfig(
        db_dir=args.data_dir,
        domain=args.domain
    )

    system = EpistemicFlowControl(config)

    if args.command == "status":
        print("System Status")
        print("=" * 60)
        health = system.get_system_health()
        for component, stats in health.items():
            print(f"\n{component.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    elif args.command == "health":
        health = system.get_system_health()
        print(json.dumps(health, indent=2, default=str))

    elif args.command == "calibrate":
        print("Running calibration...")
        success = system.recalibrate()
        if success:
            status = system.get_calibration_status()
            print("Calibration complete:")
            print(f"  Factor: {status['current_factor']:.3f}")
            print(f"  Recommendations: {status['recommendations']}")
        else:
            print("Calibration failed - not enough data")

    elif args.command == "decay":
        print("Applying temporal decay...")
        count = system.apply_temporal_decay()
        print(f"Updated {count} patterns")

    elif args.command == "training":
        status = system.get_training_data_status()
        print("Training Data Status")
        print("=" * 60)
        print(f"Stats: {status['stats']}")
        print(f"\nNext priority: {status['next_priority']}")
        print("\nRequirements:")
        for req in status['requirements']:
            status_str = "OK" if req['is_satisfied'] else "NEEDED"
            print(f"  [{status_str}] {req['data_type']}: {req['progress']}")


if __name__ == "__main__":
    main()
