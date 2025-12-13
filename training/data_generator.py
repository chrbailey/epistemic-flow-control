"""
Training Data Generator - Preparing Labeled Examples

This component consolidates all training data requirements and provides
tools for generating, collecting, and organizing training datasets.

THE KEY INSIGHT (AUNT/UNCLE DATA):
----------------------------------
"A little bit of accurate data goes a long ways no different than your
parent's siblings who fill in missing details in stories and expand
your complete understanding."

This means:
1. We don't need MASSIVE datasets
2. We need ACCURATE, HIGH-QUALITY labels
3. Expert corrections are extremely valuable
4. A few well-placed labels can calibrate the whole system

TRAINING DATA CATEGORIES:
-------------------------
1. SOURCE_RELIABILITY: How trustworthy are different sources?
2. PATTERN_EXTRACTION: Does this event contain this pattern?
3. PATTERN_VALIDATION: Is this extracted pattern correct?
4. PREDICTION_OUTCOMES: Was this prediction right or wrong?
5. HUMAN_OVERRIDE_OUTCOMES: Was the human override correct?
6. CALIBRATION_DATA: Confidence vs actual accuracy

COLLECTION STRATEGY:
--------------------
1. Start with expert labels on small, high-value dataset
2. Use those to bootstrap the system
3. Collect more labels from system usage (outcome tracking)
4. Periodically have experts review samples

MINIMUM VIABLE TRAINING DATA:
------------------------------
- 50 labeled sources with reliability scores
- 100 events with human-verified pattern extractions
- 200 predictions with recorded outcomes
- 20 human override decisions with outcomes

This is enough to bootstrap. The system learns more from usage.
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


class TaskType(Enum):
    """Types of labeling tasks."""
    SOURCE_RELIABILITY = "source_reliability"
    EVENT_VERIFICATION = "event_verification"
    PATTERN_EXTRACTION = "pattern_extraction"
    PATTERN_VALIDATION = "pattern_validation"
    PREDICTION_OUTCOME = "prediction_outcome"
    OVERRIDE_OUTCOME = "override_outcome"
    PRIOR_ELICITATION = "prior_elicitation"
    THRESHOLD_CALIBRATION = "threshold_calibration"


class TaskStatus(Enum):
    """Status of labeling tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"  # Expert reviewed the labels


@dataclass
class LabelingTask:
    """
    A task for human labelers.
    """
    task_id: str
    task_type: TaskType
    instructions: str
    items: List[Dict[str, Any]]     # Items to label
    output_format: Dict[str, str]   # Expected output format

    # Assignment
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None

    # Status
    status: TaskStatus = TaskStatus.PENDING
    completed_at: Optional[datetime] = None

    # Results
    labels: List[Dict[str, Any]] = field(default_factory=list)
    labeler_notes: str = ""

    # Quality
    expert_reviewed: bool = False
    expert_reviewer: Optional[str] = None
    review_notes: str = ""
    quality_score: Optional[float] = None  # 0-1


@dataclass
class TrainingDataset:
    """
    A compiled training dataset ready for use.
    """
    dataset_id: str
    dataset_type: str
    domain: str
    version: str

    # Data
    examples: List[Dict[str, Any]]
    size: int

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    source_tasks: List[str] = field(default_factory=list)

    # Quality metrics
    expert_labeled_count: int = 0
    avg_quality_score: float = 0.0
    coverage_notes: str = ""


class TrainingDataGenerator:
    """
    Generates and manages training data for the epistemic flow control system.
    """

    def __init__(self, db_path: str = "training_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labeling_tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                instructions TEXT NOT NULL,
                items TEXT NOT NULL,
                output_format TEXT NOT NULL,
                assigned_to TEXT,
                assigned_at TEXT,
                status TEXT NOT NULL,
                completed_at TEXT,
                labels TEXT,
                labeler_notes TEXT,
                expert_reviewed INTEGER DEFAULT 0,
                expert_reviewer TEXT,
                review_notes TEXT,
                quality_score REAL
            )
        """)

        # Datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                dataset_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                version TEXT NOT NULL,
                examples TEXT NOT NULL,
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                source_tasks TEXT,
                expert_labeled_count INTEGER DEFAULT 0,
                avg_quality_score REAL DEFAULT 0.0,
                coverage_notes TEXT
            )
        """)

        # Training data requirements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_requirements (
                requirement_id TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                data_type TEXT NOT NULL,
                description TEXT NOT NULL,
                minimum_samples INTEGER NOT NULL,
                current_samples INTEGER DEFAULT 0,
                is_satisfied INTEGER DEFAULT 0,
                priority TEXT NOT NULL,
                collection_method TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

        # Initialize requirements
        self._init_requirements()

    def _init_requirements(self):
        """Initialize training data requirements."""
        requirements = [
            {
                "requirement_id": "req_source_reliability",
                "component": "EventStore",
                "data_type": "source_reliability",
                "description": "Labeled sources with reliability scores (0-1) and reasoning",
                "minimum_samples": 50,
                "priority": "high",
                "collection_method": "Expert survey: rate sources on reliability scale with justification"
            },
            {
                "requirement_id": "req_event_verification",
                "component": "EventStore",
                "data_type": "event_verification",
                "description": "Events with verified accuracy status",
                "minimum_samples": 100,
                "priority": "high",
                "collection_method": "Expert review: verify events against primary sources"
            },
            {
                "requirement_id": "req_pattern_extraction",
                "component": "PatternExtractor",
                "data_type": "pattern_extraction",
                "description": "Events with correct pattern extractions (gold standard)",
                "minimum_samples": 100,
                "priority": "high",
                "collection_method": "Expert annotation: extract patterns from events, validate LLM extractions"
            },
            {
                "requirement_id": "req_prediction_outcomes",
                "component": "CalibrationEngine",
                "data_type": "prediction_outcome",
                "description": "Predictions with recorded actual outcomes",
                "minimum_samples": 200,
                "priority": "high",
                "collection_method": "Outcome tracking: record what actually happened for each prediction"
            },
            {
                "requirement_id": "req_override_outcomes",
                "component": "ReviewGate",
                "data_type": "override_outcome",
                "description": "Human override decisions with outcome verification",
                "minimum_samples": 20,
                "priority": "medium",
                "collection_method": "Track when humans override model, verify if override was correct"
            },
            {
                "requirement_id": "req_domain_priors",
                "component": "PatternDatabase",
                "data_type": "prior_elicitation",
                "description": "Expert priors for key pattern types",
                "minimum_samples": 30,
                "priority": "medium",
                "collection_method": "Expert elicitation: ask domain experts for reasonable priors"
            },
            {
                "requirement_id": "req_decay_parameters",
                "component": "PatternDatabase",
                "data_type": "decay_calibration",
                "description": "Calibrated decay half-lives by domain",
                "minimum_samples": 10,
                "priority": "low",
                "collection_method": "Historical analysis: measure how quickly patterns become stale"
            },
            {
                "requirement_id": "req_gate_thresholds",
                "component": "ReviewGate",
                "data_type": "threshold_calibration",
                "description": "Optimal gate thresholds by domain",
                "minimum_samples": 50,
                "priority": "medium",
                "collection_method": "Threshold search: find thresholds that balance accuracy and review load"
            }
        ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for req in requirements:
            cursor.execute("""
                INSERT OR IGNORE INTO data_requirements
                (requirement_id, component, data_type, description, minimum_samples,
                 current_samples, is_satisfied, priority, collection_method)
                VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)
            """, (
                req["requirement_id"], req["component"], req["data_type"],
                req["description"], req["minimum_samples"],
                req["priority"], req["collection_method"]
            ))

        conn.commit()
        conn.close()

    # ========== TASK GENERATION ==========

    def generate_source_reliability_task(
        self,
        sources: List[Dict],
        task_id: Optional[str] = None
    ) -> LabelingTask:
        """
        Generate a task for labeling source reliability.

        SMALL BUT VALUABLE: 50 well-labeled sources can calibrate the entire
        source reliability system.
        """
        task = LabelingTask(
            task_id=task_id or f"src_rel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=TaskType.SOURCE_RELIABILITY,
            instructions="""
TASK: Rate Information Source Reliability

For each source, provide:
1. reliability_score (0-100): How trustworthy is this source?
   - 90-100: Official records, government sources
   - 70-89: Major reputable journalism, peer-reviewed
   - 50-69: Reputable but may have bias
   - 30-49: Opinion-heavy, variable quality
   - 0-29: Unreliable, often wrong

2. reasoning: Why this score? What makes this source more/less reliable?

3. caveats: Any situations where reliability changes?
   (e.g., "Reliable for facts, unreliable for analysis")

REMEMBER: Your labels will calibrate the system. Be thoughtful.
            """.strip(),
            items=sources,
            output_format={
                "source_id": "str",
                "reliability_score": "int 0-100",
                "reasoning": "str",
                "caveats": "str"
            }
        )

        self._save_task(task)
        return task

    def generate_pattern_extraction_task(
        self,
        events: List[Dict],
        task_id: Optional[str] = None
    ) -> LabelingTask:
        """
        Generate a task for validating pattern extractions.

        KEY VALUE: Corrections to LLM extractions are extremely valuable.
        Even 20-30 corrections can significantly improve extraction quality.
        """
        task = LabelingTask(
            task_id=task_id or f"pat_ext_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=TaskType.PATTERN_EXTRACTION,
            instructions="""
TASK: Validate and Correct Pattern Extractions

For each event, an LLM extracted a pattern. Your job:

1. Review the event text
2. Review the LLM's extracted pattern
3. Rate: CORRECT, PARTIAL, or INCORRECT
4. If PARTIAL or INCORRECT, provide the correct pattern

WHAT TO LOOK FOR:
- Did LLM identify the right pattern type?
- Is the description accurate?
- Is the confidence level appropriate?
- Are there patterns LLM missed?

CORRECTIONS ARE GOLD:
When you correct an LLM mistake, you're teaching the system.
A few good corrections are worth more than many confirmations.
            """.strip(),
            items=events,
            output_format={
                "event_id": "str",
                "llm_pattern_correct": "correct | partial | incorrect",
                "corrected_pattern": "dict or null",
                "patterns_missed": "list of dicts or null",
                "confidence_appropriate": "yes | too_high | too_low",
                "notes": "str"
            }
        )

        self._save_task(task)
        return task

    def generate_prediction_outcome_task(
        self,
        predictions: List[Dict],
        task_id: Optional[str] = None
    ) -> LabelingTask:
        """
        Generate a task for recording prediction outcomes.

        CALIBRATION DATA: These outcomes teach us if our confidence is calibrated.
        """
        task = LabelingTask(
            task_id=task_id or f"pred_out_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=TaskType.PREDICTION_OUTCOME,
            instructions="""
TASK: Record Prediction Outcomes

For each prediction, determine what actually happened:

1. Look up the actual outcome (court decision, market move, etc.)
2. Determine if prediction was CORRECT or INCORRECT
3. Note the actual value if different from predicted
4. Rate if confidence was appropriate

BE HONEST:
- If prediction was partially correct, lean toward INCORRECT
- We need accurate calibration data, not optimistic data
- Incorrect predictions are valuable for learning

EXAMPLE:
- Prediction: "Judge will grant summary judgment" (80% confidence)
- Actual: Judge denied summary judgment
- Outcome: INCORRECT
- Notes: "Judge cited genuine issues of fact"
            """.strip(),
            items=predictions,
            output_format={
                "prediction_id": "str",
                "actual_outcome": "any",
                "was_correct": "bool",
                "confidence_appropriate": "yes | too_high | too_low",
                "notes": "str"
            }
        )

        self._save_task(task)
        return task

    def generate_prior_elicitation_task(
        self,
        pattern_types: List[str],
        domain: str,
        task_id: Optional[str] = None
    ) -> LabelingTask:
        """
        Generate a task for eliciting priors from experts.

        EXPERT KNOWLEDGE: These priors anchor the Bayesian system.
        """
        task = LabelingTask(
            task_id=task_id or f"prior_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=TaskType.PRIOR_ELICITATION,
            instructions=f"""
TASK: Provide Prior Beliefs for {domain.title()} Domain

Before seeing any data, what are your beliefs about these patterns?

For each pattern type, provide:

1. prior_mean (0-1): Your best guess for the base rate
   Example: "What fraction of judges grant summary judgment?" → 0.20

2. prior_strength (1-100): How confident in your prior?
   - 1-10: Very uncertain, let data speak
   - 10-30: Moderate confidence
   - 30-100: Very confident, need lots of data to change

Think of prior_strength as: "How many observations would it take
to move my belief halfway toward a surprising observation?"

YOUR EXPERTISE MATTERS:
These priors anchor the system. A good prior from an expert is
worth hundreds of data points in terms of early system accuracy.
            """.strip(),
            items=[{"pattern_type": pt, "domain": domain} for pt in pattern_types],
            output_format={
                "pattern_type": "str",
                "prior_mean": "float 0-1",
                "prior_strength": "float 1-100",
                "reasoning": "str",
                "uncertainty_notes": "str"
            }
        )

        self._save_task(task)
        return task

    def _save_task(self, task: LabelingTask):
        """Save a task to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO labeling_tasks
            (task_id, task_type, instructions, items, output_format,
             assigned_to, assigned_at, status, completed_at, labels,
             labeler_notes, expert_reviewed, expert_reviewer, review_notes, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.task_type.value,
            task.instructions,
            json.dumps(task.items),
            json.dumps(task.output_format),
            task.assigned_to,
            task.assigned_at.isoformat() if task.assigned_at else None,
            task.status.value,
            task.completed_at.isoformat() if task.completed_at else None,
            json.dumps(task.labels),
            task.labeler_notes,
            1 if task.expert_reviewed else 0,
            task.expert_reviewer,
            task.review_notes,
            task.quality_score
        ))

        conn.commit()
        conn.close()

    # ========== TASK COMPLETION ==========

    def submit_labels(
        self,
        task_id: str,
        labels: List[Dict],
        labeler_notes: str = ""
    ) -> bool:
        """
        Submit labels for a task.

        HUMAN INPUT: This is where labeled data enters the system.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE labeling_tasks
            SET labels = ?,
                labeler_notes = ?,
                status = ?,
                completed_at = ?
            WHERE task_id = ?
        """, (
            json.dumps(labels),
            labeler_notes,
            TaskStatus.COMPLETED.value,
            datetime.now().isoformat(),
            task_id
        ))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            self._update_requirements(task_id)

        return success

    def expert_review_labels(
        self,
        task_id: str,
        reviewer: str,
        quality_score: float,
        review_notes: str,
        corrected_labels: Optional[List[Dict]] = None
    ) -> bool:
        """
        Expert review of submitted labels.

        QUALITY GATE: Expert review ensures label quality.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updates = [
            "expert_reviewed = 1",
            "expert_reviewer = ?",
            "quality_score = ?",
            "review_notes = ?",
            "status = ?"
        ]
        params = [reviewer, quality_score, review_notes, TaskStatus.REVIEWED.value]

        if corrected_labels:
            updates.append("labels = ?")
            params.append(json.dumps(corrected_labels))

        params.append(task_id)

        cursor.execute(f"""
            UPDATE labeling_tasks
            SET {', '.join(updates)}
            WHERE task_id = ?
        """, params)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def _update_requirements(self, task_id: str):
        """Update requirement counts after task completion."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get task info
        cursor.execute(
            "SELECT task_type, labels FROM labeling_tasks WHERE task_id = ?",
            (task_id,)
        )
        row = cursor.fetchone()

        if row:
            task_type, labels_json = row
            labels = json.loads(labels_json) if labels_json else []

            # Map task type to requirement
            type_to_req = {
                TaskType.SOURCE_RELIABILITY.value: "req_source_reliability",
                TaskType.EVENT_VERIFICATION.value: "req_event_verification",
                TaskType.PATTERN_EXTRACTION.value: "req_pattern_extraction",
                TaskType.PREDICTION_OUTCOME.value: "req_prediction_outcomes",
                TaskType.OVERRIDE_OUTCOME.value: "req_override_outcomes",
                TaskType.PRIOR_ELICITATION.value: "req_domain_priors",
                TaskType.THRESHOLD_CALIBRATION.value: "req_gate_thresholds"
            }

            req_id = type_to_req.get(task_type)
            if req_id:
                cursor.execute("""
                    UPDATE data_requirements
                    SET current_samples = current_samples + ?,
                        is_satisfied = CASE
                            WHEN current_samples + ? >= minimum_samples THEN 1
                            ELSE 0
                        END
                    WHERE requirement_id = ?
                """, (len(labels), len(labels), req_id))

        conn.commit()
        conn.close()

    # ========== DATASET COMPILATION ==========

    def compile_dataset(
        self,
        dataset_type: str,
        domain: str,
        version: str,
        min_quality_score: float = 0.7
    ) -> TrainingDataset:
        """
        Compile reviewed labels into a training dataset.

        Only includes labels that passed expert review.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all reviewed tasks of this type
        cursor.execute("""
            SELECT task_id, labels, quality_score
            FROM labeling_tasks
            WHERE task_type = ?
              AND status = 'reviewed'
              AND quality_score >= ?
        """, (dataset_type, min_quality_score))

        rows = cursor.fetchall()

        examples = []
        source_tasks = []
        total_quality = 0.0

        for task_id, labels_json, quality in rows:
            labels = json.loads(labels_json) if labels_json else []
            examples.extend(labels)
            source_tasks.append(task_id)
            total_quality += quality

        dataset = TrainingDataset(
            dataset_id=f"{dataset_type}_{domain}_{version}",
            dataset_type=dataset_type,
            domain=domain,
            version=version,
            examples=examples,
            size=len(examples),
            source_tasks=source_tasks,
            expert_labeled_count=len(examples),
            avg_quality_score=total_quality / len(rows) if rows else 0.0
        )

        # Save dataset
        cursor.execute("""
            INSERT OR REPLACE INTO datasets
            (dataset_id, dataset_type, domain, version, examples, size,
             created_at, source_tasks, expert_labeled_count, avg_quality_score, coverage_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset.dataset_id, dataset.dataset_type, dataset.domain,
            dataset.version, json.dumps(dataset.examples), dataset.size,
            dataset.created_at.isoformat(), json.dumps(dataset.source_tasks),
            dataset.expert_labeled_count, dataset.avg_quality_score, ""
        ))

        conn.commit()
        conn.close()

        return dataset

    def export_dataset(self, dataset_id: str, output_path: str) -> bool:
        """Export a dataset to JSON file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False

        dataset = {
            "dataset_id": row[0],
            "dataset_type": row[1],
            "domain": row[2],
            "version": row[3],
            "examples": json.loads(row[4]),
            "size": row[5],
            "created_at": row[6],
            "source_tasks": json.loads(row[7]) if row[7] else [],
            "expert_labeled_count": row[8],
            "avg_quality_score": row[9],
            "coverage_notes": row[10]
        }

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        return True

    # ========== REQUIREMENTS TRACKING ==========

    def get_requirements_status(self) -> List[Dict]:
        """Get status of all training data requirements."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT requirement_id, component, data_type, description,
                   minimum_samples, current_samples, is_satisfied, priority,
                   collection_method
            FROM data_requirements
            ORDER BY
                CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
                is_satisfied ASC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "requirement_id": row[0],
                "component": row[1],
                "data_type": row[2],
                "description": row[3],
                "minimum_samples": row[4],
                "current_samples": row[5],
                "progress": f"{row[5]}/{row[4]} ({row[5]/row[4]*100:.0f}%)" if row[4] > 0 else "N/A",
                "is_satisfied": bool(row[6]),
                "priority": row[7],
                "collection_method": row[8]
            }
            for row in rows
        ]

    def get_next_priority_task(self) -> Optional[Dict]:
        """
        Get the highest priority unsatisfied requirement.

        BOOTSTRAPPING: Start with high-priority requirements.
        """
        requirements = self.get_requirements_status()

        for req in requirements:
            if not req["is_satisfied"]:
                return req

        return None

    # ========== STATISTICS ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get training data statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM labeling_tasks")
        total_tasks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM labeling_tasks WHERE status = 'completed'")
        completed_tasks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM labeling_tasks WHERE status = 'reviewed'")
        reviewed_tasks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM datasets")
        total_datasets = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(size) FROM datasets")
        total_examples = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT COUNT(*) FROM data_requirements WHERE is_satisfied = 1
        """)
        satisfied_requirements = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM data_requirements")
        total_requirements = cursor.fetchone()[0]

        conn.close()

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "reviewed_tasks": reviewed_tasks,
            "total_datasets": total_datasets,
            "total_labeled_examples": total_examples,
            "requirements_satisfied": f"{satisfied_requirements}/{total_requirements}"
        }


# ========== BOOTSTRAP HELPERS ==========

def generate_bootstrap_plan(generator: TrainingDataGenerator) -> List[Dict]:
    """
    Generate a plan for bootstrapping the training data.

    Returns ordered list of tasks to complete.
    """
    requirements = generator.get_requirements_status()

    plan = []
    for req in requirements:
        if not req["is_satisfied"]:
            samples_needed = req["minimum_samples"] - req["current_samples"]
            plan.append({
                "priority": req["priority"],
                "requirement": req["requirement_id"],
                "data_type": req["data_type"],
                "samples_needed": samples_needed,
                "collection_method": req["collection_method"],
                "estimated_effort": _estimate_effort(req["data_type"], samples_needed)
            })

    return plan


def _estimate_effort(data_type: str, samples: int) -> str:
    """Estimate effort for labeling."""
    # Minutes per sample by type
    time_per_sample = {
        "source_reliability": 2,
        "event_verification": 5,
        "pattern_extraction": 10,
        "pattern_validation": 5,
        "prediction_outcome": 3,
        "override_outcome": 5,
        "prior_elicitation": 15,
        "threshold_calibration": 10
    }

    minutes = time_per_sample.get(data_type, 5) * samples
    hours = minutes / 60

    if hours < 1:
        return f"{minutes} minutes"
    elif hours < 8:
        return f"{hours:.1f} hours"
    else:
        return f"{hours/8:.1f} days"


if __name__ == "__main__":
    # Demo
    generator = TrainingDataGenerator("demo_training.db")

    # Check requirements
    print("Training Data Requirements:")
    print("-" * 60)
    for req in generator.get_requirements_status():
        status = "SATISFIED" if req["is_satisfied"] else "NEEDED"
        print(f"[{status}] {req['data_type']}: {req['progress']}")
        print(f"         Method: {req['collection_method'][:50]}...")
        print()

    # Generate bootstrap plan
    print("\nBootstrap Plan:")
    print("-" * 60)
    plan = generate_bootstrap_plan(generator)
    for item in plan:
        print(f"[{item['priority'].upper()}] {item['data_type']}")
        print(f"  Need: {item['samples_needed']} samples")
        print(f"  Effort: {item['estimated_effort']}")
        print()

    # Create a sample task
    task = generator.generate_source_reliability_task([
        {"source_id": "pacer", "name": "PACER", "description": "Federal court records"},
        {"source_id": "law360", "name": "Law360", "description": "Legal journalism"}
    ])
    print(f"Created task: {task.task_id}")

    print(f"\nStats: {generator.get_stats()}")
