"""
Human Review Gate - The Flow Control Layer

This is the explicit "water in sand" mechanism.
Human reviewers control which LLM outputs flow through to production.

CORE CONCEPT:
-------------
LLM output is WATER - it flows probabilistically.
The gate is the CHANNEL - humans open/close/adjust it.
Production use is the DESTINATION - only gated output arrives.

GATE TYPES:
-----------
1. AUTO_PASS: High confidence, auto-approved
2. REVIEW_REQUIRED: Medium confidence, needs human review
3. BLOCKED: Low confidence or high-stakes, requires explicit approval
4. ESCALATED: Unusual pattern, needs expert review

TRAINING DATA REQUIREMENTS:
---------------------------
1. GATE_THRESHOLD_CALIBRATION:
   - Which confidence thresholds work best for which domains?
   - Need historical data on gate decisions and outcomes

   COLLECTION: Track all gate decisions and subsequent correctness

2. REVIEWER_ACCURACY:
   - How accurate are different reviewers?
   - Some reviewers may be too permissive or too strict

   COLLECTION: Compare reviewer decisions to eventual outcomes

3. ESCALATION_PATTERNS:
   - What patterns warrant escalation?
   - Need examples of correct vs incorrect escalations

   COLLECTION: Expert review of escalation decisions
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class GateDecision(Enum):
    """Decision types for the review gate."""
    AUTO_PASS = "auto_pass"           # High confidence, flows through
    REVIEW_REQUIRED = "review"        # Needs human review
    BLOCKED = "blocked"               # Blocked until explicit approval
    ESCALATED = "escalated"           # Needs expert review
    REJECTED = "rejected"             # Human rejected, does not flow


class ReviewPriority(Enum):
    """Priority levels for review queue."""
    CRITICAL = 1     # Review immediately
    HIGH = 2         # Review within hours
    MEDIUM = 3       # Review within day
    LOW = 4          # Review when convenient


class ReviewerRole(Enum):
    """Roles for reviewers with different permissions."""
    ADMIN = "admin"           # Can approve anything
    DOMAIN_EXPERT = "expert"  # Can approve in their domain
    STANDARD = "standard"     # Can approve standard items
    TRAINEE = "trainee"       # Decisions require confirmation


@dataclass
class GateThreshold:
    """
    Thresholds for gate decisions.

    TRAINING DATA INPUT:
    These thresholds should be calibrated based on historical accuracy.
    """
    domain: str
    auto_pass_threshold: float      # Above this: auto-approve
    review_threshold: float         # Above this: standard review
    block_threshold: float          # Above this: needs approval (below: blocked)
    escalation_conditions: List[str]  # Conditions that trigger escalation

    # Adjustments based on stakes
    high_stakes_penalty: float = 0.1  # Reduce thresholds for high-stakes items


@dataclass
class GatedItem:
    """
    An item passing through the review gate.
    """
    item_id: str
    item_type: str                  # "pattern", "prediction", "recommendation"
    content: Dict[str, Any]         # The actual content
    confidence: float               # System's confidence
    domain: str
    stakes: str                     # "low", "medium", "high"

    # Gate processing
    gate_decision: Optional[GateDecision] = None
    gate_reasoning: Optional[str] = None
    gated_at: Optional[datetime] = None

    # Review (if required)
    reviewed: bool = False
    reviewer: Optional[str] = None
    reviewer_role: Optional[ReviewerRole] = None
    review_decision: Optional[str] = None  # "approve", "reject", "modify"
    review_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    # Modifications (if reviewer modified)
    modified_content: Optional[Dict[str, Any]] = None

    # Outcome tracking (for training)
    outcome_recorded: bool = False
    outcome_correct: Optional[bool] = None
    outcome_notes: Optional[str] = None

    def final_content(self) -> Dict[str, Any]:
        """Get the final content after any modifications."""
        return self.modified_content or self.content


@dataclass
class Reviewer:
    """A human reviewer with accuracy tracking."""
    reviewer_id: str
    name: str
    role: ReviewerRole
    domains: List[str]              # Domains they can review

    # Accuracy tracking
    total_reviews: int = 0
    correct_approvals: int = 0
    correct_rejections: int = 0
    incorrect_approvals: int = 0    # Approved but was wrong
    incorrect_rejections: int = 0   # Rejected but was right

    def accuracy(self) -> float:
        """Calculate reviewer accuracy."""
        if self.total_reviews == 0:
            return 0.5  # No data
        correct = self.correct_approvals + self.correct_rejections
        return correct / self.total_reviews

    def approval_rate(self) -> float:
        """How often does this reviewer approve?"""
        if self.total_reviews == 0:
            return 0.5
        approvals = self.correct_approvals + self.incorrect_approvals
        return approvals / self.total_reviews


class ReviewGate:
    """
    The main gate mechanism.

    HUMAN GATES:
    1. Thresholds are set by humans (calibrated from data)
    2. Reviews are performed by humans
    3. Escalations go to human experts
    4. Outcomes are recorded by humans (ground truth)
    """

    def __init__(
        self,
        db_path: str = "review_gate.db",
        default_thresholds: Optional[Dict[str, GateThreshold]] = None
    ):
        self.db_path = db_path
        self.thresholds: Dict[str, GateThreshold] = default_thresholds or {}
        self.reviewers: Dict[str, Reviewer] = {}
        self._init_db()
        self._init_default_thresholds()

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Thresholds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thresholds (
                domain TEXT PRIMARY KEY,
                auto_pass_threshold REAL NOT NULL,
                review_threshold REAL NOT NULL,
                block_threshold REAL NOT NULL,
                escalation_conditions TEXT,
                high_stakes_penalty REAL DEFAULT 0.1,
                updated_at TEXT NOT NULL
            )
        """)

        # Gated items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gated_items (
                item_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                domain TEXT NOT NULL,
                stakes TEXT NOT NULL,
                gate_decision TEXT,
                gate_reasoning TEXT,
                gated_at TEXT,
                reviewed INTEGER DEFAULT 0,
                reviewer TEXT,
                reviewer_role TEXT,
                review_decision TEXT,
                review_notes TEXT,
                reviewed_at TEXT,
                modified_content TEXT,
                outcome_recorded INTEGER DEFAULT 0,
                outcome_correct INTEGER,
                outcome_notes TEXT
            )
        """)

        # Reviewers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviewers (
                reviewer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                domains TEXT NOT NULL,
                total_reviews INTEGER DEFAULT 0,
                correct_approvals INTEGER DEFAULT 0,
                correct_rejections INTEGER DEFAULT 0,
                incorrect_approvals INTEGER DEFAULT 0,
                incorrect_rejections INTEGER DEFAULT 0
            )
        """)

        # Review queue (for UI integration)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL,
                priority INTEGER NOT NULL,
                assigned_to TEXT,
                queued_at TEXT NOT NULL,
                deadline TEXT,
                FOREIGN KEY (item_id) REFERENCES gated_items(item_id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_gated_domain ON gated_items(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_gated_decision ON gated_items(gate_decision)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_priority ON review_queue(priority)")

        conn.commit()
        conn.close()

    def _init_default_thresholds(self):
        """Initialize default thresholds if not set."""
        defaults = {
            "judicial": GateThreshold(
                domain="judicial",
                auto_pass_threshold=0.95,  # Very conservative - legal stakes high
                review_threshold=0.70,
                block_threshold=0.50,
                escalation_conditions=[
                    "pattern_change > 15%",
                    "new_judge",
                    "contradicts_prior"
                ],
                high_stakes_penalty=0.15
            ),
            "market": GateThreshold(
                domain="market",
                auto_pass_threshold=0.90,
                review_threshold=0.65,
                block_threshold=0.40,
                escalation_conditions=[
                    "large_position",
                    "unusual_pattern",
                    "contradicts_trend"
                ],
                high_stakes_penalty=0.10
            ),
            "default": GateThreshold(
                domain="default",
                auto_pass_threshold=0.92,
                review_threshold=0.70,
                block_threshold=0.50,
                escalation_conditions=[],
                high_stakes_penalty=0.10
            )
        }

        for domain, threshold in defaults.items():
            if domain not in self.thresholds:
                self.thresholds[domain] = threshold

    # ========== GATE PROCESSING ==========

    def process(self, item: GatedItem) -> GateDecision:
        """
        Process an item through the gate.

        Returns the gate decision and updates the item.
        """
        threshold = self.thresholds.get(item.domain, self.thresholds["default"])

        # Adjust thresholds for stakes
        auto_pass = threshold.auto_pass_threshold
        review = threshold.review_threshold
        block = threshold.block_threshold

        if item.stakes == "high":
            auto_pass -= threshold.high_stakes_penalty
            review -= threshold.high_stakes_penalty
            block -= threshold.high_stakes_penalty

        # Check for escalation conditions
        should_escalate = self._check_escalation(item, threshold.escalation_conditions)

        # Determine decision
        if should_escalate:
            decision = GateDecision.ESCALATED
            reasoning = f"Escalation triggered: {should_escalate}"
        elif item.confidence >= auto_pass:
            decision = GateDecision.AUTO_PASS
            reasoning = f"Confidence {item.confidence:.2f} >= auto_pass threshold {auto_pass:.2f}"
        elif item.confidence >= review:
            decision = GateDecision.REVIEW_REQUIRED
            reasoning = f"Confidence {item.confidence:.2f} in review range [{review:.2f}, {auto_pass:.2f})"
        elif item.confidence >= block:
            decision = GateDecision.BLOCKED
            reasoning = f"Confidence {item.confidence:.2f} below review threshold, requires explicit approval"
        else:
            decision = GateDecision.BLOCKED
            reasoning = f"Confidence {item.confidence:.2f} too low, blocked"

        # Update item
        item.gate_decision = decision
        item.gate_reasoning = reasoning
        item.gated_at = datetime.now()

        # Store in database
        self._store_gated_item(item)

        # Add to review queue if needed
        if decision in [GateDecision.REVIEW_REQUIRED, GateDecision.ESCALATED, GateDecision.BLOCKED]:
            priority = self._calculate_priority(item, decision)
            self._add_to_queue(item.item_id, priority)

        return decision

    def _check_escalation(self, item: GatedItem, conditions: List[str]) -> Optional[str]:
        """Check if any escalation conditions are met."""
        content = item.content

        for condition in conditions:
            if "pattern_change" in condition:
                # Check if pattern changed significantly
                change = content.get("pattern_change_percent", 0)
                threshold = float(condition.split(">")[1].strip().replace("%", ""))
                if change > threshold:
                    return f"pattern_change {change}% > {threshold}%"

            if condition == "new_judge" and content.get("is_new_judge"):
                return "new_judge"

            if condition == "contradicts_prior" and content.get("contradicts_prior"):
                return "contradicts_prior"

            if condition == "large_position" and content.get("position_size", 0) > 100000:
                return "large_position"

        return None

    def _calculate_priority(self, item: GatedItem, decision: GateDecision) -> ReviewPriority:
        """Calculate review priority."""
        if decision == GateDecision.ESCALATED:
            return ReviewPriority.CRITICAL

        if item.stakes == "high":
            return ReviewPriority.HIGH

        if item.confidence < 0.5:
            return ReviewPriority.HIGH

        return ReviewPriority.MEDIUM

    def _store_gated_item(self, item: GatedItem):
        """Store gated item in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO gated_items
            (item_id, item_type, content, confidence, domain, stakes,
             gate_decision, gate_reasoning, gated_at, reviewed,
             reviewer, reviewer_role, review_decision, review_notes,
             reviewed_at, modified_content, outcome_recorded, outcome_correct, outcome_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.item_id, item.item_type, json.dumps(item.content),
            item.confidence, item.domain, item.stakes,
            item.gate_decision.value if item.gate_decision else None,
            item.gate_reasoning,
            item.gated_at.isoformat() if item.gated_at else None,
            1 if item.reviewed else 0,
            item.reviewer, item.reviewer_role.value if item.reviewer_role else None,
            item.review_decision, item.review_notes,
            item.reviewed_at.isoformat() if item.reviewed_at else None,
            json.dumps(item.modified_content) if item.modified_content else None,
            1 if item.outcome_recorded else 0,
            1 if item.outcome_correct else (0 if item.outcome_correct is False else None),
            item.outcome_notes
        ))

        conn.commit()
        conn.close()

    def _add_to_queue(self, item_id: str, priority: ReviewPriority):
        """Add item to review queue."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Set deadline based on priority
        now = datetime.now()
        if priority == ReviewPriority.CRITICAL:
            deadline = now + timedelta(hours=2)
        elif priority == ReviewPriority.HIGH:
            deadline = now + timedelta(hours=24)
        elif priority == ReviewPriority.MEDIUM:
            deadline = now + timedelta(days=3)
        else:
            deadline = now + timedelta(days=7)

        cursor.execute("""
            INSERT INTO review_queue (item_id, priority, queued_at, deadline)
            VALUES (?, ?, ?, ?)
        """, (item_id, priority.value, now.isoformat(), deadline.isoformat()))

        conn.commit()
        conn.close()

    # ========== HUMAN REVIEW INTERFACE ==========

    def get_review_queue(
        self,
        reviewer_id: Optional[str] = None,
        domain: Optional[str] = None,
        max_items: int = 20
    ) -> List[GatedItem]:
        """
        Get items awaiting review.

        HUMAN INTERFACE: Returns items for human reviewers.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT g.* FROM gated_items g
            JOIN review_queue q ON g.item_id = q.item_id
            WHERE g.reviewed = 0
        """
        params = []

        if domain:
            query += " AND g.domain = ?"
            params.append(domain)

        if reviewer_id:
            query += " AND (q.assigned_to IS NULL OR q.assigned_to = ?)"
            params.append(reviewer_id)

        query += " ORDER BY q.priority ASC, q.queued_at ASC LIMIT ?"
        params.append(max_items)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        items = []
        for row in rows:
            items.append(GatedItem(
                item_id=row[0],
                item_type=row[1],
                content=json.loads(row[2]),
                confidence=row[3],
                domain=row[4],
                stakes=row[5],
                gate_decision=GateDecision(row[6]) if row[6] else None,
                gate_reasoning=row[7],
                gated_at=datetime.fromisoformat(row[8]) if row[8] else None
            ))

        return items

    def submit_review(
        self,
        item_id: str,
        reviewer_id: str,
        decision: str,  # "approve", "reject", "modify"
        notes: str,
        modified_content: Optional[Dict] = None
    ) -> bool:
        """
        Submit a human review.

        THIS IS THE KEY HUMAN GATE OPERATION.

        TRAINING DATA OUTPUT:
        This creates data for learning reviewer accuracy.
        """
        reviewer = self.reviewers.get(reviewer_id)
        if not reviewer:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()

        cursor.execute("""
            UPDATE gated_items
            SET reviewed = 1,
                reviewer = ?,
                reviewer_role = ?,
                review_decision = ?,
                review_notes = ?,
                reviewed_at = ?,
                modified_content = ?
            WHERE item_id = ?
        """, (
            reviewer_id, reviewer.role.value, decision, notes,
            now.isoformat(),
            json.dumps(modified_content) if modified_content else None,
            item_id
        ))

        # Remove from queue
        cursor.execute("DELETE FROM review_queue WHERE item_id = ?", (item_id,))

        # Update reviewer stats
        cursor.execute("""
            UPDATE reviewers
            SET total_reviews = total_reviews + 1
            WHERE reviewer_id = ?
        """, (reviewer_id,))

        conn.commit()
        conn.close()

        # Update in-memory reviewer
        reviewer.total_reviews += 1

        return True

    # ========== OUTCOME TRACKING (TRAINING DATA) ==========

    def record_outcome(
        self,
        item_id: str,
        was_correct: bool,
        notes: str
    ) -> bool:
        """
        Record the actual outcome for a gated item.

        TRAINING DATA:
        This is the ground truth that lets us:
        1. Calibrate thresholds
        2. Measure reviewer accuracy
        3. Improve the system
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get the review info
        cursor.execute("""
            SELECT reviewer, review_decision FROM gated_items WHERE item_id = ?
        """, (item_id,))
        row = cursor.fetchone()

        if row:
            reviewer_id, review_decision = row

            # Update item outcome
            cursor.execute("""
                UPDATE gated_items
                SET outcome_recorded = 1,
                    outcome_correct = ?,
                    outcome_notes = ?
                WHERE item_id = ?
            """, (1 if was_correct else 0, notes, item_id))

            # Update reviewer accuracy if they reviewed this
            if reviewer_id:
                if review_decision == "approve":
                    if was_correct:
                        cursor.execute("""
                            UPDATE reviewers
                            SET correct_approvals = correct_approvals + 1
                            WHERE reviewer_id = ?
                        """, (reviewer_id,))
                    else:
                        cursor.execute("""
                            UPDATE reviewers
                            SET incorrect_approvals = incorrect_approvals + 1
                            WHERE reviewer_id = ?
                        """, (reviewer_id,))
                elif review_decision == "reject":
                    if was_correct:  # Correctly rejected
                        cursor.execute("""
                            UPDATE reviewers
                            SET correct_rejections = correct_rejections + 1
                            WHERE reviewer_id = ?
                        """, (reviewer_id,))
                    else:  # Incorrectly rejected
                        cursor.execute("""
                            UPDATE reviewers
                            SET incorrect_rejections = incorrect_rejections + 1
                            WHERE reviewer_id = ?
                        """, (reviewer_id,))

            conn.commit()
            conn.close()
            return True

        conn.close()
        return False

    # ========== REVIEWER MANAGEMENT ==========

    def register_reviewer(self, reviewer: Reviewer) -> bool:
        """Register a new reviewer."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO reviewers
                (reviewer_id, name, role, domains, total_reviews,
                 correct_approvals, correct_rejections, incorrect_approvals, incorrect_rejections)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reviewer.reviewer_id, reviewer.name, reviewer.role.value,
                json.dumps(reviewer.domains), reviewer.total_reviews,
                reviewer.correct_approvals, reviewer.correct_rejections,
                reviewer.incorrect_approvals, reviewer.incorrect_rejections
            ))
            conn.commit()
            self.reviewers[reviewer.reviewer_id] = reviewer
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def get_reviewer_stats(self, reviewer_id: str) -> Optional[Dict]:
        """Get reviewer accuracy statistics."""
        reviewer = self.reviewers.get(reviewer_id)
        if not reviewer:
            return None

        return {
            "reviewer_id": reviewer.reviewer_id,
            "name": reviewer.name,
            "role": reviewer.role.value,
            "total_reviews": reviewer.total_reviews,
            "accuracy": reviewer.accuracy(),
            "approval_rate": reviewer.approval_rate(),
            "correct_approvals": reviewer.correct_approvals,
            "incorrect_approvals": reviewer.incorrect_approvals,
            "correct_rejections": reviewer.correct_rejections,
            "incorrect_rejections": reviewer.incorrect_rejections
        }

    # ========== THRESHOLD CALIBRATION ==========

    def calibrate_thresholds(self, domain: str) -> Dict[str, float]:
        """
        Calibrate thresholds based on historical outcomes.

        TRAINING DATA ANALYSIS:
        Uses recorded outcomes to find optimal thresholds.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all items with outcomes for this domain
        cursor.execute("""
            SELECT confidence, gate_decision, review_decision, outcome_correct
            FROM gated_items
            WHERE domain = ? AND outcome_recorded = 1
        """, (domain,))
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 50:
            return {"error": "Not enough data for calibration (need 50+ outcomes)"}

        # Analyze outcomes by confidence bucket
        buckets = {}
        for conf, gate, review, correct in rows:
            bucket = round(conf, 1)
            if bucket not in buckets:
                buckets[bucket] = {"total": 0, "correct": 0}
            buckets[bucket]["total"] += 1
            if correct:
                buckets[bucket]["correct"] += 1

        # Find optimal thresholds
        recommendations = {}

        # Auto-pass threshold: Find lowest confidence where accuracy >= 95%
        for conf in sorted(buckets.keys(), reverse=True):
            if buckets[conf]["total"] >= 5:  # Need enough samples
                accuracy = buckets[conf]["correct"] / buckets[conf]["total"]
                if accuracy >= 0.95:
                    recommendations["auto_pass_threshold"] = conf
                    break

        # Review threshold: Find lowest confidence where accuracy >= 70%
        for conf in sorted(buckets.keys(), reverse=True):
            if buckets[conf]["total"] >= 5:
                accuracy = buckets[conf]["correct"] / buckets[conf]["total"]
                if accuracy >= 0.70:
                    recommendations["review_threshold"] = conf
                    break

        # Block threshold: Where accuracy drops below 50%
        for conf in sorted(buckets.keys()):
            if buckets[conf]["total"] >= 5:
                accuracy = buckets[conf]["correct"] / buckets[conf]["total"]
                if accuracy < 0.50:
                    recommendations["block_threshold"] = conf
                    break

        recommendations["sample_size"] = len(rows)
        recommendations["bucket_analysis"] = buckets

        return recommendations

    # ========== STATISTICS ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM gated_items")
        total_items = cursor.fetchone()[0]

        cursor.execute("""
            SELECT gate_decision, COUNT(*)
            FROM gated_items
            GROUP BY gate_decision
        """)
        by_decision = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM review_queue")
        pending_reviews = cursor.fetchone()[0]

        cursor.execute("""
            SELECT outcome_correct, COUNT(*)
            FROM gated_items
            WHERE outcome_recorded = 1
            GROUP BY outcome_correct
        """)
        outcomes = dict(cursor.fetchall())

        conn.close()

        return {
            "total_items_processed": total_items,
            "by_decision": by_decision,
            "pending_reviews": pending_reviews,
            "outcomes_correct": outcomes.get(1, 0),
            "outcomes_incorrect": outcomes.get(0, 0)
        }


# ========== TRAINING DATA COLLECTION ==========

def create_threshold_calibration_task(historical_data: List[Dict]) -> Dict:
    """
    Create a task for calibrating gate thresholds.

    HUMAN TASK: Review historical outcomes and recommend thresholds.
    """
    return {
        "task_type": "threshold_calibration",
        "instructions": """
            Review the historical gate decisions and outcomes.

            For each domain, determine:
            1. At what confidence should items auto-pass? (target 95%+ accuracy)
            2. At what confidence should items require review? (target 70%+ accuracy)
            3. At what confidence should items be blocked? (below 50% accuracy)

            Consider:
            - Cost of false positives (wrong item passes)
            - Cost of false negatives (right item blocked)
            - Review capacity constraints
        """,
        "historical_data": historical_data,
        "output_format": {
            "domain": "str",
            "auto_pass_threshold": "float 0-1",
            "review_threshold": "float 0-1",
            "block_threshold": "float 0-1",
            "reasoning": "str"
        }
    }


def create_reviewer_calibration_task(reviewers: List[Dict]) -> Dict:
    """
    Create a task for calibrating reviewer trust.

    HUMAN TASK: Review reviewer accuracy and adjust permissions.
    """
    return {
        "task_type": "reviewer_calibration",
        "instructions": """
            Review each reviewer's accuracy statistics.

            Determine:
            1. Should their decisions require confirmation?
            2. Should they be allowed to review high-stakes items?
            3. Are they too permissive or too strict?

            Flag reviewers who:
            - Accuracy < 70%
            - Approval rate significantly different from other reviewers
            - Haven't reviewed in 30+ days
        """,
        "reviewers": reviewers,
        "output_format": {
            "reviewer_id": "str",
            "recommended_role": "str",
            "trust_level": "high | medium | low",
            "issues": ["str", ...],
            "recommendations": "str"
        }
    }


if __name__ == "__main__":
    # Demo
    gate = ReviewGate("demo_gate.db")

    # Register a reviewer
    reviewer = Reviewer(
        reviewer_id="human_001",
        name="Expert Reviewer",
        role=ReviewerRole.DOMAIN_EXPERT,
        domains=["judicial", "market"]
    )
    gate.register_reviewer(reviewer)

    # Create a test item
    item = GatedItem(
        item_id="test_001",
        item_type="pattern",
        content={
            "subject": "Judge Smith",
            "pattern": "Grants summary judgment at 25% rate",
            "pattern_change_percent": 5
        },
        confidence=0.75,
        domain="judicial",
        stakes="medium"
    )

    # Process through gate
    decision = gate.process(item)
    print(f"Gate decision: {decision.value}")
    print(f"Reasoning: {item.gate_reasoning}")

    # Check review queue
    queue = gate.get_review_queue()
    print(f"\nItems in review queue: {len(queue)}")

    # Simulate human review
    if queue:
        gate.submit_review(
            item_id=queue[0].item_id,
            reviewer_id="human_001",
            decision="approve",
            notes="Pattern looks reasonable based on domain knowledge"
        )
        print("Review submitted")

    # Record outcome (later, when we know if it was right)
    gate.record_outcome(
        item_id="test_001",
        was_correct=True,
        notes="Pattern confirmed by subsequent events"
    )

    print(f"\nGate stats: {gate.get_stats()}")
    print(f"Reviewer stats: {gate.get_reviewer_stats('human_001')}")
