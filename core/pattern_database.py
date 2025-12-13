"""
Pattern Database - Bayesian Weights and Temporal Decay

This component stores patterns with weights that:
1. Update based on new evidence (Bayesian updating)
2. Decay over time without confirming events
3. Can be overridden by human judgment

THE STATISTICAL FOUNDATION:
---------------------------
We use Wilson score intervals for small samples, which gives us:
- Lower bound of confidence interval (conservative estimate)
- Proper handling of small sample sizes
- Intuitive "effective sample size" interpretation

TRAINING DATA REQUIREMENTS:
---------------------------
1. PRIOR_DISTRIBUTIONS: Need domain expert input on reasonable priors
   - "How likely is any judge to grant summary judgment?" → Prior rate ~20%
   - "What's typical Markman hearing length?" → Prior mean ~3 hours

   COLLECTION METHOD: Survey domain experts for priors on key patterns

2. DECAY_PARAMETERS: Need empirical data on how fast patterns become stale
   - Judicial patterns: How quickly do judges change behavior?
   - Market patterns: How quickly do trends reverse?

   COLLECTION METHOD: Analyze historical pattern stability

3. HUMAN_OVERRIDE_WEIGHTS: Need data on when human overrides are right
   - When experts override the model, how often are they correct?
   - This informs how much weight to give overrides

   COLLECTION METHOD: Track override outcomes over time
"""

import json
import sqlite3
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from .pattern_extractor import ExtractedPattern, PatternType


@dataclass
class PatternPrior:
    """
    Prior distribution for a pattern type.

    TRAINING DATA INPUT:
    Domain experts provide these priors based on general knowledge.
    Example: "Before seeing any data, what's your estimate for
             summary judgment grant rates?"
    """
    pattern_key: str                # What pattern this prior applies to
    prior_mean: float              # Expected value before seeing data
    prior_strength: float          # How confident in the prior (pseudo-observations)
    domain: str
    source: str                    # "expert_survey", "historical_data", "default"
    notes: str = ""


@dataclass
class StoredPattern:
    """
    A pattern stored in the database with computed weights.
    """
    pattern_id: str
    pattern_key: str               # Canonical key for deduplication/updating

    # Core pattern data
    subject: str
    pattern_type: PatternType
    description: str
    structured_pattern: Dict[str, Any]
    domain: str

    # Statistical data
    supporting_events: int         # Events supporting this pattern
    contradicting_events: int      # Events contradicting this pattern
    total_observations: int        # Total relevant observations

    # Computed weight (0-1)
    raw_weight: float              # Before decay
    decayed_weight: float          # After temporal decay
    human_adjusted_weight: Optional[float] = None  # If human overrode

    # Temporal data
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Human override tracking
    human_override: bool = False
    human_override_reason: Optional[str] = None
    human_overrider: Optional[str] = None

    # Metadata
    source_pattern_ids: List[str] = field(default_factory=list)  # Original extracted patterns

    def effective_weight(self) -> float:
        """Get the weight to use for predictions."""
        if self.human_override and self.human_adjusted_weight is not None:
            return self.human_adjusted_weight
        return self.decayed_weight


def wilson_score_lower(successes: int, total: int, z: float = 1.96) -> float:
    """
    Calculate lower bound of Wilson score interval.

    This gives a conservative estimate that handles small samples well.
    z=1.96 gives 95% confidence interval.

    Example:
    - 3 successes out of 5: raw rate = 60%, Wilson lower = 23%
    - 60 successes out of 100: raw rate = 60%, Wilson lower = 50%

    The more data, the closer lower bound is to raw rate.
    """
    if total == 0:
        return 0.0

    p = successes / total
    denominator = 1 + z*z/total

    center = p + z*z/(2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)

    lower = (center - spread) / denominator
    return max(0.0, lower)


def wilson_score_upper(successes: int, total: int, z: float = 1.96) -> float:
    """Calculate upper bound of Wilson score interval."""
    if total == 0:
        return 1.0

    p = successes / total
    denominator = 1 + z*z/total

    center = p + z*z/(2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)

    upper = (center + spread) / denominator
    return min(1.0, upper)


def bayesian_update(
    prior_mean: float,
    prior_strength: float,  # Pseudo-observations
    observed_successes: int,
    observed_total: int
) -> Tuple[float, float]:
    """
    Bayesian update of a beta distribution.

    Returns (posterior_mean, posterior_strength).

    TRAINING DATA INPUT:
    prior_mean and prior_strength come from domain expert priors.
    """
    # Beta distribution parameters
    prior_alpha = prior_mean * prior_strength
    prior_beta = (1 - prior_mean) * prior_strength

    # Update with observations
    posterior_alpha = prior_alpha + observed_successes
    posterior_beta = prior_beta + (observed_total - observed_successes)

    posterior_strength = posterior_alpha + posterior_beta
    posterior_mean = posterior_alpha / posterior_strength

    return posterior_mean, posterior_strength


def temporal_decay(
    base_weight: float,
    days_since_last_observation: int,
    half_life_days: float = 180.0
) -> float:
    """
    Apply temporal decay to pattern weight.

    Patterns become less reliable without new confirming evidence.

    TRAINING DATA INPUT:
    half_life_days should be calibrated based on domain.
    - Judicial patterns: 180 days (judges change slowly)
    - Market patterns: 30 days (markets change fast)
    - Regulatory patterns: 365 days (regulations change slowly)
    """
    if days_since_last_observation <= 0:
        return base_weight

    decay_factor = math.pow(0.5, days_since_last_observation / half_life_days)
    return base_weight * decay_factor


class PatternDatabase:
    """
    Database for storing and updating patterns.

    HUMAN GATES:
    1. Priors can be set by domain experts
    2. Weights can be overridden by humans
    3. Decay parameters can be adjusted per domain
    """

    def __init__(
        self,
        db_path: str = "patterns.db",
        default_half_life_days: float = 180.0
    ):
        self.db_path = db_path
        self.default_half_life = default_half_life_days
        self.domain_half_lives: Dict[str, float] = {
            "judicial": 180.0,
            "market": 30.0,
            "regulatory": 365.0,
            "behavioral": 90.0
        }
        self.priors: Dict[str, PatternPrior] = {}
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Priors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS priors (
                pattern_key TEXT PRIMARY KEY,
                prior_mean REAL NOT NULL,
                prior_strength REAL NOT NULL,
                domain TEXT NOT NULL,
                source TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_key TEXT NOT NULL,
                subject TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                structured_pattern TEXT NOT NULL,
                domain TEXT NOT NULL,
                supporting_events INTEGER DEFAULT 0,
                contradicting_events INTEGER DEFAULT 0,
                total_observations INTEGER DEFAULT 0,
                raw_weight REAL DEFAULT 0.5,
                decayed_weight REAL DEFAULT 0.5,
                human_adjusted_weight REAL,
                first_observed TEXT NOT NULL,
                last_observed TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                human_override INTEGER DEFAULT 0,
                human_override_reason TEXT,
                human_overrider TEXT,
                source_pattern_ids TEXT NOT NULL
            )
        """)

        # Pattern history (for tracking changes over time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                weight_before REAL,
                weight_after REAL,
                change_type TEXT NOT NULL,
                change_reason TEXT,
                changed_by TEXT,
                changed_at TEXT NOT NULL,
                FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
            )
        """)

        # Human overrides log (TRAINING DATA: tracks when humans override model)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                model_weight REAL NOT NULL,
                human_weight REAL NOT NULL,
                override_reason TEXT NOT NULL,
                overrider TEXT NOT NULL,
                override_at TEXT NOT NULL,
                outcome_verified INTEGER DEFAULT 0,
                outcome_correct INTEGER,
                outcome_notes TEXT,
                FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_subject ON patterns(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_domain ON patterns(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_key ON patterns(pattern_key)")

        conn.commit()
        conn.close()

    # ========== PRIOR MANAGEMENT ==========

    def set_prior(self, prior: PatternPrior) -> bool:
        """
        Set or update a prior for a pattern type.

        HUMAN INPUT: Domain experts provide priors.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO priors
                (pattern_key, prior_mean, prior_strength, domain, source, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prior.pattern_key, prior.prior_mean, prior.prior_strength,
                prior.domain, prior.source, prior.notes, datetime.now().isoformat()
            ))
            conn.commit()
            self.priors[prior.pattern_key] = prior
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def get_prior(self, pattern_key: str, domain: str) -> PatternPrior:
        """Get prior for a pattern, with default fallback."""
        if pattern_key in self.priors:
            return self.priors[pattern_key]

        # Try to load from DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM priors WHERE pattern_key = ?", (pattern_key,))
        row = cursor.fetchone()
        conn.close()

        if row:
            prior = PatternPrior(
                pattern_key=row[0],
                prior_mean=row[1],
                prior_strength=row[2],
                domain=row[3],
                source=row[4],
                notes=row[5]
            )
            self.priors[pattern_key] = prior
            return prior

        # Return default prior
        return PatternPrior(
            pattern_key=pattern_key,
            prior_mean=0.5,
            prior_strength=2.0,  # Weak prior (2 pseudo-observations)
            domain=domain,
            source="default",
            notes="Auto-generated default prior"
        )

    # ========== PATTERN STORAGE ==========

    def store_pattern(self, extracted: ExtractedPattern) -> str:
        """
        Store or update a pattern from extraction.

        Returns pattern_id.
        """
        pattern_key = self._generate_pattern_key(extracted)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if pattern exists
        cursor.execute(
            "SELECT pattern_id, supporting_events, total_observations FROM patterns WHERE pattern_key = ?",
            (pattern_key,)
        )
        existing = cursor.fetchone()

        now = datetime.now()

        if existing:
            # Update existing pattern
            pattern_id = existing[0]
            old_supporting = existing[1]
            old_total = existing[2]

            # Increment counts (simplified - real system would analyze event)
            new_supporting = old_supporting + 1
            new_total = old_total + 1

            # Compute new weight
            prior = self.get_prior(pattern_key, extracted.domain)
            new_weight, _ = bayesian_update(
                prior.prior_mean, prior.prior_strength,
                new_supporting, new_total
            )

            # Apply decay
            half_life = self.domain_half_lives.get(extracted.domain, self.default_half_life)
            # No decay since we just observed it
            decayed_weight = new_weight

            cursor.execute("""
                UPDATE patterns
                SET supporting_events = ?,
                    total_observations = ?,
                    raw_weight = ?,
                    decayed_weight = ?,
                    last_observed = ?,
                    last_updated = ?,
                    source_pattern_ids = source_pattern_ids || ',' || ?
                WHERE pattern_id = ?
            """, (
                new_supporting, new_total, new_weight, decayed_weight,
                now.isoformat(), now.isoformat(), extracted.pattern_id, pattern_id
            ))

            # Record history
            cursor.execute("""
                INSERT INTO pattern_history
                (pattern_id, weight_before, weight_after, change_type, change_reason, changed_by, changed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id, old_supporting / max(1, old_total), new_weight,
                "observation_update", "New supporting event observed",
                "system", now.isoformat()
            ))

        else:
            # Create new pattern
            pattern_id = f"stored_{extracted.pattern_id}"

            # Compute initial weight with prior
            prior = self.get_prior(pattern_key, extracted.domain)
            initial_weight, _ = bayesian_update(
                prior.prior_mean, prior.prior_strength,
                1, 1  # One supporting observation
            )

            cursor.execute("""
                INSERT INTO patterns
                (pattern_id, pattern_key, subject, pattern_type, description,
                 structured_pattern, domain, supporting_events, contradicting_events,
                 total_observations, raw_weight, decayed_weight, first_observed,
                 last_observed, last_updated, source_pattern_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id, pattern_key, extracted.subject,
                extracted.pattern_type.value, extracted.description,
                json.dumps(extracted.structured_pattern), extracted.domain,
                1, 0, 1, initial_weight, initial_weight,
                now.isoformat(), now.isoformat(), now.isoformat(),
                extracted.pattern_id
            ))

        conn.commit()
        conn.close()

        return pattern_id

    def _generate_pattern_key(self, extracted: ExtractedPattern) -> str:
        """Generate canonical key for pattern deduplication."""
        # Key is based on subject + pattern type + domain
        # Plus the actual VALUES of key fields from structured pattern
        key_parts = [
            extracted.subject.lower().replace(" ", "_"),
            extracted.pattern_type.value,
            extracted.domain
        ]

        # Add key VALUES from structured pattern for deduplication
        # These fields distinguish different patterns for same subject
        key_fields = ["motion_type", "outcome", "behavior", "metric"]
        for field in key_fields:
            if field in extracted.structured_pattern:
                value = str(extracted.structured_pattern[field]).lower().replace(" ", "_")
                key_parts.append(f"{field}:{value}")

        return "|".join(key_parts)

    def get_pattern(self, pattern_id: str) -> Optional[StoredPattern]:
        """Retrieve a pattern by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patterns WHERE pattern_id = ?", (pattern_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return StoredPattern(
            pattern_id=row[0],
            pattern_key=row[1],
            subject=row[2],
            pattern_type=PatternType(row[3]),
            description=row[4],
            structured_pattern=json.loads(row[5]),
            domain=row[6],
            supporting_events=row[7],
            contradicting_events=row[8],
            total_observations=row[9],
            raw_weight=row[10],
            decayed_weight=row[11],
            human_adjusted_weight=row[12],
            first_observed=datetime.fromisoformat(row[13]),
            last_observed=datetime.fromisoformat(row[14]),
            last_updated=datetime.fromisoformat(row[15]),
            human_override=bool(row[16]),
            human_override_reason=row[17],
            human_overrider=row[18],
            source_pattern_ids=row[19].split(",") if row[19] else []
        )

    def query_patterns(
        self,
        subject: Optional[str] = None,
        domain: Optional[str] = None,
        min_weight: float = 0.0,
        include_stale: bool = False,
        stale_days: int = 180
    ) -> List[StoredPattern]:
        """Query patterns with filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM patterns WHERE 1=1"
        params = []

        if subject:
            query += " AND subject LIKE ?"
            params.append(f"%{subject}%")

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        if not include_stale:
            cutoff = (datetime.now() - timedelta(days=stale_days)).isoformat()
            query += " AND last_observed >= ?"
            params.append(cutoff)

        query += " ORDER BY decayed_weight DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        patterns = []
        for row in rows:
            pattern = StoredPattern(
                pattern_id=row[0],
                pattern_key=row[1],
                subject=row[2],
                pattern_type=PatternType(row[3]),
                description=row[4],
                structured_pattern=json.loads(row[5]),
                domain=row[6],
                supporting_events=row[7],
                contradicting_events=row[8],
                total_observations=row[9],
                raw_weight=row[10],
                decayed_weight=row[11],
                human_adjusted_weight=row[12],
                first_observed=datetime.fromisoformat(row[13]),
                last_observed=datetime.fromisoformat(row[14]),
                last_updated=datetime.fromisoformat(row[15]),
                human_override=bool(row[16]),
                human_override_reason=row[17],
                human_overrider=row[18],
                source_pattern_ids=row[19].split(",") if row[19] else []
            )

            if pattern.effective_weight() >= min_weight:
                patterns.append(pattern)

        return patterns

    # ========== DECAY APPLICATION ==========

    def apply_decay_all(self) -> int:
        """
        Apply temporal decay to all patterns.

        Run this periodically (e.g., daily).
        Returns number of patterns updated.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT pattern_id, raw_weight, last_observed, domain FROM patterns")
        rows = cursor.fetchall()

        updated = 0
        now = datetime.now()

        for pattern_id, raw_weight, last_observed_str, domain in rows:
            last_observed = datetime.fromisoformat(last_observed_str)
            days_since = (now - last_observed).days

            half_life = self.domain_half_lives.get(domain, self.default_half_life)
            new_decayed = temporal_decay(raw_weight, days_since, half_life)

            cursor.execute("""
                UPDATE patterns
                SET decayed_weight = ?, last_updated = ?
                WHERE pattern_id = ?
            """, (new_decayed, now.isoformat(), pattern_id))

            updated += 1

        conn.commit()
        conn.close()

        return updated

    # ========== HUMAN OVERRIDE (KEY GATE) ==========

    def human_override_weight(
        self,
        pattern_id: str,
        new_weight: float,
        reason: str,
        overrider: str
    ) -> bool:
        """
        Human override of pattern weight.

        THIS IS THE KEY HUMAN GATE for pattern weights.

        TRAINING DATA OUTPUT:
        This creates data for learning when human overrides are correct.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current weight
        cursor.execute(
            "SELECT decayed_weight FROM patterns WHERE pattern_id = ?",
            (pattern_id,)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        model_weight = row[0]

        # Apply override
        cursor.execute("""
            UPDATE patterns
            SET human_adjusted_weight = ?,
                human_override = 1,
                human_override_reason = ?,
                human_overrider = ?,
                last_updated = ?
            WHERE pattern_id = ?
        """, (new_weight, reason, overrider, datetime.now().isoformat(), pattern_id))

        # Log override for training data
        cursor.execute("""
            INSERT INTO human_overrides
            (pattern_id, model_weight, human_weight, override_reason, overrider, override_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            pattern_id, model_weight, new_weight, reason,
            overrider, datetime.now().isoformat()
        ))

        # Record in history
        cursor.execute("""
            INSERT INTO pattern_history
            (pattern_id, weight_before, weight_after, change_type, change_reason, changed_by, changed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern_id, model_weight, new_weight,
            "human_override", reason, overrider, datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()
        return True

    def record_override_outcome(
        self,
        override_id: int,
        was_correct: bool,
        notes: str
    ) -> bool:
        """
        Record whether a human override turned out to be correct.

        TRAINING DATA:
        This teaches us when to trust human overrides vs model weights.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE human_overrides
            SET outcome_verified = 1,
                outcome_correct = ?,
                outcome_notes = ?
            WHERE id = ?
        """, (1 if was_correct else 0, notes, override_id))

        conn.commit()
        conn.close()
        return True

    # ========== STATISTICS AND EXPORT ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM patterns")
        total_patterns = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM patterns WHERE human_override = 1")
        overridden = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(decayed_weight) FROM patterns")
        avg_weight = cursor.fetchone()[0] or 0.0

        cursor.execute("""
            SELECT domain, COUNT(*), AVG(decayed_weight)
            FROM patterns GROUP BY domain
        """)
        by_domain = {row[0]: {"count": row[1], "avg_weight": row[2]} for row in cursor.fetchall()}

        conn.close()

        return {
            "total_patterns": total_patterns,
            "human_overridden": overridden,
            "avg_weight": avg_weight,
            "by_domain": by_domain
        }

    def export_override_training_data(self) -> List[Dict]:
        """
        Export human override data for training.

        TRAINING DATA OUTPUT:
        Use this to learn when human overrides are more accurate than model.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pattern_id, model_weight, human_weight, override_reason,
                   overrider, override_at, outcome_verified, outcome_correct, outcome_notes
            FROM human_overrides
            WHERE outcome_verified = 1
        """)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "pattern_id": row[0],
                "model_weight": row[1],
                "human_weight": row[2],
                "override_reason": row[3],
                "overrider": row[4],
                "override_at": row[5],
                "outcome_correct": bool(row[7]),
                "outcome_notes": row[8]
            }
            for row in rows
        ]


# ========== TRAINING DATA COLLECTION ==========

def create_prior_elicitation_task(pattern_types: List[str], domain: str) -> Dict:
    """
    Create a task for domain experts to provide prior distributions.

    HUMAN TASK: Provide reasonable priors for pattern types.
    """
    return {
        "task_type": "prior_elicitation",
        "instructions": """
            For each pattern type, provide your PRIOR belief:

            1. prior_mean: What's your best guess for the base rate?
               Example: "What percentage of judges typically grant summary judgment?"

            2. prior_strength: How confident are you? (expressed as pseudo-observations)
               - 2 = Very uncertain, let data speak quickly
               - 10 = Moderately confident
               - 50 = Very confident, need lots of data to change mind

            Think of prior_strength as: "How many observations would it take
            to cut my uncertainty in half?"
        """,
        "domain": domain,
        "pattern_types": pattern_types,
        "output_format": {
            "pattern_key": "str",
            "prior_mean": "float 0-1",
            "prior_strength": "float > 0",
            "reasoning": "str"
        }
    }


def create_decay_calibration_task(domain: str, historical_patterns: List[Dict]) -> Dict:
    """
    Create a task to calibrate decay parameters.

    HUMAN TASK: Review historical patterns and estimate how quickly they become stale.
    """
    return {
        "task_type": "decay_calibration",
        "instructions": f"""
            Review these historical patterns from the {domain} domain.

            For each pattern, estimate:
            1. After how many days without new evidence should we halve our confidence?
            2. What signals indicate a pattern is becoming stale?
            3. What would make you completely distrust a pattern?

            Think about domain-specific factors:
            - How quickly do actors in this domain change behavior?
            - Are there regular events that could confirm/update patterns?
            - What external factors could invalidate patterns?
        """,
        "domain": domain,
        "patterns": historical_patterns,
        "output_format": {
            "recommended_half_life_days": "int",
            "reasoning": "str",
            "staleness_signals": ["str", ...],
            "invalidation_events": ["str", ...]
        }
    }


if __name__ == "__main__":
    # Demo
    db = PatternDatabase("demo_patterns.db")

    # Set a prior (HUMAN INPUT)
    prior = PatternPrior(
        pattern_key="judge_smith|outcome|judicial",
        prior_mean=0.20,  # Prior: 20% summary judgment rate
        prior_strength=10.0,  # Moderate confidence
        domain="judicial",
        source="expert_survey",
        notes="Based on national average SJ grant rates"
    )
    db.set_prior(prior)

    # Demo: Wilson score examples
    print("Wilson score examples (95% CI lower bound):")
    print(f"  3/5 successes: raw=60%, Wilson lower={wilson_score_lower(3, 5):.1%}")
    print(f"  60/100 successes: raw=60%, Wilson lower={wilson_score_lower(60, 100):.1%}")
    print(f"  600/1000 successes: raw=60%, Wilson lower={wilson_score_lower(600, 1000):.1%}")

    # Demo: Bayesian update
    print("\nBayesian update example:")
    prior_mean, prior_strength = 0.2, 10  # Expect 20% rate
    post_mean, post_strength = bayesian_update(prior_mean, prior_strength, 5, 10)
    print(f"  Prior: {prior_mean:.0%} (strength={prior_strength})")
    print(f"  Observed: 5/10 = 50%")
    print(f"  Posterior: {post_mean:.1%} (strength={post_strength})")

    # Demo: Temporal decay
    print("\nTemporal decay example (180-day half-life):")
    for days in [0, 30, 90, 180, 365]:
        decayed = temporal_decay(0.80, days, 180)
        print(f"  After {days} days: {decayed:.1%}")

    print(f"\nDatabase stats: {db.get_stats()}")
