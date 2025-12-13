"""
Event Store - The Ground Truth Layer

This is the foundation of the epistemic flow control system.
Events are immutable facts from verified sources.

TRAINING DATA REQUIREMENTS:
---------------------------
1. SOURCE_RELIABILITY: Need labeled examples of sources with reliability scores
   - Example: {"source": "PACER", "reliability": 0.99, "reason": "Official court records"}
   - Example: {"source": "Law360", "reliability": 0.85, "reason": "Reputable legal journalism"}
   - Example: {"source": "Twitter/X", "reliability": 0.40, "reason": "Unverified, often opinion"}

   COLLECTION METHOD: Human expert labels ~50 sources, system extrapolates

2. EVENT_VERIFICATION: Need examples of events with verification status
   - Verified events: Official records, multiple independent sources
   - Unverified events: Single source, opinion-based, hearsay

   COLLECTION METHOD: Human reviews sample of events, labels verification confidence

3. EVENT_LINKAGE: Need examples of events that relate to each other
   - Same case, different rulings
   - Same judge, pattern evolution
   - Same legal issue, different outcomes

   COLLECTION METHOD: Human identifies related events, system learns linkage patterns
"""

import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


class VerificationStatus(Enum):
    """How confident are we this event actually happened as described?"""
    VERIFIED = "verified"           # Official record or 3+ independent sources
    HIGH_CONFIDENCE = "high"        # 2 independent sources or reputable single source
    MEDIUM_CONFIDENCE = "medium"    # Single reputable source
    LOW_CONFIDENCE = "low"          # Single unverified source
    UNVERIFIED = "unverified"       # No verification attempted
    DISPUTED = "disputed"           # Conflicting sources


class SourceType(Enum):
    """What kind of source is this?"""
    OFFICIAL_RECORD = "official"    # PACER, SEC EDGAR, government records
    LEGAL_JOURNALISM = "journalism" # Law360, Bloomberg Law, Reuters Legal
    ACADEMIC = "academic"           # Law review articles, research papers
    PRACTITIONER = "practitioner"   # Attorney observations, survey responses
    SOCIAL_MEDIA = "social"         # Twitter, LinkedIn, blogs
    INTERNAL = "internal"           # Our own analysis, derived data


@dataclass
class Source:
    """
    A source of information with reliability scoring.

    TRAINING DATA NEEDED: ~50-100 labeled sources with reliability scores
    and human reasoning for why that score was assigned.
    """
    source_id: str
    name: str
    source_type: SourceType
    base_reliability: float         # 0.0 to 1.0, from training data
    url_pattern: Optional[str] = None
    notes: str = ""

    # These get updated based on validation outcomes
    accuracy_track_record: float = 0.0  # Starts at 0, updated as we verify
    total_events_sourced: int = 0
    verified_events: int = 0

    def effective_reliability(self) -> float:
        """
        Combine base reliability with track record.
        As we verify more events from this source, track record matters more.
        """
        if self.total_events_sourced < 10:
            # Not enough data, use base reliability
            return self.base_reliability

        # Bayesian-ish blend: base reliability + track record
        track_weight = min(0.5, self.total_events_sourced / 100)  # Max 50% weight to track record
        return (1 - track_weight) * self.base_reliability + track_weight * self.accuracy_track_record


@dataclass
class Event:
    """
    An immutable fact from the world.

    Events are the atoms of our system. Everything traces back to events.
    Once created, events don't change (immutability principle).
    """
    event_id: str

    # 5W1H Core Fields
    who: List[str]                  # Entities involved (judge, parties, lawyers)
    what: str                       # What happened (ruling, statement, action)
    when: datetime                  # When it happened
    where: str                      # Jurisdiction, court, location
    why: Optional[str]              # Stated reasoning (if available)
    how: Optional[str]              # Process/procedure used

    # Source and Verification
    source_id: str                  # Which source provided this
    source_url: Optional[str]       # Link to original
    raw_text: str                   # Original text/content
    verification_status: VerificationStatus
    verification_notes: str = ""

    # Metadata
    domain: str = "general"         # "judicial", "market", "compliance", etc.
    event_type: str = "event"       # More specific classification
    created_at: datetime = field(default_factory=datetime.now)

    # Linkage (populated after creation)
    related_events: List[str] = field(default_factory=list)

    def content_hash(self) -> str:
        """Generate deterministic hash of event content for deduplication."""
        content = f"{self.what}|{self.when.isoformat()}|{self.where}|{'|'.join(sorted(self.who))}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class EventStore:
    """
    Persistent storage for events with verification tracking.

    The Event Store is append-only for events (immutability).
    Verification status and linkage can be updated.
    """

    def __init__(self, db_path: str = "events.db"):
        self.db_path = db_path
        self._init_db()
        self._source_cache: Dict[str, Source] = {}

    def _init_db(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                base_reliability REAL NOT NULL,
                url_pattern TEXT,
                notes TEXT,
                accuracy_track_record REAL DEFAULT 0.0,
                total_events_sourced INTEGER DEFAULT 0,
                verified_events INTEGER DEFAULT 0
            )
        """)

        # Events table (append-only for core fields)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                who TEXT NOT NULL,
                what TEXT NOT NULL,
                event_when TEXT NOT NULL,
                event_where TEXT NOT NULL,
                why TEXT,
                how TEXT,
                source_id TEXT NOT NULL,
                source_url TEXT,
                raw_text TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                verification_notes TEXT,
                domain TEXT NOT NULL,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
        """)

        # Event linkage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_links (
                event_id_1 TEXT NOT NULL,
                event_id_2 TEXT NOT NULL,
                link_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (event_id_1, event_id_2, link_type),
                FOREIGN KEY (event_id_1) REFERENCES events(event_id),
                FOREIGN KEY (event_id_2) REFERENCES events(event_id)
            )
        """)

        # Verification history (tracks changes to verification status)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                old_status TEXT,
                new_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                verified_by TEXT NOT NULL,
                verified_at TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_domain ON events(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_when ON events(event_when)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_hash ON events(content_hash)")

        conn.commit()
        conn.close()

    # ========== SOURCE MANAGEMENT ==========

    def register_source(self, source: Source) -> bool:
        """
        Register a new information source.

        HUMAN INPUT REQUIRED: Initial base_reliability score
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO sources
                (source_id, name, source_type, base_reliability, url_pattern, notes,
                 accuracy_track_record, total_events_sourced, verified_events)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source.source_id, source.name, source.source_type.value,
                source.base_reliability, source.url_pattern, source.notes,
                source.accuracy_track_record, source.total_events_sourced,
                source.verified_events
            ))
            conn.commit()
            self._source_cache[source.source_id] = source
            return True
        except sqlite3.IntegrityError:
            return False  # Source already exists
        finally:
            conn.close()

    def get_source(self, source_id: str) -> Optional[Source]:
        """Retrieve a source by ID."""
        if source_id in self._source_cache:
            return self._source_cache[source_id]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            source = Source(
                source_id=row[0],
                name=row[1],
                source_type=SourceType(row[2]),
                base_reliability=row[3],
                url_pattern=row[4],
                notes=row[5],
                accuracy_track_record=row[6],
                total_events_sourced=row[7],
                verified_events=row[8]
            )
            self._source_cache[source_id] = source
            return source
        return None

    # ========== EVENT MANAGEMENT ==========

    def add_event(self, event: Event) -> tuple[bool, str]:
        """
        Add a new event to the store.

        Returns (success, message).
        Checks for duplicates via content hash.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        content_hash = event.content_hash()

        # Check for duplicate
        cursor.execute("SELECT event_id FROM events WHERE content_hash = ?", (content_hash,))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return False, f"Duplicate event detected: {existing[0]}"

        # Verify source exists
        source = self.get_source(event.source_id)
        if not source:
            conn.close()
            return False, f"Unknown source: {event.source_id}"

        try:
            cursor.execute("""
                INSERT INTO events
                (event_id, who, what, event_when, event_where, why, how,
                 source_id, source_url, raw_text, verification_status,
                 verification_notes, domain, event_type, created_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                json.dumps(event.who),
                event.what,
                event.when.isoformat(),
                event.where,
                event.why,
                event.how,
                event.source_id,
                event.source_url,
                event.raw_text,
                event.verification_status.value,
                event.verification_notes,
                event.domain,
                event.event_type,
                event.created_at.isoformat(),
                content_hash
            ))

            # Update source stats
            cursor.execute("""
                UPDATE sources
                SET total_events_sourced = total_events_sourced + 1
                WHERE source_id = ?
            """, (event.source_id,))

            conn.commit()
            return True, event.event_id
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def get_event(self, event_id: str) -> Optional[Event]:
        """Retrieve an event by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Event(
                event_id=row[0],
                who=json.loads(row[1]),
                what=row[2],
                when=datetime.fromisoformat(row[3]),
                where=row[4],
                why=row[5],
                how=row[6],
                source_id=row[7],
                source_url=row[8],
                raw_text=row[9],
                verification_status=VerificationStatus(row[10]),
                verification_notes=row[11],
                domain=row[12],
                event_type=row[13],
                created_at=datetime.fromisoformat(row[14])
            )
        return None

    def query_events(
        self,
        domain: Optional[str] = None,
        who: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_verification: Optional[VerificationStatus] = None,
        limit: int = 100
    ) -> List[Event]:
        """Query events with filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        if who:
            query += " AND who LIKE ?"
            params.append(f'%"{who}"%')

        if start_date:
            query += " AND event_when >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND event_when <= ?"
            params.append(end_date.isoformat())

        if min_verification:
            verification_order = {
                VerificationStatus.VERIFIED: 5,
                VerificationStatus.HIGH_CONFIDENCE: 4,
                VerificationStatus.MEDIUM_CONFIDENCE: 3,
                VerificationStatus.LOW_CONFIDENCE: 2,
                VerificationStatus.UNVERIFIED: 1,
                VerificationStatus.DISPUTED: 0
            }
            min_level = verification_order[min_verification]
            valid_statuses = [s.value for s, level in verification_order.items() if level >= min_level]
            placeholders = ",".join("?" * len(valid_statuses))
            query += f" AND verification_status IN ({placeholders})"
            params.extend(valid_statuses)

        query += " ORDER BY event_when DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append(Event(
                event_id=row[0],
                who=json.loads(row[1]),
                what=row[2],
                when=datetime.fromisoformat(row[3]),
                where=row[4],
                why=row[5],
                how=row[6],
                source_id=row[7],
                source_url=row[8],
                raw_text=row[9],
                verification_status=VerificationStatus(row[10]),
                verification_notes=row[11],
                domain=row[12],
                event_type=row[13],
                created_at=datetime.fromisoformat(row[14])
            ))

        return events

    # ========== VERIFICATION (HUMAN GATE) ==========

    def update_verification(
        self,
        event_id: str,
        new_status: VerificationStatus,
        reason: str,
        verified_by: str
    ) -> bool:
        """
        Update verification status of an event.

        HUMAN INPUT REQUIRED: This is a human gate.
        The verified_by field must be a human identifier.

        This also updates the source's track record.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current status
        cursor.execute(
            "SELECT verification_status, source_id FROM events WHERE event_id = ?",
            (event_id,)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        old_status = row[0]
        source_id = row[1]

        # Update event
        cursor.execute("""
            UPDATE events
            SET verification_status = ?, verification_notes = verification_notes || ' | ' || ?
            WHERE event_id = ?
        """, (new_status.value, reason, event_id))

        # Record history
        cursor.execute("""
            INSERT INTO verification_history
            (event_id, old_status, new_status, reason, verified_by, verified_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event_id, old_status, new_status.value, reason,
            verified_by, datetime.now().isoformat()
        ))

        # Update source track record if verified
        if new_status == VerificationStatus.VERIFIED:
            cursor.execute("""
                UPDATE sources
                SET verified_events = verified_events + 1,
                    accuracy_track_record = CAST(verified_events + 1 AS REAL) / total_events_sourced
                WHERE source_id = ?
            """, (source_id,))

            # Invalidate source cache
            if source_id in self._source_cache:
                del self._source_cache[source_id]

        conn.commit()
        conn.close()
        return True

    # ========== EVENT LINKAGE ==========

    def link_events(
        self,
        event_id_1: str,
        event_id_2: str,
        link_type: str,
        confidence: float,
        created_by: str
    ) -> bool:
        """
        Link two related events.

        Link types:
        - "same_case": Same legal case, different events
        - "same_entity": Same judge/party, different cases
        - "precedent": One event cites/follows another
        - "contradiction": Events appear to conflict
        - "temporal": Sequential events in a process

        HUMAN INPUT CAN IMPROVE: Human-identified links are higher quality
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Store link in both directions for easy querying
            cursor.execute("""
                INSERT OR REPLACE INTO event_links
                (event_id_1, event_id_2, link_type, confidence, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_id_1, event_id_2, link_type, confidence,
                created_by, datetime.now().isoformat()
            ))

            cursor.execute("""
                INSERT OR REPLACE INTO event_links
                (event_id_1, event_id_2, link_type, confidence, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_id_2, event_id_1, link_type, confidence,
                created_by, datetime.now().isoformat()
            ))

            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def get_linked_events(self, event_id: str, link_type: Optional[str] = None) -> List[tuple]:
        """Get all events linked to this one."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if link_type:
            cursor.execute("""
                SELECT event_id_2, link_type, confidence, created_by
                FROM event_links
                WHERE event_id_1 = ? AND link_type = ?
            """, (event_id, link_type))
        else:
            cursor.execute("""
                SELECT event_id_2, link_type, confidence, created_by
                FROM event_links
                WHERE event_id_1 = ?
            """, (event_id,))

        results = cursor.fetchall()
        conn.close()
        return results

    # ========== STATISTICS ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sources")
        total_sources = cursor.fetchone()[0]

        cursor.execute("""
            SELECT verification_status, COUNT(*)
            FROM events
            GROUP BY verification_status
        """)
        verification_breakdown = dict(cursor.fetchall())

        cursor.execute("""
            SELECT domain, COUNT(*)
            FROM events
            GROUP BY domain
        """)
        domain_breakdown = dict(cursor.fetchall())

        conn.close()

        return {
            "total_events": total_events,
            "total_sources": total_sources,
            "verification_breakdown": verification_breakdown,
            "domain_breakdown": domain_breakdown
        }


# ========== TRAINING DATA COLLECTION HELPERS ==========

def create_source_labeling_task(sources: List[Dict]) -> Dict:
    """
    Generate a task for human labelers to score source reliability.

    Input: List of sources to label
    Output: Structured task for human labelers

    HUMAN TASK: Rate each source 0-100 on reliability and explain why.
    """
    return {
        "task_type": "source_reliability_labeling",
        "instructions": """
            For each source, provide:
            1. Reliability score (0-100)
            2. Reasoning for the score
            3. Any caveats or context

            Consider:
            - Is this an official/primary source?
            - Does it have editorial oversight?
            - What is its track record for accuracy?
            - Is it opinion or fact-based?
        """,
        "sources_to_label": sources,
        "output_format": {
            "source_id": "str",
            "reliability_score": "int 0-100",
            "reasoning": "str",
            "caveats": "str"
        }
    }


def create_event_verification_task(events: List[Event]) -> Dict:
    """
    Generate a task for human labelers to verify events.

    HUMAN TASK: Verify each event against available evidence.
    """
    return {
        "task_type": "event_verification",
        "instructions": """
            For each event, determine:
            1. Did this event actually happen as described?
            2. What is your confidence level?
            3. What evidence supports or contradicts it?

            Verification levels:
            - VERIFIED: Official record confirms or 3+ independent sources
            - HIGH: 2 independent sources or very reputable single source
            - MEDIUM: Single reputable source
            - LOW: Single unverified source
            - DISPUTED: Conflicting information found
        """,
        "events_to_verify": [asdict(e) for e in events],
        "output_format": {
            "event_id": "str",
            "verification_status": "enum",
            "confidence": "float 0-1",
            "evidence_summary": "str",
            "conflicting_evidence": "str or null"
        }
    }


def create_event_linkage_task(events: List[Event]) -> Dict:
    """
    Generate a task for human labelers to identify event relationships.

    HUMAN TASK: Identify which events are related and how.
    """
    return {
        "task_type": "event_linkage",
        "instructions": """
            Review these events and identify relationships:

            Link types:
            - same_case: Same legal case, different events
            - same_entity: Same judge/party across cases
            - precedent: One event cites or follows another
            - contradiction: Events appear to conflict
            - temporal: Sequential events in a process

            For each link, rate your confidence (0-1).
        """,
        "events_to_link": [asdict(e) for e in events],
        "output_format": {
            "event_id_1": "str",
            "event_id_2": "str",
            "link_type": "enum",
            "confidence": "float 0-1",
            "reasoning": "str"
        }
    }


if __name__ == "__main__":
    # Demo: Create store and add sample data
    store = EventStore("demo_events.db")

    # Register a source (HUMAN INPUT: reliability score)
    pacer = Source(
        source_id="pacer",
        name="PACER - Public Access to Court Electronic Records",
        source_type=SourceType.OFFICIAL_RECORD,
        base_reliability=0.99,  # Human labeled this
        url_pattern="https://ecf.*.uscourts.gov/*",
        notes="Official federal court records"
    )
    store.register_source(pacer)

    # Add an event
    event = Event(
        event_id="evt_gilstrap_2025_001",
        who=["Judge Rodney Gilstrap", "Samsung", "Collision Communications"],
        what="Jury verdict of $445,494,160 in patent infringement case",
        when=datetime(2025, 1, 15),
        where="E.D. Texas, Marshall Division",
        why="Jury found Samsung infringed six claims across four patents",
        how="Jury trial",
        source_id="pacer",
        source_url="https://ecf.txed.uscourts.gov/...",
        raw_text="The jury returned a verdict for plaintiff...",
        verification_status=VerificationStatus.VERIFIED,
        domain="judicial",
        event_type="jury_verdict"
    )

    success, msg = store.add_event(event)
    print(f"Added event: {success}, {msg}")

    # Show stats
    print(f"\nStore stats: {store.get_stats()}")
