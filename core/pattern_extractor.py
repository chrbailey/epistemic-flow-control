"""
Pattern Extractor - Where LLM Meets Human Validation

This component extracts patterns from events using LLM analysis.
Every extraction is probabilistic and can be validated by humans.

THE KEY INSIGHT:
----------------
LLM output is a PROPOSAL. Humans VALIDATE.
The system learns from validation feedback to improve future proposals.

TRAINING DATA REQUIREMENTS:
---------------------------
1. EXTRACTION_EXAMPLES: Need ~100 labeled examples of:
   - Event text → Extracted pattern (human verified as correct)
   - Event text → Extracted pattern (human verified as incorrect, with correction)

   COLLECTION METHOD:
   - Start with LLM extraction on 100 events
   - Human reviews each, marks correct/incorrect
   - Incorrect ones get human correction
   - This becomes training data for prompt improvement

2. PATTERN_TYPE_TAXONOMY: Need labeled examples of pattern types
   - "judicial_preference": Judge shows consistent preference
   - "procedural_pattern": Timing/process consistency
   - "outcome_pattern": Consistent ruling direction
   - "behavioral_signal": Tone, strictness, leniency

   COLLECTION METHOD: Domain expert defines taxonomy, labels 50+ examples

3. CONFIDENCE_CALIBRATION: Need predictions with outcomes to calibrate
   - LLM says 0.8 confident → Was it right 80% of the time?
   - If not, we need to adjust

   COLLECTION METHOD: Track all predictions, measure actual accuracy per confidence bucket
"""

import json
import hashlib
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import re

# Import from event store
from .event_store import Event, EventStore, VerificationStatus

# Import LLM client (optional - graceful degradation if not available)
try:
    from llm import UnifiedLLMClient, LLMClientConfig, CompletionResult
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    UnifiedLLMClient = None
    LLMClientConfig = None
    CompletionResult = None

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns we can extract."""
    JUDICIAL_PREFERENCE = "preference"      # Judge prefers X over Y
    PROCEDURAL_PATTERN = "procedural"       # Timing, process consistency
    OUTCOME_PATTERN = "outcome"             # Consistent ruling direction
    BEHAVIORAL_SIGNAL = "behavioral"        # Tone, strictness
    TEMPORAL_TREND = "temporal"             # Pattern changing over time
    CONTEXTUAL = "contextual"               # Pattern depends on context
    ANOMALY = "anomaly"                     # Deviation from expected pattern


class ExtractionSource(Enum):
    """Who/what extracted this pattern?"""
    LLM_AUTOMATED = "llm_auto"              # LLM extracted automatically
    LLM_HUMAN_REVIEWED = "llm_reviewed"     # LLM extracted, human validated
    HUMAN_DIRECT = "human"                  # Human extracted directly
    RULE_BASED = "rule"                     # Deterministic rule extraction


@dataclass
class ExtractedPattern:
    """
    A pattern extracted from one or more events.

    This is PROPOSED by the LLM and VALIDATED by humans.
    """
    pattern_id: str
    source_event_ids: List[str]             # Events this pattern was extracted from

    # Pattern content
    pattern_type: PatternType
    subject: str                            # Who/what exhibits this pattern
    description: str                        # Human-readable description
    structured_pattern: Dict[str, Any]      # Machine-readable pattern

    # LLM extraction metadata
    extraction_source: ExtractionSource
    llm_confidence: float                   # LLM's stated confidence (0-1)
    extraction_reasoning: str               # LLM's explanation

    # Human validation (filled in during review)
    human_validated: bool = False
    human_validation_result: Optional[str] = None  # "correct", "incorrect", "partial"
    human_correction: Optional[str] = None  # If incorrect, what's the correction
    human_validator: Optional[str] = None   # Who validated
    validated_at: Optional[datetime] = None

    # Computed confidence (combines LLM + human validation)
    effective_confidence: float = 0.0

    # Metadata
    domain: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def compute_effective_confidence(self, calibration_factor: float = 1.0) -> float:
        """
        Compute effective confidence combining LLM and human validation.

        calibration_factor: Learned adjustment based on historical accuracy
        """
        base_confidence = self.llm_confidence * calibration_factor

        if not self.human_validated:
            # Not yet validated - use calibrated LLM confidence with penalty
            return base_confidence * 0.8  # 20% penalty for unvalidated

        if self.human_validation_result == "correct":
            # Human confirmed - boost confidence
            return min(1.0, base_confidence * 1.2)

        elif self.human_validation_result == "partial":
            # Partially correct - moderate confidence
            return base_confidence * 0.7

        else:  # incorrect
            # Human rejected - this should not be used
            return 0.0


@dataclass
class ExtractionPrompt:
    """
    Structured prompt for LLM pattern extraction.

    This is where we can inject training data learnings.
    """
    system_prompt: str
    event_context: str
    extraction_instructions: str
    output_format: str
    few_shot_examples: List[Dict[str, str]]


class PatternExtractor:
    """
    Extracts patterns from events using LLM analysis.

    HUMAN GATES:
    1. All extractions can be reviewed by humans
    2. Human corrections improve future extraction
    3. Calibration adjusts confidence based on track record
    """

    def __init__(
        self,
        event_store: EventStore,
        llm_client: Optional["UnifiedLLMClient"] = None,
        llm_config: Optional["LLMClientConfig"] = None,
        calibration_data_path: Optional[str] = None
    ):
        """
        Initialize the pattern extractor.

        Args:
            event_store: Event storage backend
            llm_client: Pre-configured UnifiedLLMClient (preferred)
            llm_config: LLM config to create client (alternative to llm_client)
            calibration_data_path: Path to calibration data JSON
        """
        self.event_store = event_store
        self.calibration_factors: Dict[PatternType, float] = {}
        self.extraction_history: List[ExtractedPattern] = []

        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        elif llm_config and LLM_AVAILABLE:
            self.llm_client = UnifiedLLMClient(llm_config)
        elif LLM_AVAILABLE:
            # Try to create with defaults (will use ANTHROPIC_API_KEY env var)
            try:
                self.llm_client = UnifiedLLMClient()
                logger.info("Created UnifiedLLMClient with default config")
            except Exception as e:
                logger.warning(f"Could not create LLM client: {e}. Using demo mode.")
                self.llm_client = None
        else:
            logger.warning("LLM module not available. Using demo mode (rule-based extraction only).")
            self.llm_client = None

        # Load calibration data if available
        if calibration_data_path:
            self._load_calibration_data(calibration_data_path)
        else:
            # Default calibration (conservative)
            for pt in PatternType:
                self.calibration_factors[pt] = 0.85  # 15% discount on LLM confidence

    def _load_calibration_data(self, path: str):
        """Load calibration factors from historical validation data."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for pt_name, factor in data.get("calibration_factors", {}).items():
                    try:
                        pt = PatternType(pt_name)
                        self.calibration_factors[pt] = factor
                    except ValueError:
                        pass
        except FileNotFoundError:
            # Use defaults
            for pt in PatternType:
                self.calibration_factors[pt] = 0.85

    def build_extraction_prompt(
        self,
        events: List[Event],
        pattern_type: Optional[PatternType] = None,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> ExtractionPrompt:
        """
        Build structured prompt for LLM extraction.

        TRAINING DATA INJECTION POINT:
        few_shot_examples should come from human-validated extractions
        """

        # Default few-shot examples (should be replaced with real training data)
        default_examples = [
            {
                "event": "Judge Gilstrap granted summary judgment in 3 of 47 patent cases in 2023",
                "pattern": {
                    "type": "outcome_pattern",
                    "subject": "Judge Gilstrap",
                    "description": "Rarely grants summary judgment in patent cases (6.4% rate)",
                    "structured": {
                        "metric": "summary_judgment_grant_rate",
                        "value": 0.064,
                        "sample_size": 47,
                        "time_period": "2023"
                    },
                    "confidence": 0.92,
                    "reasoning": "Large sample size (47 cases) provides statistical significance"
                }
            },
            {
                "event": "Judge Alsup required defendant to submit tutorial on neural network architecture before Markman hearing",
                "pattern": {
                    "type": "preference",
                    "subject": "Judge Alsup",
                    "description": "Demands technical education before complex tech hearings",
                    "structured": {
                        "behavior": "technical_tutorial_requirement",
                        "context": "complex_technology_cases",
                        "frequency": "when_needed"
                    },
                    "confidence": 0.78,
                    "reasoning": "Single event but consistent with known tech-savvy reputation"
                }
            }
        ]

        examples = few_shot_examples or default_examples

        system_prompt = """You are a pattern extraction system for legal intelligence.

Your job is to extract PATTERNS from EVENTS. A pattern is a consistent tendency, preference,
or behavior that can be used to predict future actions.

KEY PRINCIPLES:
1. Only extract patterns supported by the evidence
2. State confidence honestly - don't overstate
3. Explain your reasoning
4. If uncertain, say so explicitly
5. Distinguish between one-time events and true patterns

OUTPUT MUST BE VALID JSON."""

        event_context = "\n\n".join([
            f"EVENT {i+1}:\n"
            f"- What: {e.what}\n"
            f"- Who: {', '.join(e.who)}\n"
            f"- When: {e.when.strftime('%Y-%m-%d')}\n"
            f"- Where: {e.where}\n"
            f"- Why: {e.why or 'Not stated'}\n"
            f"- Source: {e.source_id} (Verification: {e.verification_status.value})"
            for i, e in enumerate(events)
        ])

        pattern_filter = ""
        if pattern_type:
            pattern_filter = f"\nFOCUS ON: {pattern_type.value} patterns only."

        extraction_instructions = f"""Analyze the following events and extract any patterns.
{pattern_filter}

For each pattern found, provide:
1. pattern_type: One of [preference, procedural, outcome, behavioral, temporal, contextual, anomaly]
2. subject: Who/what exhibits this pattern
3. description: Clear, human-readable description
4. structured_pattern: Machine-readable representation with specific metrics
5. confidence: Your confidence (0.0 to 1.0) that this is a TRUE PATTERN, not a one-time event
6. reasoning: Why you believe this is a pattern, and any caveats

CRITICAL:
- A single event is NOT a pattern unless it confirms a known tendency
- Low confidence (< 0.6) is appropriate for limited evidence
- State what ADDITIONAL evidence would increase confidence"""

        output_format = """{
    "patterns": [
        {
            "pattern_type": "string",
            "subject": "string",
            "description": "string",
            "structured_pattern": {
                "metric_name": "value",
                ...
            },
            "confidence": 0.0-1.0,
            "reasoning": "string",
            "evidence_strength": "single_event | few_events | many_events | statistical",
            "caveats": ["string", ...]
        }
    ],
    "no_patterns_found": boolean,
    "additional_context_needed": ["string", ...]
}"""

        examples_text = "\n\nEXAMPLES:\n" + "\n---\n".join([
            f"Event: {ex['event']}\nExtracted Pattern: {json.dumps(ex['pattern'], indent=2)}"
            for ex in examples
        ])

        return ExtractionPrompt(
            system_prompt=system_prompt,
            event_context=event_context,
            extraction_instructions=extraction_instructions + examples_text,
            output_format=output_format,
            few_shot_examples=examples
        )

    async def extract_patterns_async(
        self,
        events: List[Event],
        pattern_type: Optional[PatternType] = None,
        auto_validate_threshold: float = 0.95
    ) -> List[ExtractedPattern]:
        """
        Extract patterns from events using LLM (async version).

        HUMAN GATE: Patterns below auto_validate_threshold require human review.

        Args:
            events: List of events to extract patterns from
            pattern_type: Optional filter for specific pattern type
            auto_validate_threshold: Confidence threshold for automatic validation

        Returns:
            List of ExtractedPattern objects
        """
        if not events:
            return []

        prompt = self.build_extraction_prompt(events, pattern_type)

        # If we have an LLM client, use it
        if self.llm_client:
            response = await self._call_llm_async(prompt)
            patterns = self._parse_llm_response(response, events)

            if not patterns:
                # LLM didn't return valid patterns, fall back to rule-based
                logger.warning("LLM extraction returned no patterns, using rule-based fallback")
                patterns = self._demo_extraction(events)
        else:
            # Demo mode: rule-based extraction
            patterns = self._demo_extraction(events)

        # Apply calibration and determine validation status
        for pattern in patterns:
            calibration = self.calibration_factors.get(pattern.pattern_type, 0.85)
            pattern.effective_confidence = pattern.compute_effective_confidence(calibration)

            # Auto-validate only if extremely high confidence
            if pattern.effective_confidence >= auto_validate_threshold:
                pattern.human_validated = True
                pattern.human_validation_result = "auto_validated"
                pattern.human_validator = "system_auto"
                pattern.validated_at = datetime.now()

        self.extraction_history.extend(patterns)
        return patterns

    def extract_patterns(
        self,
        events: List[Event],
        pattern_type: Optional[PatternType] = None,
        auto_validate_threshold: float = 0.95
    ) -> List[ExtractedPattern]:
        """
        Extract patterns from events using LLM (sync wrapper).

        HUMAN GATE: Patterns below auto_validate_threshold require human review.

        Note: This is a synchronous wrapper around extract_patterns_async().
        For better performance in async contexts, use extract_patterns_async() directly.

        Returns list of ExtractedPattern objects.
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - need to handle differently
            # Create a new task and run it
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.extract_patterns_async(events, pattern_type, auto_validate_threshold)
                )
                return future.result()
        except RuntimeError:
            # No event loop running - safe to use asyncio.run
            return asyncio.run(
                self.extract_patterns_async(events, pattern_type, auto_validate_threshold)
            )

    async def _call_llm_async(self, prompt: ExtractionPrompt) -> str:
        """
        Call LLM API for pattern extraction (async).

        This is the real implementation using UnifiedLLMClient.
        Includes robust error handling and JSON validation.

        Args:
            prompt: Structured extraction prompt

        Returns:
            Raw JSON response string from LLM
        """
        if not self.llm_client:
            logger.warning("No LLM client configured, returning empty response")
            return "{}"

        # Build the user prompt from components
        user_prompt = (
            f"EVENTS TO ANALYZE:\n{prompt.event_context}\n\n"
            f"INSTRUCTIONS:\n{prompt.extraction_instructions}\n\n"
            f"OUTPUT FORMAT:\n{prompt.output_format}"
        )

        try:
            # Use the unified client's JSON completion method
            # This handles retries, rate limiting, and JSON parsing automatically
            result = await self.llm_client.complete_json(
                prompt=user_prompt,
                system_prompt=prompt.system_prompt,
                expected_fields=["patterns", "no_patterns_found"],
                max_tokens=4096,
                temperature=0.0,  # Deterministic for extraction
                metadata={
                    "task": "pattern_extraction",
                    "event_count": len(prompt.few_shot_examples),
                }
            )

            if result.success and result.data:
                # Return the raw content for parsing by _parse_llm_response
                # The JSON has already been validated by complete_json
                logger.debug(
                    f"LLM extraction successful: {result.output_tokens} tokens, "
                    f"${result.estimated_cost_usd:.6f}"
                )
                return result.content
            else:
                # Log the error but don't raise - we'll fall back to rule-based
                logger.warning(
                    f"LLM extraction failed: {result.error_type}: {result.error}"
                )
                return "{}"

        except Exception as e:
            logger.exception(f"Unexpected error in LLM extraction: {e}")
            return "{}"

    def _call_llm(self, prompt: ExtractionPrompt) -> str:
        """
        Call LLM API for extraction (sync wrapper).

        DEPRECATED: Use _call_llm_async() for new code.
        This method exists for backwards compatibility.
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._call_llm_async(prompt))
                return future.result()
        except RuntimeError:
            return asyncio.run(self._call_llm_async(prompt))

    def _parse_llm_response(self, response: str, source_events: List[Event]) -> List[ExtractedPattern]:
        """
        Parse LLM response into ExtractedPattern objects.

        Uses the robust JSON parser to handle malformed responses from LLMs.
        Falls back to partial extraction if full parsing fails.
        """
        patterns = []

        if not response or response.strip() == "{}":
            return patterns

        # Use robust JSON parser if available
        if LLM_AVAILABLE:
            from llm import parse_json_response
            parse_result = parse_json_response(
                response,
                expected_fields=["patterns", "no_patterns_found"]
            )

            if parse_result.is_success() and parse_result.data:
                data = parse_result.data
                if parse_result.recovered_issues:
                    logger.debug(f"JSON parsing recovered: {parse_result.recovered_issues}")
            else:
                logger.warning(f"JSON parsing failed: {parse_result.errors}")
                return patterns
        else:
            # Fallback to basic JSON parsing
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return patterns

        # Extract patterns from parsed data
        for p in data.get("patterns", []):
            try:
                # Parse pattern type with fallback
                pattern_type_str = p.get("pattern_type", "contextual")
                try:
                    pattern_type = PatternType(pattern_type_str)
                except ValueError:
                    # Unknown pattern type - map to closest or use contextual
                    pattern_type_map = {
                        "preference": PatternType.JUDICIAL_PREFERENCE,
                        "behavioral": PatternType.BEHAVIORAL_SIGNAL,
                        "outcome": PatternType.OUTCOME_PATTERN,
                        "procedural": PatternType.PROCEDURAL_PATTERN,
                        "temporal": PatternType.TEMPORAL_TREND,
                        "anomaly": PatternType.ANOMALY,
                    }
                    pattern_type = pattern_type_map.get(pattern_type_str, PatternType.CONTEXTUAL)

                # Parse confidence with bounds checking
                raw_confidence = p.get("confidence", 0.5)
                try:
                    confidence = max(0.0, min(1.0, float(raw_confidence)))
                except (ValueError, TypeError):
                    confidence = 0.5

                pattern = ExtractedPattern(
                    pattern_id=self._generate_pattern_id(p),
                    source_event_ids=[e.event_id for e in source_events],
                    pattern_type=pattern_type,
                    subject=str(p.get("subject", "unknown")),
                    description=str(p.get("description", "")),
                    structured_pattern=p.get("structured_pattern", {}),
                    extraction_source=ExtractionSource.LLM_AUTOMATED,
                    llm_confidence=confidence,
                    extraction_reasoning=str(p.get("reasoning", "")),
                    domain=source_events[0].domain if source_events else ""
                )
                patterns.append(pattern)

            except Exception as e:
                logger.warning(f"Error parsing individual pattern: {e}")
                continue

        return patterns

    def _demo_extraction(self, events: List[Event]) -> List[ExtractedPattern]:
        """
        Rule-based extraction without LLM.

        Extracts patterns from event text by analyzing:
        - Grant/deny outcomes
        - Motion types
        - Judge behavior patterns
        """
        if not events:
            return []

        patterns = []

        for event in events:
            # Extract subject (judge) from who field
            subject = None
            for who in event.who:
                if "Judge" in who:
                    subject = who
                    break

            if not subject:
                continue

            # Analyze the event text for outcome patterns
            what_lower = event.what.lower()

            # Determine outcome type
            outcome = None
            if "granted" in what_lower or "grant" in what_lower:
                outcome = "grant"
            elif "denied" in what_lower or "deny" in what_lower:
                outcome = "deny"
            elif "scheduled" in what_lower:
                outcome = "schedule"

            if not outcome:
                continue

            # Determine motion type
            motion_type = event.event_type or "general"
            if "summary judgment" in what_lower or "summary_judgment" in motion_type:
                motion_type = "summary_judgment"
            elif "motion to dismiss" in what_lower or "motion_to_dismiss" in motion_type:
                motion_type = "motion_to_dismiss"
            elif "injunction" in what_lower:
                motion_type = "injunction"
            elif "markman" in what_lower or "claim construction" in what_lower:
                motion_type = "claim_construction"

            # Create pattern based on outcome
            pattern_key = f"{subject}|{motion_type}|{outcome}"

            # Compute confidence based on source reliability
            source = self.event_store.get_source(event.source_id) if self.event_store else None
            base_confidence = 0.6  # Base for single observation
            if source:
                # Higher reliability source = higher confidence
                base_confidence = 0.5 + (source.effective_reliability() * 0.3)

            pattern = ExtractedPattern(
                pattern_id=self._generate_pattern_id({
                    "subject": subject,
                    "motion": motion_type,
                    "outcome": outcome,
                    "event_id": event.event_id
                }),
                source_event_ids=[event.event_id],
                pattern_type=PatternType.OUTCOME_PATTERN,
                subject=subject,
                description=f"{subject} {outcome}s {motion_type.replace('_', ' ')} motions",
                structured_pattern={
                    "judge": subject,
                    "motion_type": motion_type,
                    "outcome": outcome,
                    "observation_count": 1
                },
                extraction_source=ExtractionSource.RULE_BASED,
                llm_confidence=base_confidence,
                extraction_reasoning=f"Extracted from event: {event.what[:50]}...",
                domain=event.domain or "judicial"
            )
            patterns.append(pattern)

        return patterns

    def _generate_pattern_id(self, pattern_data: Dict) -> str:
        """Generate deterministic ID for pattern."""
        content = json.dumps(pattern_data, sort_keys=True)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"pat_{hash_val}"

    # ========== HUMAN VALIDATION INTERFACE ==========

    def get_patterns_needing_review(self, max_count: int = 10) -> List[ExtractedPattern]:
        """Get patterns that need human review."""
        return [
            p for p in self.extraction_history
            if not p.human_validated
        ][:max_count]

    def submit_human_validation(
        self,
        pattern_id: str,
        validation_result: str,  # "correct", "incorrect", "partial"
        validator: str,
        correction: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Submit human validation for a pattern.

        THIS IS THE KEY HUMAN GATE.
        Human validation:
        1. Confirms or rejects LLM extraction
        2. Provides corrections for incorrect extractions
        3. Creates training data for future improvement
        """
        for pattern in self.extraction_history:
            if pattern.pattern_id == pattern_id:
                pattern.human_validated = True
                pattern.human_validation_result = validation_result
                pattern.human_validator = validator
                pattern.validated_at = datetime.now()
                pattern.human_correction = correction

                # Recompute effective confidence
                calibration = self.calibration_factors.get(pattern.pattern_type, 0.85)
                pattern.effective_confidence = pattern.compute_effective_confidence(calibration)

                return True

        return False

    def export_validation_data(self) -> List[Dict]:
        """
        Export validation data for training.

        TRAINING DATA OUTPUT:
        This is what we use to improve extraction prompts.
        """
        return [
            {
                "pattern_id": p.pattern_id,
                "source_events": p.source_event_ids,
                "pattern_type": p.pattern_type.value,
                "llm_extraction": {
                    "description": p.description,
                    "structured": p.structured_pattern,
                    "confidence": p.llm_confidence,
                    "reasoning": p.extraction_reasoning
                },
                "human_validation": {
                    "validated": p.human_validated,
                    "result": p.human_validation_result,
                    "correction": p.human_correction,
                    "validator": p.human_validator
                },
                "effective_confidence": p.effective_confidence
            }
            for p in self.extraction_history
            if p.human_validated  # Only export validated ones
        ]


# ========== TRAINING DATA COLLECTION ==========

def create_extraction_labeling_task(events: List[Event]) -> Dict:
    """
    Create a task for human labelers to validate/correct pattern extractions.

    WORKFLOW:
    1. LLM extracts patterns from events
    2. Human reviews each extraction
    3. Human marks correct/incorrect/partial
    4. Human provides corrections for incorrect ones
    5. Results become training data

    THIS IS WHERE "AUNT/UNCLE" DATA COMES IN:
    A few expert corrections dramatically improve future extraction.
    """
    return {
        "task_type": "pattern_extraction_validation",
        "instructions": """
            For each LLM-extracted pattern:

            1. Read the source event(s)
            2. Read the extracted pattern
            3. Evaluate:
               - CORRECT: The pattern accurately captures what the events show
               - PARTIAL: The pattern is partially right but needs refinement
               - INCORRECT: The pattern misinterprets the events

            4. If PARTIAL or INCORRECT, provide correction:
               - What should the pattern say instead?
               - What did the LLM miss or misunderstand?

            5. Rate your confidence in your evaluation (0-100)

            IMPORTANT: Even a few corrections help enormously.
            Focus on the ones where LLM is clearly wrong.
        """,
        "events": [asdict(e) for e in events],
        "output_format": {
            "pattern_id": "str",
            "validation_result": "correct | partial | incorrect",
            "correction": "str or null",
            "correction_reasoning": "str",
            "evaluator_confidence": "int 0-100"
        }
    }


def create_pattern_type_labeling_task(patterns: List[ExtractedPattern]) -> Dict:
    """
    Create a task for human labelers to verify pattern type classification.

    This helps train the LLM to correctly categorize patterns.
    """
    return {
        "task_type": "pattern_type_classification",
        "instructions": """
            For each pattern, verify the type classification:

            TYPES:
            - preference: Subject consistently prefers X over Y
            - procedural: Timing, process, or procedure consistency
            - outcome: Consistent direction of decisions/rulings
            - behavioral: Tone, strictness, temperament signals
            - temporal: Pattern that's changing over time
            - contextual: Pattern depends on specific context
            - anomaly: Deviation from expected behavior

            If the type is wrong, provide the correct type and explain why.
        """,
        "patterns": [asdict(p) for p in patterns],
        "output_format": {
            "pattern_id": "str",
            "llm_type": "str",
            "correct_type": "str",
            "type_is_correct": "bool",
            "reasoning": "str"
        }
    }


def calculate_calibration_factors(validation_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate calibration factors from validation data.

    CALIBRATION FORMULA:
    If LLM says 80% confident, and it's actually correct 60% of the time,
    calibration_factor = 0.60 / 0.80 = 0.75

    Apply this factor to future confidence scores.

    THIS IS THE "LEARNING" STEP:
    Human validation data teaches us how much to trust LLM confidence.
    """
    # Group by pattern type
    by_type: Dict[str, List[Dict]] = {}
    for item in validation_data:
        pt = item.get("pattern_type", "unknown")
        if pt not in by_type:
            by_type[pt] = []
        by_type[pt].append(item)

    calibration_factors = {}

    for pattern_type, items in by_type.items():
        if len(items) < 10:
            # Not enough data for reliable calibration
            calibration_factors[pattern_type] = 0.85  # Conservative default
            continue

        # Bucket by LLM confidence
        buckets = {
            "high": {"stated": [], "actual": []},    # 0.8-1.0
            "medium": {"stated": [], "actual": []},  # 0.5-0.8
            "low": {"stated": [], "actual": []}      # 0.0-0.5
        }

        for item in items:
            llm_conf = item.get("llm_extraction", {}).get("confidence", 0.5)
            is_correct = item.get("human_validation", {}).get("result") == "correct"

            if llm_conf >= 0.8:
                bucket = "high"
            elif llm_conf >= 0.5:
                bucket = "medium"
            else:
                bucket = "low"

            buckets[bucket]["stated"].append(llm_conf)
            buckets[bucket]["actual"].append(1.0 if is_correct else 0.0)

        # Calculate overall calibration factor
        total_stated = sum(sum(b["stated"]) for b in buckets.values())
        total_actual = sum(sum(b["actual"]) for b in buckets.values())

        if total_stated > 0:
            calibration_factors[pattern_type] = total_actual / total_stated
        else:
            calibration_factors[pattern_type] = 0.85

    return calibration_factors


if __name__ == "__main__":
    from .event_store import EventStore, Source, SourceType

    # Demo
    store = EventStore("demo_events.db")
    extractor = PatternExtractor(store)

    # Create sample events
    events = [
        Event(
            event_id="evt_demo_001",
            who=["Judge Smith"],
            what="Granted motion for summary judgment",
            when=datetime(2024, 1, 15),
            where="N.D. Cal",
            why="Plaintiff failed to establish genuine issue of material fact",
            how="Written order",
            source_id="pacer",
            source_url="",
            raw_text="Order granting MSJ...",
            verification_status=VerificationStatus.VERIFIED,
            domain="judicial",
            event_type="order"
        ),
        Event(
            event_id="evt_demo_002",
            who=["Judge Smith"],
            what="Granted motion for summary judgment",
            when=datetime(2024, 3, 20),
            where="N.D. Cal",
            why="No triable issues remain",
            how="Written order",
            source_id="pacer",
            source_url="",
            raw_text="Order granting MSJ...",
            verification_status=VerificationStatus.VERIFIED,
            domain="judicial",
            event_type="order"
        )
    ]

    # Extract patterns
    patterns = extractor.extract_patterns(events)

    print(f"Extracted {len(patterns)} patterns")
    for p in patterns:
        print(f"  - {p.description} (confidence: {p.effective_confidence:.2f})")

    # Show patterns needing review
    needs_review = extractor.get_patterns_needing_review()
    print(f"\n{len(needs_review)} patterns need human review")
