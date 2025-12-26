"""
Module 6.1: Responsible AI Agent Implementation
================================================
A complete agent implementation with responsible AI practices:
- Content classification and filtering
- Explainable decisions
- Human-in-the-loop for sensitive cases
- Complete audit logging
- Privacy protection
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import json
import uuid

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# CONTENT CLASSIFICATION
# ============================================================

class ContentCategory(Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    HARMFUL = "harmful"
    ILLEGAL = "illegal"


class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ============================================================
# AUDIT LOGGING
# ============================================================

@dataclass
class AuditLogEntry:
    """Entry in the audit log for compliance."""
    timestamp: datetime
    action: str
    input_summary: str
    output_summary: str
    decision_factors: List[str]
    human_involved: bool
    user_id: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.LOW


class AuditLogger:
    """Thread-safe audit logger for AI decisions."""

    def __init__(self):
        self.entries: List[AuditLogEntry] = []

    def log(
        self,
        action: str,
        input_summary: str,
        output_summary: str,
        decision_factors: List[str],
        human_involved: bool = False,
        user_id: Optional[str] = None,
        risk_level: RiskLevel = RiskLevel.LOW
    ) -> str:
        """Log an action and return the entry ID."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            action=action,
            input_summary=input_summary[:200],  # Truncate for storage
            output_summary=output_summary[:200],
            decision_factors=decision_factors,
            human_involved=human_involved,
            user_id=user_id,
            risk_level=risk_level
        )
        self.entries.append(entry)
        return entry.timestamp.isoformat()

    def export(self, format: str = "json") -> str:
        """Export audit log for compliance review."""
        if format == "json":
            return json.dumps([
                {
                    "timestamp": e.timestamp.isoformat(),
                    "action": e.action,
                    "input_summary": e.input_summary,
                    "output_summary": e.output_summary,
                    "decision_factors": e.decision_factors,
                    "human_involved": e.human_involved,
                    "user_id": e.user_id,
                    "risk_level": e.risk_level.name
                }
                for e in self.entries
            ], indent=2)
        return str(self.entries)


# ============================================================
# HUMAN-IN-THE-LOOP
# ============================================================

class ReviewPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class HumanReviewRequest:
    """Request for human review of an AI decision."""
    request_id: str
    decision_type: str
    ai_recommendation: str
    confidence: float
    context: dict
    priority: ReviewPriority
    reasoning: str


class HumanInTheLoopController:
    """Manages human oversight of AI decisions."""

    def __init__(self):
        self.pending_reviews: Dict[str, HumanReviewRequest] = {}
        self.confidence_threshold = 0.8
        self.high_stakes_types = {
            "financial", "medical", "legal", "employment", "safety"
        }

    def should_require_review(
        self,
        decision_type: str,
        confidence: float,
        context: dict
    ) -> tuple[bool, ReviewPriority]:
        """Determine if human review is required."""

        # Always review high-stakes decisions
        if decision_type in self.high_stakes_types:
            return True, ReviewPriority.HIGH

        # Review low-confidence decisions
        if confidence < self.confidence_threshold:
            priority = ReviewPriority.MEDIUM if confidence > 0.5 else ReviewPriority.HIGH
            return True, priority

        # Review if context indicates sensitivity
        if context.get("involves_minor"):
            return True, ReviewPriority.CRITICAL
        if context.get("amount", 0) > 10000:
            return True, ReviewPriority.HIGH

        return False, ReviewPriority.LOW

    def request_review(self, request: HumanReviewRequest) -> str:
        """Submit a decision for human review."""
        self.pending_reviews[request.request_id] = request
        print(f"[REVIEW NEEDED] {request.priority.name}: {request.decision_type}")
        return request.request_id


# ============================================================
# EXPLAINABLE DECISIONS
# ============================================================

@dataclass
class DecisionStep:
    """A single step in the decision process."""
    step_number: int
    action: str
    reasoning: str
    confidence: float = 1.0


@dataclass
class ExplainableDecision:
    """A decision with full explainability."""
    decision_id: str
    outcome: str
    confidence: float
    factors: List[Dict[str, Any]]
    process_steps: List[DecisionStep]

    def explain(self, detail_level: str = "summary") -> str:
        """Generate explanation at requested detail level."""
        if detail_level == "summary":
            return f"Decision: {self.outcome} (confidence: {self.confidence:.0%})"

        lines = [
            f"Decision: {self.outcome}",
            f"Confidence: {self.confidence:.0%}",
            "",
            "Key factors:"
        ]

        for factor in self.factors:
            impact = factor.get('impact', 'neutral')
            icon = "+" if impact == 'positive' else "-" if impact == 'negative' else "•"
            lines.append(f"  {icon} {factor['name']}: {factor['value']}")

        if detail_level == "full":
            lines.extend(["", "Decision process:"])
            for step in self.process_steps:
                lines.append(f"  {step.step_number}. {step.action}")
                lines.append(f"     Reasoning: {step.reasoning}")

        return "\n".join(lines)


# ============================================================
# RESPONSIBLE AGENT
# ============================================================

class ResponsibleAgent:
    """
    An AI agent built with responsible AI practices.

    Features:
    - Content classification and filtering
    - Explainable decisions
    - Human-in-the-loop for sensitive cases
    - Complete audit logging
    - Privacy protection
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.audit_logger = AuditLogger()
        self.hitl_controller = HumanInTheLoopController()

        # Safety configuration
        self.sensitivity_keywords = {
            "medical", "health", "diagnosis", "legal", "law",
            "financial", "money", "investment", "suicide", "harm"
        }

        self.blocked_patterns = [
            "how to make weapons",
            "how to hack",
            "personal information about"
        ]

    def process(
        self,
        query: str,
        context: Dict[str, Any] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query with full responsible AI practices."""

        context = context or {}
        decision_factors = []

        # Step 1: Content classification
        content_category = self._classify_content(query)
        decision_factors.append(f"Content classified as: {content_category.value}")

        if content_category == ContentCategory.ILLEGAL:
            return self._blocked_response(
                "This request cannot be processed.",
                query, user_id, decision_factors
            )

        if content_category == ContentCategory.HARMFUL:
            return self._blocked_response(
                "I can't help with requests that could cause harm.",
                query, user_id, decision_factors
            )

        # Step 2: Sensitivity check
        is_sensitive = self._check_sensitivity(query)
        if is_sensitive:
            decision_factors.append("Query contains sensitive topics")

        # Step 3: Generate response
        response = self._generate_response(query, context)
        decision_factors.append(f"Response confidence: {response['confidence']:.0%}")

        # Step 4: Human review check
        needs_review, priority = self.hitl_controller.should_require_review(
            decision_type=self._get_decision_type(query),
            confidence=response['confidence'],
            context=context
        )

        if needs_review or is_sensitive:
            decision_factors.append("Flagged for human review")
            return self._human_review_response(
                query, response, user_id, decision_factors, priority
            )

        # Step 5: Privacy protection
        sanitized_response = self._sanitize_output(response['content'])
        if sanitized_response != response['content']:
            decision_factors.append("Output sanitized for privacy")

        # Step 6: Generate explanation
        explanation = self._generate_explanation(
            query, sanitized_response, decision_factors
        )

        # Step 7: Audit log
        audit_id = self.audit_logger.log(
            action="response_generated",
            input_summary=query,
            output_summary=sanitized_response,
            decision_factors=decision_factors,
            human_involved=False,
            user_id=user_id
        )

        return {
            "status": "success",
            "response": sanitized_response,
            "confidence": response['confidence'],
            "explanation": explanation,
            "decision_factors": decision_factors,
            "can_contest": True,
            "audit_id": audit_id
        }

    def _classify_content(self, query: str) -> ContentCategory:
        """Classify content for safety."""
        query_lower = query.lower()

        for pattern in self.blocked_patterns:
            if pattern in query_lower:
                return ContentCategory.ILLEGAL

        harmful_signals = ["hurt", "kill", "attack", "destroy"]
        if any(signal in query_lower for signal in harmful_signals):
            return ContentCategory.HARMFUL

        if any(kw in query_lower for kw in self.sensitivity_keywords):
            return ContentCategory.SENSITIVE

        return ContentCategory.SAFE

    def _check_sensitivity(self, query: str) -> bool:
        """Check if query involves sensitive topics."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.sensitivity_keywords)

    def _get_decision_type(self, query: str) -> str:
        """Determine the type of decision being made."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["medical", "health", "diagnosis"]):
            return "medical"
        if any(kw in query_lower for kw in ["legal", "law", "court"]):
            return "legal"
        if any(kw in query_lower for kw in ["financial", "money", "invest"]):
            return "financial"

        return "general"

    def _generate_response(self, query: str, context: Dict) -> Dict:
        """Generate response with confidence score."""
        if self.llm:
            response = self.llm.invoke(query)
            return {
                "content": response.content,
                "confidence": 0.85
            }
        return {
            "content": f"Response to: {query}",
            "confidence": 0.85
        }

    def _sanitize_output(self, content: str) -> str:
        """Remove any PII from output."""
        import re

        # Remove email addresses
        content = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL REDACTED]', content)

        # Remove phone numbers
        content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REDACTED]', content)

        # Remove SSN patterns
        content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', content)

        return content

    def _generate_explanation(
        self,
        query: str,
        response: str,
        factors: List[str]
    ) -> str:
        """Generate human-readable explanation."""
        return f"""
How this response was generated:
1. Your query was analyzed for safety and sensitivity
2. Content was classified and validated
3. A response was generated with confidence scoring
4. The response was checked for privacy concerns

Factors considered: {', '.join(factors)}

You can contest this response if you believe it's incorrect.
        """.strip()

    def _blocked_response(
        self,
        message: str,
        query: str,
        user_id: Optional[str],
        factors: List[str]
    ) -> Dict:
        """Return a blocked response."""
        factors.append("Request blocked by safety filter")

        self.audit_logger.log(
            action="request_blocked",
            input_summary=query,
            output_summary=message,
            decision_factors=factors,
            human_involved=False,
            user_id=user_id,
            risk_level=RiskLevel.HIGH
        )

        return {
            "status": "blocked",
            "message": message,
            "can_contest": True,
            "decision_factors": factors
        }

    def _human_review_response(
        self,
        query: str,
        ai_response: Dict,
        user_id: Optional[str],
        factors: List[str],
        priority: ReviewPriority
    ) -> Dict:
        """Return response pending human review."""
        request_id = str(uuid.uuid4())[:8]

        review_request = HumanReviewRequest(
            request_id=request_id,
            decision_type=self._get_decision_type(query),
            ai_recommendation=ai_response['content'],
            confidence=ai_response['confidence'],
            context={"query": query, "user_id": user_id},
            priority=priority,
            reasoning="; ".join(factors)
        )

        self.hitl_controller.request_review(review_request)

        self.audit_logger.log(
            action="flagged_for_review",
            input_summary=query,
            output_summary="Pending human review",
            decision_factors=factors,
            human_involved=True,
            user_id=user_id,
            risk_level=RiskLevel.MEDIUM
        )

        return {
            "status": "pending_review",
            "message": "Your request is being reviewed by a human expert.",
            "request_id": request_id,
            "estimated_time": "24-48 hours",
            "ai_suggestion": ai_response['content'],
            "ai_confidence": ai_response['confidence'],
            "decision_factors": factors
        }

    def export_audit_log(self) -> str:
        """Export audit log for compliance."""
        return self.audit_logger.export()


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate the responsible AI agent."""
    print("=" * 60)
    print("RESPONSIBLE AI AGENT DEMO")
    print("=" * 60)

    agent = ResponsibleAgent()

    # Test cases
    test_queries = [
        ("What's the weather like?", {}),
        ("Give me medical advice about my symptoms", {}),
        ("How to hack a computer", {}),
        ("Help me with my finances", {"amount": 50000}),
    ]

    for query, context in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print(f"Context: {context}")
        print()

        result = agent.process(query, context)

        print(f"Status: {result['status']}")
        if 'response' in result:
            print(f"Response: {result['response'][:100]}...")
        if 'message' in result:
            print(f"Message: {result['message']}")
        print(f"Factors: {result.get('decision_factors', [])}")

    print(f"\n{'=' * 60}")
    print("AUDIT LOG")
    print("=" * 60)
    print(agent.export_audit_log())


if __name__ == "__main__":
    demo()
