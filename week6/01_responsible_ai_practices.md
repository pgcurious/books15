# Module 6.1: Responsible AI Practices

> "With great power comes great responsibility." — Voltaire (often attributed to Spider-Man)

## What You'll Learn

- Why responsible AI is a business necessity, not just ethics
- Core principles of AI ethics and how to implement them
- Regulatory landscape: GDPR, EU AI Act, and industry standards
- Designing for transparency, explainability, and accountability
- Human-in-the-loop patterns for appropriate oversight
- Building trust through responsible practices

---

## First Principles: What Makes AI "Responsible"?

Let's start from fundamentals. Why does responsible AI matter, and what does it actually mean?

### The Three Pillars of Responsible AI

```
RESPONSIBLE AI = SAFETY + TRUST + ACCOUNTABILITY

Where:
├── SAFETY: The system doesn't cause harm
│   ├── Physical safety (no dangerous actions)
│   ├── Psychological safety (no harmful content)
│   ├── Economic safety (no financial damage)
│   └── Social safety (no discrimination/bias)
│
├── TRUST: Stakeholders can rely on the system
│   ├── Reliability (consistent performance)
│   ├── Transparency (understandable behavior)
│   ├── Predictability (expected outcomes)
│   └── Honesty (admits limitations)
│
└── ACCOUNTABILITY: Clear responsibility for outcomes
    ├── Traceability (who made what decision)
    ├── Explainability (why decisions were made)
    ├── Auditability (verifiable compliance)
    └── Remediation (ability to correct mistakes)
```

### Why Responsible AI is a Business Necessity

```
The Business Case for Responsible AI:

1. REGULATORY COMPLIANCE
   ├── EU AI Act: Fines up to 6% of global revenue
   ├── GDPR: Right to explanation for automated decisions
   └── Industry regulations: Healthcare, finance, legal

2. REPUTATION PROTECTION
   ├── AI failures go viral instantly
   ├── Trust takes years to build, seconds to destroy
   └── Customers increasingly demand ethical AI

3. OPERATIONAL RESILIENCE
   ├── Safe systems have fewer failures
   ├── Transparent systems are easier to debug
   └── Accountable systems enable learning

4. COMPETITIVE ADVANTAGE
   ├── Trustworthy AI attracts enterprise customers
   ├── Responsible practices attract top talent
   └── Proactive compliance avoids future costs
```

### The Cost of Irresponsible AI

| Incident Type | Impact | Example |
|---------------|--------|---------|
| **Bias** | Lawsuits, reputation damage | Amazon hiring tool discrimination |
| **Hallucination** | Wrong decisions, liability | Legal cases citing fake precedents |
| **Privacy Breach** | Fines, customer loss | PII in training data leaked |
| **Safety Failure** | Harm, criminal liability | Autonomous vehicle accidents |
| **Manipulation** | Regulatory action | Deepfakes, misinformation |

---

## Analogical Thinking: AI Ethics as Medical Ethics

The medical profession has centuries of experience with ethics in high-stakes decisions. We can learn from their frameworks.

### The Hippocratic Oath for AI

```
MEDICAL ETHICS                      AI ETHICS
─────────────────────────────────────────────────────────────────

"First, do no harm"                 Safety First
├── Consider all risks              ├── Risk assessment
├── Minimize side effects           ├── Minimize unintended effects
└── Err on side of caution          └── Fail safe, not fail deadly

"Informed consent"                  Transparency
├── Explain procedures              ├── Explain AI decisions
├── Discuss alternatives            ├── Show confidence levels
└── Patient decides                 └── User maintains control

"Beneficence"                       Beneficial AI
├── Act in patient's interest       ├── Serve user's true needs
├── Balance benefit vs risk         ├── Optimize for user welfare
└── Promote wellbeing               └── Consider societal impact

"Non-maleficence"                   Harm Prevention
├── Don't cause unnecessary harm    ├── Block harmful outputs
├── Protect vulnerable patients     ├── Protect vulnerable users
└── Report adverse events           └── Log and report incidents

"Confidentiality"                   Privacy Protection
├── Protect patient information     ├── Protect user data
├── Share only with consent         ├── Minimize data collection
└── Secure records                  └── Encrypt and access control

"Justice"                           Fairness
├── Equal treatment                 ├── Unbiased responses
├── Fair resource allocation        ├── Equal access
└── Non-discrimination              └── No protected class bias
```

### The AI Ethics Review Board (Like Hospital Ethics Committees)

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI ETHICS REVIEW BOARD                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Role: Review high-risk AI decisions and policies               │
│                                                                  │
│  Members:                                                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   Technical  │ │    Legal     │ │    Domain    │            │
│  │   Lead       │ │    Counsel   │ │    Expert    │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   Ethics     │ │   User       │ │   Executive  │            │
│  │   Specialist │ │   Advocate   │ │   Sponsor    │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
│  Reviews:                                                        │
│  • New AI feature deployments                                   │
│  • High-risk use case expansions                                │
│  • Incident post-mortems                                        │
│  • Policy changes                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Emergence Thinking: Ethics from Simple Rules

Complex ethical behavior emerges from simple, well-designed rules:

### The Rule Stack

```
Simple Rules                        →  Emergent Ethical Behavior
─────────────────────────────────────────────────────────────────────

Level 1: HARD CONSTRAINTS (Never violate)
├── "Never generate CSAM"           →  Absolute safety boundary
├── "Never assist with violence"    →  Physical harm prevention
└── "Never reveal private data"     →  Privacy protection

Level 2: SOFT CONSTRAINTS (Strong preference)
├── "Admit uncertainty"             →  Honest AI
├── "Cite sources when possible"    →  Verifiable claims
└── "Defer to humans on ethics"     →  Appropriate humility

Level 3: GUIDELINES (Best practices)
├── "Explain your reasoning"        →  Transparency
├── "Consider multiple viewpoints"  →  Reduced bias
└── "Suggest human review for high-stakes" → Accountability

                Combined, these produce:

                ┌────────────────────────────────────────┐
                │                                        │
                │   ETHICAL AI AGENT                     │
                │                                        │
                │   - Won't cross safety lines           │
                │   - Self-aware of limitations          │
                │   - Transparent in reasoning           │
                │   - Appropriately cautious             │
                │   - Continuously learning              │
                │                                        │
                └────────────────────────────────────────┘
```

---

## Regulatory Landscape

Understanding the rules that govern AI systems:

### EU AI Act (2024)

```
EU AI ACT RISK CATEGORIES
─────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────┐
│  UNACCEPTABLE RISK (Banned)                                      │
│  ─────────────────────────────                                   │
│  • Social scoring systems                                        │
│  • Real-time biometric identification in public spaces          │
│  • Manipulation of vulnerable groups                             │
│  • AI that exploits behavioral vulnerabilities                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  HIGH RISK (Heavy regulation)                                    │
│  ────────────────────────────                                    │
│  • Critical infrastructure (energy, transport)                  │
│  • Education (exam scoring, admissions)                         │
│  • Employment (recruitment, performance)                        │
│  • Essential services (credit, insurance)                       │
│  • Law enforcement applications                                 │
│  • Migration and border control                                 │
│                                                                  │
│  Requirements:                                                   │
│  ✓ Risk assessment and mitigation                               │
│  ✓ High-quality training data                                   │
│  ✓ Logging and traceability                                     │
│  ✓ Transparency and user information                            │
│  ✓ Human oversight                                              │
│  ✓ Accuracy, robustness, cybersecurity                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LIMITED RISK (Transparency obligations)                         │
│  ─────────────────────────────────────────                       │
│  • Chatbots (must disclose AI nature)                           │
│  • Emotion recognition systems                                   │
│  • Biometric categorization                                      │
│  • Deep fakes (must be labeled)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MINIMAL RISK (No specific requirements)                         │
│  ───────────────────────────────────────                        │
│  • AI-enabled video games                                        │
│  • Spam filters                                                  │
│  • Most business applications                                    │
└─────────────────────────────────────────────────────────────────┘
```

### GDPR Requirements for AI

```python
# GDPR Article 22: Automated Decision-Making

# Users have the right to:
# 1. Not be subject to purely automated decisions with significant effects
# 2. Obtain human intervention
# 3. Express their point of view
# 4. Obtain an explanation of the decision
# 5. Contest the decision

class GDPRCompliantAgent:
    """Agent that complies with GDPR automated decision-making rules."""

    def make_decision(self, user_id: str, context: dict) -> dict:
        """Make a decision with GDPR compliance."""

        # 1. Check if human intervention is required
        if self._is_significant_decision(context):
            return self._route_to_human(user_id, context)

        # 2. Generate decision with explanation
        decision = self._generate_decision(context)
        explanation = self._generate_explanation(decision, context)

        # 3. Log for auditability
        self._log_decision(user_id, context, decision, explanation)

        # 4. Return with contestation option
        return {
            "decision": decision,
            "explanation": explanation,
            "can_contest": True,
            "contest_url": f"/contest/{decision['id']}",
            "human_review_available": True
        }

    def _is_significant_decision(self, context: dict) -> bool:
        """Determine if decision has significant effects."""
        significant_categories = [
            "credit_decision",
            "employment_decision",
            "insurance_decision",
            "legal_decision"
        ]
        return context.get("category") in significant_categories

    def _route_to_human(self, user_id: str, context: dict) -> dict:
        """Route significant decisions to human review."""
        return {
            "status": "pending_human_review",
            "message": "This decision requires human review",
            "estimated_time": "24-48 hours",
            "reference_id": self._create_review_ticket(user_id, context)
        }
```

### Industry-Specific Regulations

| Industry | Regulations | AI Requirements |
|----------|-------------|-----------------|
| **Healthcare** | HIPAA, FDA | Privacy, clinical validation, explainability |
| **Finance** | SOX, Basel III, Fair Lending | Audit trails, bias testing, model validation |
| **Legal** | Bar rules, court requirements | Accuracy verification, no unauthorized practice |
| **Education** | FERPA, accessibility laws | Student privacy, equal access, no bias |
| **Employment** | EEOC, local employment law | Non-discrimination, adverse impact testing |

---

## Implementing Transparency and Explainability

### Levels of Explainability

```
EXPLAINABILITY SPECTRUM
─────────────────────────────────────────────────────────────────

Level 0: BLACK BOX (No explanation)
┌──────────────────────────────────────────────────────────────┐
│  Input → [????] → Output                                      │
│                                                               │
│  "Your loan was denied."                                      │
│                                                               │
│  Acceptable for: Low-stakes, non-regulated use cases         │
└──────────────────────────────────────────────────────────────┘

Level 1: OUTCOME EXPLANATION (What happened)
┌──────────────────────────────────────────────────────────────┐
│  Input → [????] → Output + Summary                           │
│                                                               │
│  "Your loan was denied because your credit score             │
│   doesn't meet our minimum requirements."                    │
│                                                               │
│  Acceptable for: Most consumer applications                  │
└──────────────────────────────────────────────────────────────┘

Level 2: FACTOR EXPLANATION (Why it happened)
┌──────────────────────────────────────────────────────────────┐
│  Input → [Factors] → Output + Detailed Reasoning             │
│                                                               │
│  "Your loan was denied. Key factors:                         │
│   - Credit score: 620 (minimum 650)                          │
│   - Debt-to-income: 45% (maximum 40%)                        │
│   - Employment: <2 years (minimum 2 years)"                  │
│                                                               │
│  Acceptable for: Regulated decisions, GDPR compliance        │
└──────────────────────────────────────────────────────────────┘

Level 3: PROCESS EXPLANATION (How it was decided)
┌──────────────────────────────────────────────────────────────┐
│  Input → [Full Process] → Output + Complete Trace            │
│                                                               │
│  "Decision Process:                                           │
│   1. Retrieved your credit report (Experian, 2024-01-15)    │
│   2. Calculated debt-to-income from reported debts          │
│   3. Verified employment with employer                       │
│   4. Applied scoring model v2.3                              │
│   5. Score: 45/100 (threshold: 60)                          │
│   6. Decision: Denied                                        │
│   7. Reviewed by system (no override triggered)"            │
│                                                               │
│  Required for: High-risk AI, legal proceedings               │
└──────────────────────────────────────────────────────────────┘
```

### Implementing Explanations

```python
# code/01_explainability.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ExplanationLevel(Enum):
    OUTCOME = 1
    FACTORS = 2
    PROCESS = 3

@dataclass
class DecisionStep:
    """A single step in the decision process."""
    step_number: int
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0

@dataclass
class ExplainableDecision:
    """A decision with full explainability."""
    decision_id: str
    outcome: str
    confidence: float
    factors: List[Dict[str, Any]]
    process_steps: List[DecisionStep]
    timestamp: datetime = field(default_factory=datetime.now)

    def explain(self, level: ExplanationLevel) -> str:
        """Generate explanation at requested level."""
        if level == ExplanationLevel.OUTCOME:
            return self._outcome_explanation()
        elif level == ExplanationLevel.FACTORS:
            return self._factor_explanation()
        else:
            return self._process_explanation()

    def _outcome_explanation(self) -> str:
        return f"Decision: {self.outcome} (confidence: {self.confidence:.0%})"

    def _factor_explanation(self) -> str:
        lines = [self._outcome_explanation(), "", "Key factors:"]
        for factor in self.factors:
            impact = factor.get('impact', 'neutral')
            icon = "+" if impact == 'positive' else "-" if impact == 'negative' else "•"
            lines.append(f"  {icon} {factor['name']}: {factor['value']} ({factor.get('explanation', '')})")
        return "\n".join(lines)

    def _process_explanation(self) -> str:
        lines = [self._factor_explanation(), "", "Decision process:"]
        for step in self.process_steps:
            lines.append(f"  {step.step_number}. {step.action}")
            lines.append(f"     Reasoning: {step.reasoning}")
            lines.append(f"     Confidence: {step.confidence:.0%}")
        return "\n".join(lines)


class ExplainableAgent:
    """Agent that produces explainable decisions."""

    def __init__(self, llm):
        self.llm = llm
        self.decision_log: List[ExplainableDecision] = []

    def decide(self, query: str, context: Dict[str, Any]) -> ExplainableDecision:
        """Make an explainable decision."""
        steps = []
        factors = []

        # Step 1: Understand the query
        step1 = DecisionStep(
            step_number=1,
            action="Parse and classify query",
            input_data={"query": query},
            output_data={"intent": self._classify_intent(query)},
            reasoning="Identified query intent to determine appropriate response type"
        )
        steps.append(step1)

        # Step 2: Gather relevant information
        step2 = DecisionStep(
            step_number=2,
            action="Retrieve relevant context",
            input_data={"context_keys": list(context.keys())},
            output_data={"retrieved": self._gather_context(context)},
            reasoning="Collected all relevant information for decision"
        )
        steps.append(step2)

        # Step 3: Evaluate factors
        evaluated_factors = self._evaluate_factors(query, context)
        factors.extend(evaluated_factors)
        step3 = DecisionStep(
            step_number=3,
            action="Evaluate decision factors",
            input_data={"num_factors": len(evaluated_factors)},
            output_data={"factors": [f["name"] for f in evaluated_factors]},
            reasoning="Analyzed each factor's contribution to the decision"
        )
        steps.append(step3)

        # Step 4: Make decision
        outcome, confidence = self._make_decision(query, context, factors)
        step4 = DecisionStep(
            step_number=4,
            action="Generate decision",
            input_data={"positive_factors": sum(1 for f in factors if f.get("impact") == "positive")},
            output_data={"outcome": outcome, "confidence": confidence},
            reasoning=f"Combined factors to reach decision with {confidence:.0%} confidence"
        )
        steps.append(step4)

        decision = ExplainableDecision(
            decision_id=self._generate_id(),
            outcome=outcome,
            confidence=confidence,
            factors=factors,
            process_steps=steps
        )

        self.decision_log.append(decision)
        return decision

    def _classify_intent(self, query: str) -> str:
        # Simplified intent classification
        return "information_request"

    def _gather_context(self, context: Dict) -> Dict:
        return {"items_retrieved": len(context)}

    def _evaluate_factors(self, query: str, context: Dict) -> List[Dict]:
        # Simplified factor evaluation
        return [
            {"name": "Query Clarity", "value": "High", "impact": "positive",
             "explanation": "Query is clear and specific"},
            {"name": "Context Available", "value": f"{len(context)} items", "impact": "positive",
             "explanation": "Sufficient context provided"}
        ]

    def _make_decision(self, query: str, context: Dict, factors: List) -> tuple:
        positive = sum(1 for f in factors if f.get("impact") == "positive")
        confidence = min(0.95, 0.5 + (positive * 0.15))
        return "Approved", confidence

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())[:8]
```

---

## Human-in-the-Loop Patterns

### When to Involve Humans

```
HUMAN INVOLVEMENT DECISION TREE
─────────────────────────────────────────────────────────────────

                    Start
                      │
                      ▼
            ┌─────────────────┐
            │ Is this a high- │
            │ stakes decision?│
            └────────┬────────┘
                     │
            ┌────────┴────────┐
            │                 │
           YES               NO
            │                 │
            ▼                 ▼
    ┌───────────────┐  ┌───────────────┐
    │ REQUIRE       │  │ Is confidence │
    │ Human Review  │  │ below 80%?    │
    └───────────────┘  └───────┬───────┘
                               │
                      ┌────────┴────────┐
                      │                 │
                     YES               NO
                      │                 │
                      ▼                 ▼
              ┌───────────────┐  ┌───────────────┐
              │ REQUEST       │  │ Is this a new │
              │ Human Review  │  │ edge case?    │
              └───────────────┘  └───────┬───────┘
                                         │
                                ┌────────┴────────┐
                                │                 │
                               YES               NO
                                │                 │
                                ▼                 ▼
                        ┌───────────────┐  ┌───────────────┐
                        │ FLAG for      │  │ PROCEED       │
                        │ Human Review  │  │ Autonomously  │
                        └───────────────┘  └───────────────┘


HIGH-STAKES DECISIONS include:
• Financial decisions > $1000
• Healthcare recommendations
• Legal advice
• Employment decisions
• Content affecting minors
• Safety-critical operations
```

### Human-in-the-Loop Implementation

```python
# code/01_human_in_loop.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Any
import asyncio

class ReviewPriority(Enum):
    CRITICAL = 1  # Requires immediate review
    HIGH = 2      # Review within 1 hour
    MEDIUM = 3    # Review within 24 hours
    LOW = 4       # Review when available

class ReviewDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    ESCALATE = "escalate"

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
    timeout_hours: int = 24
    fallback_action: str = "reject"

@dataclass
class HumanReviewResult:
    """Result of human review."""
    request_id: str
    reviewer_id: str
    decision: ReviewDecision
    modified_output: Optional[str] = None
    feedback: Optional[str] = None
    review_time_seconds: float = 0

class HumanInTheLoopController:
    """Manages human oversight of AI decisions."""

    def __init__(self):
        self.pending_reviews: dict[str, HumanReviewRequest] = {}
        self.completed_reviews: dict[str, HumanReviewResult] = {}
        self.review_handlers: dict[str, Callable] = {}

        # Thresholds for automatic routing
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

        # Check for edge cases
        if context.get("is_edge_case"):
            return True, ReviewPriority.LOW

        return False, ReviewPriority.LOW

    def request_review(self, request: HumanReviewRequest) -> str:
        """Submit a decision for human review."""
        self.pending_reviews[request.request_id] = request
        self._notify_reviewers(request)
        return request.request_id

    def submit_review(self, result: HumanReviewResult) -> bool:
        """Submit a human review decision."""
        if result.request_id not in self.pending_reviews:
            return False

        del self.pending_reviews[result.request_id]
        self.completed_reviews[result.request_id] = result
        self._process_feedback(result)
        return True

    def _notify_reviewers(self, request: HumanReviewRequest):
        """Notify appropriate reviewers based on priority."""
        # In production, this would send notifications
        print(f"[REVIEW NEEDED] {request.priority.name}: {request.decision_type}")
        print(f"  AI recommends: {request.ai_recommendation}")
        print(f"  Confidence: {request.confidence:.0%}")

    def _process_feedback(self, result: HumanReviewResult):
        """Use review feedback to improve the system."""
        # Log for model improvement
        if result.decision == ReviewDecision.REJECT:
            print(f"[LEARNING] AI decision rejected: {result.feedback}")
        elif result.decision == ReviewDecision.MODIFY:
            print(f"[LEARNING] AI decision modified: {result.modified_output}")


class HumanOversightAgent:
    """Agent with built-in human oversight capabilities."""

    def __init__(self, llm, hitl_controller: HumanInTheLoopController):
        self.llm = llm
        self.hitl = hitl_controller

    async def process_with_oversight(
        self,
        query: str,
        decision_type: str,
        context: dict
    ) -> dict:
        """Process a query with appropriate human oversight."""

        # Generate AI decision
        ai_response = await self._generate_response(query, context)
        confidence = ai_response["confidence"]

        # Check if human review is needed
        needs_review, priority = self.hitl.should_require_review(
            decision_type, confidence, context
        )

        if needs_review:
            # Create review request
            review_request = HumanReviewRequest(
                request_id=self._generate_id(),
                decision_type=decision_type,
                ai_recommendation=ai_response["response"],
                confidence=confidence,
                context=context,
                priority=priority,
                reasoning=ai_response["reasoning"]
            )

            request_id = self.hitl.request_review(review_request)

            return {
                "status": "pending_review",
                "request_id": request_id,
                "ai_recommendation": ai_response["response"],
                "confidence": confidence,
                "message": f"This decision requires human review ({priority.name} priority)"
            }

        return {
            "status": "approved",
            "response": ai_response["response"],
            "confidence": confidence,
            "human_reviewed": False
        }

    async def _generate_response(self, query: str, context: dict) -> dict:
        # Simplified response generation
        return {
            "response": f"AI response to: {query}",
            "confidence": 0.85,
            "reasoning": "Based on available context"
        }

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())[:8]
```

---

## Building an Ethics Framework

### The Ethics Checklist

```python
# code/01_ethics_checklist.py

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class EthicsCheckItem:
    """A single item in the ethics checklist."""
    category: str
    question: str
    guidance: str
    required: bool = True

@dataclass
class EthicsAssessment:
    """Complete ethics assessment for an AI feature."""
    feature_name: str
    assessor: str
    risk_level: RiskLevel
    checks: dict[str, bool]
    mitigations: List[str]
    approval_status: str
    notes: str

class AIEthicsChecklist:
    """Ethics checklist for AI feature deployment."""

    CHECKLIST = [
        # Safety
        EthicsCheckItem(
            category="Safety",
            question="Could this feature cause physical harm?",
            guidance="Consider all ways the AI output could lead to physical harm"
        ),
        EthicsCheckItem(
            category="Safety",
            question="Could this feature cause psychological harm?",
            guidance="Consider harassment, manipulation, distress"
        ),
        EthicsCheckItem(
            category="Safety",
            question="Are there adequate safeguards against misuse?",
            guidance="Consider prompt injection, adversarial inputs, abuse"
        ),

        # Fairness
        EthicsCheckItem(
            category="Fairness",
            question="Has bias testing been performed?",
            guidance="Test across protected classes: race, gender, age, etc."
        ),
        EthicsCheckItem(
            category="Fairness",
            question="Does the feature treat all users equitably?",
            guidance="Consider accessibility, language, socioeconomic factors"
        ),
        EthicsCheckItem(
            category="Fairness",
            question="Are there feedback mechanisms for bias reports?",
            guidance="Users should be able to report unfair treatment"
        ),

        # Privacy
        EthicsCheckItem(
            category="Privacy",
            question="What personal data does this feature access?",
            guidance="Minimize data collection, document necessity"
        ),
        EthicsCheckItem(
            category="Privacy",
            question="Is user consent obtained appropriately?",
            guidance="Clear disclosure, opt-in where required"
        ),
        EthicsCheckItem(
            category="Privacy",
            question="How long is data retained?",
            guidance="Minimize retention, document deletion procedures"
        ),

        # Transparency
        EthicsCheckItem(
            category="Transparency",
            question="Is it clear to users they're interacting with AI?",
            guidance="Disclosure requirements, no deceptive practices"
        ),
        EthicsCheckItem(
            category="Transparency",
            question="Can users understand how decisions are made?",
            guidance="Explanations appropriate to context and risk level"
        ),
        EthicsCheckItem(
            category="Transparency",
            question="Is there documentation for auditors?",
            guidance="Complete audit trail, model cards, data sheets"
        ),

        # Accountability
        EthicsCheckItem(
            category="Accountability",
            question="Who is responsible for this feature's outcomes?",
            guidance="Clear ownership, escalation paths"
        ),
        EthicsCheckItem(
            category="Accountability",
            question="What happens when something goes wrong?",
            guidance="Incident response plan, user remediation"
        ),
        EthicsCheckItem(
            category="Accountability",
            question="How are improvements tracked and implemented?",
            guidance="Feedback loops, continuous monitoring"
        ),

        # Human Oversight
        EthicsCheckItem(
            category="Human Oversight",
            question="Are high-stakes decisions reviewed by humans?",
            guidance="Define thresholds, ensure capacity"
        ),
        EthicsCheckItem(
            category="Human Oversight",
            question="Can users contest AI decisions?",
            guidance="Appeal process, human review option"
        ),
        EthicsCheckItem(
            category="Human Oversight",
            question="Is there a kill switch for emergencies?",
            guidance="Ability to disable feature immediately"
        ),
    ]

    def assess(
        self,
        feature_name: str,
        assessor: str,
        responses: dict[str, bool],
        mitigations: List[str],
        notes: str = ""
    ) -> EthicsAssessment:
        """Perform ethics assessment."""

        # Check all required items are addressed
        failed_required = []
        for item in self.CHECKLIST:
            key = f"{item.category}:{item.question[:30]}"
            if item.required and not responses.get(key, False):
                failed_required.append(item)

        # Determine risk level
        if len(failed_required) > 5:
            risk_level = RiskLevel.CRITICAL
            approval = "BLOCKED"
        elif len(failed_required) > 2:
            risk_level = RiskLevel.HIGH
            approval = "REQUIRES_ESCALATION"
        elif len(failed_required) > 0:
            risk_level = RiskLevel.MEDIUM
            approval = "CONDITIONAL"
        else:
            risk_level = RiskLevel.LOW
            approval = "APPROVED"

        return EthicsAssessment(
            feature_name=feature_name,
            assessor=assessor,
            risk_level=risk_level,
            checks=responses,
            mitigations=mitigations,
            approval_status=approval,
            notes=notes
        )

    def generate_report(self, assessment: EthicsAssessment) -> str:
        """Generate ethics assessment report."""
        lines = [
            f"# Ethics Assessment: {assessment.feature_name}",
            f"",
            f"**Assessor:** {assessment.assessor}",
            f"**Risk Level:** {assessment.risk_level.name}",
            f"**Status:** {assessment.approval_status}",
            f"",
            f"## Checklist Results",
            ""
        ]

        # Group by category
        by_category = {}
        for item in self.CHECKLIST:
            if item.category not in by_category:
                by_category[item.category] = []
            key = f"{item.category}:{item.question[:30]}"
            passed = assessment.checks.get(key, False)
            by_category[item.category].append((item, passed))

        for category, items in by_category.items():
            lines.append(f"### {category}")
            for item, passed in items:
                icon = "[x]" if passed else "[ ]"
                lines.append(f"- {icon} {item.question}")
            lines.append("")

        if assessment.mitigations:
            lines.append("## Mitigations")
            for m in assessment.mitigations:
                lines.append(f"- {m}")
            lines.append("")

        if assessment.notes:
            lines.append("## Notes")
            lines.append(assessment.notes)

        return "\n".join(lines)
```

---

## Practical Exercise: Building a Responsible Agent

Let's build an agent with all responsible AI practices built in:

```python
# code/01_responsible_agent.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json

class ContentCategory(Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    HARMFUL = "harmful"
    ILLEGAL = "illegal"

@dataclass
class AuditLogEntry:
    """Entry in the audit log."""
    timestamp: datetime
    action: str
    input_summary: str
    output_summary: str
    decision_factors: List[str]
    human_involved: bool
    user_id: Optional[str] = None

class ResponsibleAgent:
    """
    An agent built with responsible AI practices.

    Features:
    - Content classification and filtering
    - Explainable decisions
    - Human-in-the-loop for sensitive cases
    - Complete audit logging
    - Privacy protection
    - Bias mitigation
    """

    def __init__(self, llm):
        self.llm = llm
        self.audit_log: List[AuditLogEntry] = []

        # Safety thresholds
        self.confidence_threshold = 0.8
        self.sensitivity_keywords = {
            "medical", "health", "diagnosis", "legal", "law",
            "financial", "money", "investment", "suicide", "harm"
        }

        # Blocked content patterns
        self.blocked_patterns = [
            "how to make weapons",
            "how to hack",
            "personal information about"
        ]

    def process(
        self,
        query: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query with full responsible AI practices."""

        decision_factors = []

        # Step 1: Input validation and classification
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

        # Step 2: Check for sensitivity
        is_sensitive = self._check_sensitivity(query)
        if is_sensitive:
            decision_factors.append("Query contains sensitive topics")

        # Step 3: Generate response
        response = self._generate_response(query, context)
        decision_factors.append(f"Response confidence: {response['confidence']:.0%}")

        # Step 4: Check if human review needed
        needs_human = (
            is_sensitive or
            response['confidence'] < self.confidence_threshold or
            content_category == ContentCategory.SENSITIVE
        )

        if needs_human:
            decision_factors.append("Flagged for human review")
            return self._human_review_response(
                query, response, user_id, decision_factors
            )

        # Step 5: Privacy check on output
        sanitized_response = self._sanitize_output(response['content'])
        if sanitized_response != response['content']:
            decision_factors.append("Output sanitized for privacy")

        # Step 6: Generate explanation
        explanation = self._generate_explanation(
            query, sanitized_response, decision_factors
        )

        # Step 7: Log for audit
        self._log_action(
            action="response_generated",
            input_summary=query[:100],
            output_summary=sanitized_response[:100],
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
            "audit_id": self.audit_log[-1].timestamp.isoformat()
        }

    def _classify_content(self, query: str) -> ContentCategory:
        """Classify content for safety."""
        query_lower = query.lower()

        for pattern in self.blocked_patterns:
            if pattern in query_lower:
                return ContentCategory.ILLEGAL

        # Check for harmful intent
        harmful_signals = ["hurt", "kill", "attack", "destroy"]
        if any(signal in query_lower for signal in harmful_signals):
            return ContentCategory.HARMFUL

        # Check for sensitive topics
        if any(kw in query_lower for kw in self.sensitivity_keywords):
            return ContentCategory.SENSITIVE

        return ContentCategory.SAFE

    def _check_sensitivity(self, query: str) -> bool:
        """Check if query involves sensitive topics."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.sensitivity_keywords)

    def _generate_response(self, query: str, context: Dict) -> Dict:
        """Generate response with confidence score."""
        # Simplified - in production, use actual LLM
        return {
            "content": f"Response to: {query}",
            "confidence": 0.85
        }

    def _sanitize_output(self, content: str) -> str:
        """Remove any PII from output."""
        # Simplified - in production, use Presidio or similar
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
2. Relevant context was retrieved
3. A response was generated with confidence scoring
4. The response was checked for privacy concerns
5. This explanation was generated

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

        self._log_action(
            action="request_blocked",
            input_summary=query[:100],
            output_summary=message,
            decision_factors=factors,
            human_involved=False,
            user_id=user_id
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
        factors: List[str]
    ) -> Dict:
        """Return response pending human review."""
        self._log_action(
            action="flagged_for_review",
            input_summary=query[:100],
            output_summary="Pending human review",
            decision_factors=factors,
            human_involved=True,
            user_id=user_id
        )

        return {
            "status": "pending_review",
            "message": "Your request is being reviewed by a human expert.",
            "estimated_time": "24-48 hours",
            "ai_suggestion": ai_response['content'],
            "ai_confidence": ai_response['confidence'],
            "decision_factors": factors
        }

    def _log_action(
        self,
        action: str,
        input_summary: str,
        output_summary: str,
        decision_factors: List[str],
        human_involved: bool,
        user_id: Optional[str] = None
    ):
        """Log action for audit purposes."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            decision_factors=decision_factors,
            human_involved=human_involved,
            user_id=user_id
        )
        self.audit_log.append(entry)

    def export_audit_log(self) -> str:
        """Export audit log for compliance."""
        entries = []
        for entry in self.audit_log:
            entries.append({
                "timestamp": entry.timestamp.isoformat(),
                "action": entry.action,
                "input_summary": entry.input_summary,
                "output_summary": entry.output_summary,
                "decision_factors": entry.decision_factors,
                "human_involved": entry.human_involved,
                "user_id": entry.user_id
            })
        return json.dumps(entries, indent=2)


# === Demo ===

if __name__ == "__main__":
    print("=" * 60)
    print("RESPONSIBLE AI AGENT DEMO")
    print("=" * 60)

    agent = ResponsibleAgent(llm=None)  # Using mock LLM

    # Test cases
    test_queries = [
        ("What's the weather like?", {}),  # Safe query
        ("Give me medical advice", {}),  # Sensitive query
        ("How to hack a computer", {}),  # Blocked query
        ("Help me with my finances", {"amount": 50000}),  # Sensitive + high value
    ]

    for query, context in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print(f"Context: {context}")
        print()

        result = agent.process(query, context)

        print(f"Status: {result['status']}")
        if 'response' in result:
            print(f"Response: {result['response']}")
        if 'message' in result:
            print(f"Message: {result['message']}")
        print(f"Factors: {result.get('decision_factors', [])}")

    print(f"\n{'=' * 60}")
    print("AUDIT LOG")
    print("=" * 60)
    print(agent.export_audit_log())
```

---

## Key Takeaways

### 1. Responsible AI is a Business Necessity
Not just ethics—it's regulatory compliance, reputation protection, and competitive advantage.

### 2. Use Medical Ethics as a Model
The principles of "first do no harm," informed consent, beneficence, and confidentiality translate directly to AI.

### 3. Layer Simple Rules for Complex Ethics
Don't try to encode complex ethics—use simple rules at multiple levels (hard constraints, soft constraints, guidelines).

### 4. Transparency Builds Trust
Explainable AI isn't optional for high-stakes decisions—it's legally required in many jurisdictions.

### 5. Humans Must Remain in Control
Human-in-the-loop isn't weakness—it's appropriate humility about AI limitations.

### 6. Document Everything
Complete audit trails are essential for compliance, debugging, and continuous improvement.

---

## What's Next?

In **Module 6.2: Guardrails Implementation**, we'll dive into the technical details of:
- Input validation and injection detection
- Output filtering and content moderation
- Hallucination detection and prevention
- Building defense-in-depth systems

We've established the "why" of responsible AI—now let's build the "how."

[Continue to Module 6.2 →](02_guardrails_implementation.md)
