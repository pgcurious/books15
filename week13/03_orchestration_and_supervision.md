# Module 13.3: Agent Orchestration & Supervision

> "Coming together is a beginning, staying together is progress, and working together is success." — Henry Ford

## Introduction

Single agents have limits. Complex tasks require multiple specialized agents working together. But how do you coordinate autonomous systems? How do you maintain oversight without micromanaging? In this module, we'll use analogical thinking to design multi-agent systems that work like high-performing organizations—with clear roles, effective coordination, and appropriate supervision.

---

## Learning Objectives

By the end of this module, you will:
- Design multi-agent architectures using organizational patterns
- Implement supervisor-worker relationships with appropriate autonomy
- Build human-in-the-loop systems that maintain meaningful oversight
- Create guardrails that enable rather than restrict agent capability
- Develop trust through transparency and accountability

---

## Analogical Thinking: Agents as Organizations

### The Core Analogy

Multi-agent systems face the same challenges as human organizations:

| Organizational Challenge | Multi-Agent Equivalent |
|-------------------------|----------------------|
| Hiring specialists | Deploying specialized agents |
| Management hierarchy | Supervisor agents |
| Task delegation | Work distribution |
| Communication protocols | Message passing |
| Performance reviews | Agent evaluation |
| Escalation paths | Human-in-the-loop triggers |
| Company policies | Guardrails and constraints |
| Training programs | Agent improvement |
| Accountability | Audit trails |

**Key Insight:** Centuries of organizational design wisdom apply directly to multi-agent systems.

### From Analogy to Architecture

```
Traditional Organization          Multi-Agent System
====================             ==================

┌─────────────┐                  ┌─────────────────┐
│     CEO     │                  │   Orchestrator  │
│  (Strategy) │                  │     Agent       │
└──────┬──────┘                  └────────┬────────┘
       │                                  │
   ┌───┴────┐                        ┌────┴─────┐
   │        │                        │          │
┌──▼──┐  ┌──▼──┐                ┌────▼───┐ ┌────▼───┐
│ VP  │  │ VP  │                │Supervi-│ │Supervi-│
│Sales│  │Eng  │                │sor A   │ │sor B   │
└──┬──┘  └──┬──┘                └────┬───┘ └────┬───┘
   │        │                        │          │
┌──▼──┐  ┌──▼──┐                ┌────▼───┐ ┌────▼───┐
│Team │  │Team │                │Workers │ │Workers │
└─────┘  └─────┘                └────────┘ └────────┘
```

---

## Multi-Agent Coordination Patterns

### Pattern 1: Hub and Spoke (Centralized Coordination)

Like a call center with a dispatcher:

```python
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"

@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    receiver: str
    content: str
    message_type: str  # "task", "result", "question", "escalation"
    priority: int = 5
    requires_response: bool = False
    metadata: Dict = field(default_factory=dict)

class HubAndSpokeOrchestrator:
    """
    Central orchestrator that coordinates specialized agents.

    Analogy: Like a project manager who delegates to specialists
    but maintains visibility into all activities.
    """

    def __init__(self, llm):
        self.llm = llm
        self.agents: Dict[str, 'SpecializedAgent'] = {}
        self.message_queue: List[AgentMessage] = []
        self.task_registry: Dict[str, Dict] = {}
        self.human_escalation_handler = None

    def register_agent(self, agent: 'SpecializedAgent'):
        """Add a specialized agent to the system."""
        self.agents[agent.name] = agent
        agent.orchestrator = self

    async def execute_complex_task(self, task: str) -> Dict:
        """
        Break down and coordinate a complex task.

        The orchestrator:
        1. Decomposes the task
        2. Assigns sub-tasks to appropriate agents
        3. Monitors progress
        4. Handles coordination and conflicts
        5. Assembles final result
        """

        # 1. Decompose task
        subtasks = await self.decompose_task(task)

        # 2. Assign to agents
        assignments = await self.assign_subtasks(subtasks)

        # 3. Execute with coordination
        results = {}
        for subtask_id, assignment in assignments.items():
            agent = self.agents[assignment["agent"]]

            # Execute subtask
            result = await agent.execute(
                task=assignment["task"],
                context=self.get_context_for_agent(agent, results)
            )

            results[subtask_id] = result

            # Check for escalation needs
            if result.get("needs_escalation"):
                await self.handle_escalation(subtask_id, result)

            # Check for inter-agent communication needs
            if result.get("needs_input_from"):
                await self.coordinate_agents(
                    subtask_id,
                    result["needs_input_from"]
                )

        # 4. Assemble final result
        return await self.assemble_results(task, results)

    async def decompose_task(self, task: str) -> List[Dict]:
        """Break task into subtasks using LLM."""
        prompt = f"""
        Complex Task: {task}

        Available Specialist Agents:
        {self._format_available_agents()}

        Decompose this task into subtasks, where each subtask can be
        handled by one specialist agent. Consider dependencies between subtasks.

        For each subtask, specify:
        - id: unique identifier
        - description: what needs to be done
        - best_agent_type: which specialist should handle it
        - dependencies: list of subtask ids this depends on
        - estimated_complexity: simple/moderate/complex
        """

        response = await self.llm.ainvoke(prompt)
        return self._parse_subtasks(response.content)

    async def assign_subtasks(self, subtasks: List[Dict]) -> Dict[str, Dict]:
        """Assign subtasks to specific agents."""
        assignments = {}

        for subtask in subtasks:
            # Find best available agent of the required type
            agent_name = self._select_agent(subtask["best_agent_type"])

            assignments[subtask["id"]] = {
                "task": subtask["description"],
                "agent": agent_name,
                "dependencies": subtask.get("dependencies", []),
                "complexity": subtask.get("estimated_complexity", "moderate")
            }

        return assignments

    def _select_agent(self, agent_type: str) -> str:
        """Select the best agent for a task type."""
        # Find agents of the right type
        candidates = [
            name for name, agent in self.agents.items()
            if agent.specialization == agent_type
        ]

        if not candidates:
            # Fall back to general-purpose agent
            return list(self.agents.keys())[0]

        # Select based on current load and past performance
        return min(candidates, key=lambda n: self.agents[n].current_load)

    def _format_available_agents(self) -> str:
        return "\n".join([
            f"- {name}: {agent.description} (specialization: {agent.specialization})"
            for name, agent in self.agents.items()
        ])


class SpecializedAgent:
    """
    A specialized agent that works under orchestrator coordination.

    Analogy: Like a specialist employee who has autonomy within
    their domain but reports to management.
    """

    def __init__(
        self,
        name: str,
        llm,
        specialization: str,
        description: str,
        tools: List = None
    ):
        self.name = name
        self.llm = llm
        self.specialization = specialization
        self.description = description
        self.tools = tools or []
        self.orchestrator: Optional[HubAndSpokeOrchestrator] = None
        self.current_load = 0

    async def execute(self, task: str, context: Dict) -> Dict:
        """Execute a task within the agent's specialization."""
        self.current_load += 1

        try:
            # Check if task is within capabilities
            if not await self.can_handle(task):
                return {
                    "status": "declined",
                    "reason": "Outside specialization",
                    "needs_escalation": True
                }

            # Execute the task
            result = await self._perform_task(task, context)

            # Self-evaluate
            evaluation = await self._evaluate_result(task, result)

            if evaluation["confidence"] < 0.6:
                result["needs_review"] = True
                result["confidence"] = evaluation["confidence"]

            return result

        finally:
            self.current_load -= 1

    async def can_handle(self, task: str) -> bool:
        """Determine if this task is within the agent's capabilities."""
        prompt = f"""
        My specialization: {self.specialization}
        My capabilities: {self.description}

        Task: {task}

        Can I handle this task effectively? Answer YES or NO with brief reasoning.
        """
        response = await self.llm.ainvoke(prompt)
        return "YES" in response.content.upper()

    async def request_collaboration(self, needed_specialization: str, question: str):
        """Request help from another agent through the orchestrator."""
        if self.orchestrator:
            message = AgentMessage(
                sender=self.name,
                receiver="orchestrator",
                content=question,
                message_type="collaboration_request",
                metadata={"needed_specialization": needed_specialization}
            )
            return await self.orchestrator.route_message(message)
```

### Pattern 2: Peer-to-Peer (Decentralized Coordination)

Like a team of equals collaborating:

```python
class PeerAgent:
    """
    Agent that coordinates directly with peers.

    Analogy: Like a cross-functional team where members
    communicate directly rather than through a manager.
    """

    def __init__(self, name: str, llm, specialization: str):
        self.name = name
        self.llm = llm
        self.specialization = specialization
        self.peers: Dict[str, 'PeerAgent'] = {}
        self.shared_context: SharedContext = None

    def register_peer(self, peer: 'PeerAgent'):
        """Establish bidirectional peer relationship."""
        self.peers[peer.name] = peer
        peer.peers[self.name] = self

    async def execute_collaborative_task(self, task: str) -> Dict:
        """Execute task, collaborating with peers as needed."""

        # 1. Assess what parts I can do
        my_contribution = await self.assess_contribution(task)

        # 2. Identify what I need from peers
        needed_help = await self.identify_needed_help(task, my_contribution)

        # 3. Request help from appropriate peers
        peer_contributions = {}
        for need in needed_help:
            peer = self._find_best_peer(need["specialization"])
            if peer:
                contribution = await peer.contribute(
                    request=need["request"],
                    context=self.shared_context.get_relevant(task)
                )
                peer_contributions[peer.name] = contribution

        # 4. Integrate contributions
        result = await self.integrate_contributions(
            task=task,
            my_work=my_contribution,
            peer_work=peer_contributions
        )

        # 5. Update shared context
        self.shared_context.add(task, result)

        return result

    async def contribute(self, request: str, context: Dict) -> Dict:
        """Contribute to another agent's task."""
        prompt = f"""
        A peer agent is requesting help with their task.

        Their request: {request}
        Shared context: {context}
        My specialization: {self.specialization}

        Provide a helpful contribution based on my expertise.
        """
        response = await self.llm.ainvoke(prompt)
        return {"contribution": response.content}

    def _find_best_peer(self, needed_specialization: str) -> Optional['PeerAgent']:
        """Find the peer best suited for a specialization."""
        for peer in self.peers.values():
            if peer.specialization == needed_specialization:
                return peer
        return None


class SharedContext:
    """
    Shared knowledge space for peer agents.

    Analogy: Like a shared document or wiki that team members
    can all read from and contribute to.
    """

    def __init__(self):
        self.items: List[Dict] = []
        self.lock = asyncio.Lock()

    async def add(self, task: str, result: Dict):
        """Add to shared context (thread-safe)."""
        async with self.lock:
            self.items.append({
                "task": task,
                "result": result,
                "timestamp": datetime.now()
            })

    def get_relevant(self, query: str, k: int = 5) -> List[Dict]:
        """Get relevant shared context items."""
        # Simple recency-based - production would use embeddings
        return self.items[-k:]
```

### Pattern 3: Hierarchical (Layered Management)

Like a corporate hierarchy:

```python
class HierarchicalAgentSystem:
    """
    Multi-level agent hierarchy with delegation.

    Analogy: Corporate structure with executives, managers, and workers.
    Each level has appropriate autonomy and escalation paths.
    """

    def __init__(self, llm):
        self.llm = llm
        self.levels: Dict[int, List['HierarchicalAgent']] = {
            0: [],  # Executives (highest level)
            1: [],  # Managers
            2: [],  # Workers (lowest level)
        }

    async def execute_from_top(self, task: str) -> Dict:
        """Execute a task starting from executive level."""
        # Executive agent receives the task
        executive = self.levels[0][0]
        return await executive.handle(task)


class HierarchicalAgent:
    """Agent that operates within a hierarchy."""

    def __init__(
        self,
        name: str,
        llm,
        level: int,
        specialization: str,
        authority_scope: List[str]
    ):
        self.name = name
        self.llm = llm
        self.level = level  # 0 = highest authority
        self.specialization = specialization
        self.authority_scope = authority_scope  # What this agent can decide
        self.subordinates: List['HierarchicalAgent'] = []
        self.superior: Optional['HierarchicalAgent'] = None

    async def handle(self, task: str) -> Dict:
        """Handle a task, delegating as appropriate."""

        # 1. Assess task complexity and scope
        assessment = await self.assess_task(task)

        # 2. Decide: handle, delegate, or escalate
        if assessment["within_my_scope"] and assessment["complexity"] == "simple":
            # Handle directly
            return await self.execute_directly(task)

        elif self.subordinates and assessment["should_delegate"]:
            # Delegate to subordinates
            return await self.delegate(task, assessment)

        elif self.superior and assessment["needs_escalation"]:
            # Escalate to superior
            return await self.escalate(task, assessment["escalation_reason"])

        else:
            # Handle with effort
            return await self.execute_directly(task)

    async def delegate(self, task: str, assessment: Dict) -> Dict:
        """Delegate task to subordinates."""

        # Decompose for delegation
        subtasks = await self.decompose_for_subordinates(task)

        # Assign to appropriate subordinates
        results = {}
        for subtask in subtasks:
            subordinate = self._select_subordinate(subtask)
            result = await subordinate.handle(subtask["task"])

            # Review subordinate's work
            if self._requires_review(subtask, result):
                result = await self.review_and_improve(subtask, result)

            results[subtask["id"]] = result

        # Integrate results
        return await self.integrate_subordinate_results(task, results)

    async def escalate(self, task: str, reason: str) -> Dict:
        """Escalate to superior when beyond authority or capability."""
        if not self.superior:
            return {
                "status": "blocked",
                "reason": "No superior to escalate to",
                "task": task
            }

        # Package escalation with context
        escalation = {
            "original_task": task,
            "escalation_reason": reason,
            "my_assessment": await self.assess_task(task),
            "my_recommendation": await self.form_recommendation(task)
        }

        return await self.superior.handle_escalation(escalation)

    async def handle_escalation(self, escalation: Dict) -> Dict:
        """Handle escalation from subordinate."""

        # Review the escalation
        prompt = f"""
        A subordinate has escalated a task to me.

        Task: {escalation['original_task']}
        Reason for escalation: {escalation['escalation_reason']}
        Subordinate's assessment: {escalation['my_assessment']}
        Subordinate's recommendation: {escalation['my_recommendation']}

        My authority scope: {self.authority_scope}

        Should I:
        1. Approve the subordinate's recommendation
        2. Provide guidance and send back
        3. Handle directly
        4. Escalate further

        Decide and explain.
        """

        response = await self.llm.ainvoke(prompt)
        decision = self._parse_escalation_decision(response.content)

        # Act on decision
        if decision["action"] == "approve":
            return {"status": "approved", "recommendation": escalation["my_recommendation"]}
        elif decision["action"] == "send_back":
            return {"status": "guidance", "guidance": decision["guidance"]}
        elif decision["action"] == "handle":
            return await self.execute_directly(escalation["original_task"])
        else:
            return await self.escalate(escalation["original_task"], decision["reason"])
```

---

## Human-in-the-Loop Design

### First Principles: When Do Humans Need to Be Involved?

```
Human Involvement Decision Tree:

Is the action reversible?
├── No → Is it high-stakes? → Yes → REQUIRE human approval
│                           → No → NOTIFY human, proceed
└── Yes → Is it routine? → Yes → PROCEED, log for review
                        → No → ASK human preference
```

### The Escalation Framework

```python
from enum import Enum
from typing import Callable, Awaitable

class EscalationLevel(Enum):
    NONE = 0           # Fully autonomous
    NOTIFY = 1         # Inform human, proceed
    ADVISE = 2         # Recommend, human decides
    REQUIRE = 3        # Must have human approval
    TRANSFER = 4       # Hand off entirely to human


@dataclass
class EscalationTrigger:
    """Condition that triggers human involvement."""
    name: str
    condition: Callable[[Dict], bool]
    level: EscalationLevel
    message_template: str


class HumanInTheLoopManager:
    """
    Manages human involvement in agent operations.

    Analogy: Like an assistant who knows when to handle things
    independently vs. when to check with the boss.
    """

    def __init__(self, human_interface):
        self.human_interface = human_interface
        self.triggers: List[EscalationTrigger] = []
        self.escalation_history: List[Dict] = []
        self._initialize_default_triggers()

    def _initialize_default_triggers(self):
        """Set up standard escalation triggers."""

        # Financial threshold
        self.triggers.append(EscalationTrigger(
            name="financial_threshold",
            condition=lambda ctx: ctx.get("estimated_cost", 0) > 100,
            level=EscalationLevel.REQUIRE,
            message_template="Action may cost ${estimated_cost}. Approve?"
        ))

        # External communication
        self.triggers.append(EscalationTrigger(
            name="external_communication",
            condition=lambda ctx: ctx.get("sends_external_message", False),
            level=EscalationLevel.ADVISE,
            message_template="Agent wants to send external message: {message_preview}"
        ))

        # Data deletion
        self.triggers.append(EscalationTrigger(
            name="data_deletion",
            condition=lambda ctx: ctx.get("deletes_data", False),
            level=EscalationLevel.REQUIRE,
            message_template="Agent wants to delete: {deletion_target}"
        ))

        # Low confidence
        self.triggers.append(EscalationTrigger(
            name="low_confidence",
            condition=lambda ctx: ctx.get("confidence", 1.0) < 0.5,
            level=EscalationLevel.ADVISE,
            message_template="Agent is uncertain (confidence: {confidence}). Please advise."
        ))

        # Novel situation
        self.triggers.append(EscalationTrigger(
            name="novel_situation",
            condition=lambda ctx: ctx.get("is_novel", False),
            level=EscalationLevel.NOTIFY,
            message_template="Handling new situation type: {situation_type}"
        ))

    async def check_escalation(self, action: Dict, context: Dict) -> Dict:
        """Check if an action requires human involvement."""

        triggered = []
        highest_level = EscalationLevel.NONE

        for trigger in self.triggers:
            full_context = {**action, **context}
            if trigger.condition(full_context):
                triggered.append(trigger)
                if trigger.level.value > highest_level.value:
                    highest_level = trigger.level

        if highest_level == EscalationLevel.NONE:
            return {"proceed": True, "level": EscalationLevel.NONE}

        # Handle based on level
        return await self._handle_escalation(
            action=action,
            context=context,
            level=highest_level,
            triggers=triggered
        )

    async def _handle_escalation(
        self,
        action: Dict,
        context: Dict,
        level: EscalationLevel,
        triggers: List[EscalationTrigger]
    ) -> Dict:
        """Handle escalation at the appropriate level."""

        # Format message for human
        messages = [
            t.message_template.format(**{**action, **context})
            for t in triggers
        ]
        combined_message = "\n".join(messages)

        if level == EscalationLevel.NOTIFY:
            # Just notify, proceed anyway
            await self.human_interface.notify(combined_message)
            return {"proceed": True, "level": level, "human_notified": True}

        elif level == EscalationLevel.ADVISE:
            # Get advice but agent makes final call
            advice = await self.human_interface.get_advice(
                message=combined_message,
                options=["proceed", "modify", "cancel"],
                context=context
            )
            return {
                "proceed": advice["decision"] != "cancel",
                "level": level,
                "human_advice": advice,
                "modifications": advice.get("modifications")
            }

        elif level == EscalationLevel.REQUIRE:
            # Must have explicit approval
            approval = await self.human_interface.request_approval(
                message=combined_message,
                action_details=action,
                context=context
            )
            return {
                "proceed": approval["approved"],
                "level": level,
                "human_approval": approval
            }

        elif level == EscalationLevel.TRANSFER:
            # Hand off to human entirely
            result = await self.human_interface.transfer_task(
                message=combined_message,
                task=action,
                context=context
            )
            return {
                "proceed": False,
                "level": level,
                "human_handled": result
            }

    def add_custom_trigger(self, trigger: EscalationTrigger):
        """Add a custom escalation trigger."""
        self.triggers.append(trigger)


class HumanInterface:
    """Interface for human communication."""

    async def notify(self, message: str):
        """Send notification to human."""
        print(f"[NOTIFICATION] {message}")

    async def get_advice(self, message: str, options: List[str], context: Dict) -> Dict:
        """Get advice from human."""
        print(f"\n[ADVICE REQUESTED]\n{message}")
        print(f"Options: {options}")
        # In production, would await actual human input
        # For demo, simulate quick approval
        return {"decision": "proceed", "modifications": None}

    async def request_approval(self, message: str, action_details: Dict, context: Dict) -> Dict:
        """Request explicit approval."""
        print(f"\n[APPROVAL REQUIRED]\n{message}")
        print(f"Action: {action_details}")
        # In production, would await actual human approval
        return {"approved": True, "approver": "human", "timestamp": datetime.now()}

    async def transfer_task(self, message: str, task: Dict, context: Dict) -> Dict:
        """Transfer task entirely to human."""
        print(f"\n[TASK TRANSFER]\n{message}")
        print(f"Task: {task}")
        # In production, would await human completion
        return {"completed_by": "human", "result": "Human handled this task"}
```

---

## Guardrails That Enable Autonomy

### First Principles: What Are Guardrails For?

Guardrails aren't restrictions—they're enablers. By clearly defining what's allowed, agents can operate with confidence within those bounds.

```
Good Guardrails:                   Bad Guardrails:
├── Clear boundaries               ├── Vague rules
├── Explain the "why"              ├── Arbitrary restrictions
├── Enable autonomy within bounds  ├── Micromanagement
├── Consistent enforcement         ├── Inconsistent application
└── Adaptable to context           └── Rigid regardless of context
```

### Guardrail Implementation

```python
from abc import ABC, abstractmethod
from typing import Tuple

class Guardrail(ABC):
    """Base class for guardrails."""

    @abstractmethod
    async def check(self, action: Dict, context: Dict) -> Tuple[bool, str]:
        """
        Check if action is allowed.
        Returns (allowed, reason).
        """
        pass

    @abstractmethod
    def explain(self) -> str:
        """Explain the purpose of this guardrail."""
        pass


class BudgetGuardrail(Guardrail):
    """Prevent exceeding resource budgets."""

    def __init__(self, max_api_calls: int = 100, max_cost: float = 10.0):
        self.max_api_calls = max_api_calls
        self.max_cost = max_cost
        self.current_calls = 0
        self.current_cost = 0.0

    async def check(self, action: Dict, context: Dict) -> Tuple[bool, str]:
        estimated_calls = action.get("estimated_api_calls", 1)
        estimated_cost = action.get("estimated_cost", 0.1)

        if self.current_calls + estimated_calls > self.max_api_calls:
            return False, f"Would exceed API call budget ({self.max_api_calls})"

        if self.current_cost + estimated_cost > self.max_cost:
            return False, f"Would exceed cost budget (${self.max_cost})"

        return True, "Within budget"

    def explain(self) -> str:
        return f"Prevents exceeding {self.max_api_calls} API calls or ${self.max_cost} cost"

    def record_usage(self, calls: int, cost: float):
        self.current_calls += calls
        self.current_cost += cost


class ScopeGuardrail(Guardrail):
    """Ensure actions stay within defined scope."""

    def __init__(
        self,
        allowed_actions: List[str],
        allowed_resources: List[str],
        llm
    ):
        self.allowed_actions = allowed_actions
        self.allowed_resources = allowed_resources
        self.llm = llm

    async def check(self, action: Dict, context: Dict) -> Tuple[bool, str]:
        action_type = action.get("type", "")
        resources = action.get("resources", [])

        # Check action type
        if action_type and action_type not in self.allowed_actions:
            return False, f"Action type '{action_type}' not in allowed list"

        # Check resources
        for resource in resources:
            if not self._resource_allowed(resource):
                return False, f"Resource '{resource}' not in allowed scope"

        return True, "Within scope"

    def _resource_allowed(self, resource: str) -> bool:
        """Check if resource matches any allowed pattern."""
        import fnmatch
        return any(fnmatch.fnmatch(resource, pattern) for pattern in self.allowed_resources)

    def explain(self) -> str:
        return f"Actions limited to {self.allowed_actions}, resources to {self.allowed_resources}"


class SafetyGuardrail(Guardrail):
    """Prevent potentially harmful actions."""

    def __init__(self, llm):
        self.llm = llm
        self.blocked_patterns = [
            "delete production",
            "drop database",
            "rm -rf",
            "format drive",
            "send to all users",
            "share publicly"
        ]

    async def check(self, action: Dict, context: Dict) -> Tuple[bool, str]:
        action_description = str(action).lower()

        # Quick pattern check
        for pattern in self.blocked_patterns:
            if pattern in action_description:
                return False, f"Action contains blocked pattern: '{pattern}'"

        # LLM safety check for complex cases
        if action.get("requires_safety_review", False):
            is_safe, reason = await self._llm_safety_check(action, context)
            if not is_safe:
                return False, reason

        return True, "Passed safety checks"

    async def _llm_safety_check(self, action: Dict, context: Dict) -> Tuple[bool, str]:
        """Use LLM to evaluate safety of complex actions."""
        prompt = f"""
        Evaluate the safety of this action:

        Action: {action}
        Context: {context}

        Consider:
        1. Could this action cause data loss?
        2. Could this affect production systems?
        3. Could this impact users negatively?
        4. Is this action reversible?
        5. Are there any security concerns?

        Respond with SAFE or UNSAFE, followed by brief reasoning.
        """

        response = await self.llm.ainvoke(prompt)
        is_safe = "SAFE" in response.content.upper() and "UNSAFE" not in response.content.upper()
        return is_safe, response.content

    def explain(self) -> str:
        return "Prevents potentially harmful actions like data deletion, mass messaging, etc."


class GuardrailSystem:
    """Manages multiple guardrails."""

    def __init__(self):
        self.guardrails: List[Guardrail] = []

    def add_guardrail(self, guardrail: Guardrail):
        self.guardrails.append(guardrail)

    async def check_all(self, action: Dict, context: Dict) -> Dict:
        """Check action against all guardrails."""
        results = []
        all_passed = True

        for guardrail in self.guardrails:
            passed, reason = await guardrail.check(action, context)
            results.append({
                "guardrail": guardrail.__class__.__name__,
                "passed": passed,
                "reason": reason,
                "explanation": guardrail.explain()
            })
            if not passed:
                all_passed = False

        return {
            "allowed": all_passed,
            "results": results,
            "blocking_guardrails": [r for r in results if not r["passed"]]
        }

    def explain_all(self) -> str:
        """Explain all active guardrails."""
        explanations = [
            f"- {g.__class__.__name__}: {g.explain()}"
            for g in self.guardrails
        ]
        return "\n".join(explanations)
```

---

## Complete Example: Supervised Multi-Agent System

```python
import asyncio
from langchain_openai import ChatOpenAI

class SupervisedMultiAgentSystem:
    """
    A complete multi-agent system with:
    - Specialized worker agents
    - Supervisor coordination
    - Human-in-the-loop escalation
    - Comprehensive guardrails

    Analogy: A well-run department with clear roles,
    good communication, and appropriate oversight.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")

        # Initialize components
        self.guardrails = GuardrailSystem()
        self.human_manager = HumanInTheLoopManager(HumanInterface())
        self.orchestrator = None
        self.agents: Dict[str, SpecializedAgent] = {}

        # Setup
        self._initialize_guardrails()
        self._initialize_agents()

    def _initialize_guardrails(self):
        """Set up system guardrails."""
        self.guardrails.add_guardrail(BudgetGuardrail(max_api_calls=500, max_cost=50.0))
        self.guardrails.add_guardrail(ScopeGuardrail(
            allowed_actions=["research", "analyze", "summarize", "draft", "review"],
            allowed_resources=["documents/*", "web/*", "internal_db/*"],
            llm=self.llm
        ))
        self.guardrails.add_guardrail(SafetyGuardrail(self.llm))

    def _initialize_agents(self):
        """Set up the agent team."""

        # Create orchestrator
        self.orchestrator = HubAndSpokeOrchestrator(self.llm)

        # Create specialized agents
        agents_config = [
            {
                "name": "researcher",
                "specialization": "research",
                "description": "Searches and gathers information from various sources"
            },
            {
                "name": "analyst",
                "specialization": "analysis",
                "description": "Analyzes data and identifies patterns and insights"
            },
            {
                "name": "writer",
                "specialization": "writing",
                "description": "Creates clear, well-structured written content"
            },
            {
                "name": "reviewer",
                "specialization": "review",
                "description": "Reviews work for quality, accuracy, and completeness"
            }
        ]

        for config in agents_config:
            agent = SpecializedAgent(
                name=config["name"],
                llm=self.llm,
                specialization=config["specialization"],
                description=config["description"]
            )
            self.orchestrator.register_agent(agent)
            self.agents[config["name"]] = agent

    async def execute(self, task: str) -> Dict:
        """
        Execute a task using the supervised multi-agent system.
        """
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print('='*60)

        # 1. Pre-execution guardrail check
        task_action = {
            "type": "complex_task",
            "description": task,
            "estimated_api_calls": 20,
            "estimated_cost": 2.0
        }

        guardrail_result = await self.guardrails.check_all(task_action, {})

        if not guardrail_result["allowed"]:
            print(f"Blocked by guardrails: {guardrail_result['blocking_guardrails']}")
            return {
                "status": "blocked",
                "reason": "guardrails",
                "details": guardrail_result
            }

        # 2. Check human escalation triggers
        escalation_result = await self.human_manager.check_escalation(
            action=task_action,
            context={"task": task}
        )

        if not escalation_result["proceed"]:
            print(f"Blocked by human: {escalation_result}")
            return {
                "status": "blocked",
                "reason": "human_decision",
                "details": escalation_result
            }

        # 3. Execute through orchestrator
        print("\nDecomposing task...")
        result = await self.orchestrator.execute_complex_task(task)

        # 4. Post-execution review
        if result.get("needs_review"):
            print("\nResult flagged for review...")
            review_result = await self.human_manager.check_escalation(
                action={"type": "review_result", "result": result},
                context={"original_task": task}
            )
            result["human_review"] = review_result

        # 5. Return final result
        return {
            "status": "completed",
            "result": result,
            "guardrails_checked": len(self.guardrails.guardrails),
            "agents_involved": list(self.agents.keys())
        }

    async def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            "active_guardrails": self.guardrails.explain_all(),
            "registered_agents": {
                name: {
                    "specialization": agent.specialization,
                    "current_load": agent.current_load
                }
                for name, agent in self.agents.items()
            },
            "escalation_triggers": len(self.human_manager.triggers)
        }


# --- Demo ---

async def demonstrate_supervised_system():
    """Demonstrate the supervised multi-agent system."""

    system = SupervisedMultiAgentSystem()

    # Show system status
    status = await system.get_system_status()
    print("\nSystem Status:")
    print(f"Guardrails:\n{status['active_guardrails']}")
    print(f"\nAgents: {list(status['registered_agents'].keys())}")

    # Execute various tasks
    tasks = [
        "Research the current state of AI regulation in the EU and summarize key points",
        "Analyze customer feedback from Q3 and identify top 3 issues",
        "Draft a project proposal for implementing a new feature"
    ]

    for task in tasks:
        result = await system.execute(task)
        print(f"\nResult status: {result['status']}")


if __name__ == "__main__":
    asyncio.run(demonstrate_supervised_system())
```

---

## Exercises

### Exercise 1: Role Design
Design a multi-agent system for a software development team. Define the agents (PM, developer, QA, DevOps), their interactions, and escalation paths.

### Exercise 2: Custom Guardrails
Create a custom guardrail system for a healthcare application. Consider privacy, accuracy requirements, and regulatory compliance.

### Exercise 3: Escalation Optimization
Analyze an escalation system and identify ways to reduce unnecessary escalations while maintaining safety. Implement adaptive thresholds.

### Exercise 4: Trust Building
Design a transparency system that builds user trust by showing agent reasoning and decision-making in real-time.

---

## Key Takeaways

1. **Multi-agent patterns mirror organizational patterns** - Use centuries of organizational wisdom to design agent systems.

2. **Human-in-the-loop is a spectrum** - From notification to full approval, right-size human involvement.

3. **Guardrails enable, not restrict** - Clear boundaries allow agents to operate confidently within them.

4. **Trust requires transparency** - Show what agents are doing and why.

5. **Supervision scales** - Hierarchical structures allow autonomous operation at lower levels with oversight at higher levels.

---

## Week 13 Conclusion

You've now reached the frontier of Agentic AI. You understand:

- **Autonomy from first principles** - What makes agents truly autonomous
- **Self-improvement through emergence** - How learning arises from simple feedback loops
- **Orchestration through analogy** - How organizational patterns guide multi-agent design

The future of AI is agentic. These systems will operate with increasing autonomy, continuously improve, and coordinate to solve complex problems. By understanding the fundamentals—first principles, analogies, and emergence—you're prepared to build that future responsibly.

---

*"The question of whether a computer can think is no more interesting than the question of whether a submarine can swim."* — Edsger W. Dijkstra

The question isn't whether agents can be autonomous. It's whether we can design autonomy that serves human flourishing.

---

## Continue Your Journey

- Review [Week 13 README](README.md) for the complete overview
- Revisit earlier weeks to see how these advanced concepts build on foundations
- Start building: The best way to learn is by creating

*"We are what we repeatedly do. Excellence, then, is not an act, but a habit."* — Will Durant

Build excellent agents by building agents, repeatedly, learning from each iteration.
