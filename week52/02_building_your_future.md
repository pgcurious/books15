# Module 52.2: Building Your Agentic Future

> "The best way to predict the future is to create it." — Peter Drucker

## Introduction

You have the knowledge. You have the frameworks. Now it's time to build something that matters. This module guides you through creating a capstone project that demonstrates your integrated mastery and prepares you for your next chapter in the agentic AI landscape.

This isn't just an academic exercise—it's the beginning of your portfolio, your proof of capability, your stake in the ground that says: "I build intelligent systems."

---

## The Capstone Philosophy

### What Makes a Great Capstone?

```
A GREAT CAPSTONE
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   TECHNICAL EXCELLENCE                                                   │
│   ├── Demonstrates multiple techniques in integration                    │
│   ├── Handles edge cases gracefully                                      │
│   ├── Has clear architecture with separation of concerns                 │
│   └── Includes observability and error handling                          │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   THOUGHTFUL DESIGN                                                      │
│   ├── Every choice is justified                                          │
│   ├── Trade-offs are acknowledged                                        │
│   ├── Simpler alternatives were considered                               │
│   └── Future evolution is anticipated                                    │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   REAL VALUE                                                             │
│   ├── Solves a genuine problem                                           │
│   ├── Someone would actually use this                                    │
│   ├── Impact is measurable                                               │
│   └── Domain understanding is evident                                    │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ETHICAL GROUNDING                                                      │
│   ├── Potential harms are considered                                     │
│   ├── Safeguards are implemented                                         │
│   ├── Privacy and security are addressed                                 │
│   └── Human oversight is preserved                                       │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   CLEAR COMMUNICATION                                                    │
│   ├── Documentation explains the "why"                                   │
│   ├── Architecture is visually explained                                 │
│   ├── Limitations are honest                                             │
│   └── Demo is compelling                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### First Principles: What Problem Deserves Your Best Work?

Not all problems are equal. Choose a capstone that:

```
PROBLEM SELECTION CRITERIA
─────────────────────────────────────────────────────────────────────────

1. GENUINE NEED
   Ask: "Would someone pay for this? Would someone thank me for this?"

   Good: "I'll help researchers synthesize papers more effectively"
   Bad:  "I'll make a chatbot because chatbots are cool"

2. AGENT APPROPRIATENESS
   Ask: "Does this problem benefit from agency, or is it just automation?"

   Good: "Dynamic research that adapts based on findings"
   Bad:  "Formatting data from one schema to another"

3. BOUNDED SCOPE
   Ask: "Can I build something meaningful in the time available?"

   Good: "An agent that helps debug specific types of errors"
   Bad:  "An agent that writes entire applications autonomously"

4. DEMONSTRABLE VALUE
   Ask: "Can I show this working and measure improvement?"

   Good: "Compare agent-assisted code review to manual review"
   Bad:  "Makes users feel better about their work"

5. PERSONAL RESONANCE
   Ask: "Do I actually care about this problem?"

   Good: "I've experienced this pain myself"
   Bad:  "This seems like a good resume item"
```

---

## Capstone Design Process

### Phase 1: Problem Definition

#### Step 1: Identify the Pain

```
PAIN POINT DISCOVERY
─────────────────────────────────────────────────────────────────────────

Ask yourself:
├── What tasks do I do repeatedly that feel like they should be automated?
├── What decisions do I make that require gathering scattered information?
├── What expertise do I have that others struggle without?
├── What problems have I seen in my work that remain unsolved?
├── What would I build if I had unlimited time and resources?

Then filter:
├── Does this require reasoning, not just data transformation?
├── Does this benefit from memory and learning?
├── Is there genuine uncertainty that an agent could navigate?
├── Would a simple script or API call solve this equally well?
└── Can I validate success objectively?
```

#### Step 2: Define Success

```python
"""
Success Criteria Template
"""

@dataclass
class CapstoneSuccessCriteria:
    """Define what success looks like before building."""

    # Functional requirements
    must_have: List[str]  # The system must do these things
    should_have: List[str]  # These would make it better
    nice_to_have: List[str]  # Stretch goals

    # Quality attributes
    reliability: str  # e.g., "Succeeds on 80%+ of test cases"
    performance: str  # e.g., "Responds within 30 seconds"
    safety: str  # e.g., "Never takes destructive actions without confirmation"

    # User experience
    who_is_the_user: str  # Be specific
    what_do_they_get: str  # The value proposition
    how_will_they_feel: str  # The experience

    # Evaluation plan
    test_cases: List[str]  # Specific scenarios to test
    metrics: List[str]  # How you'll measure success
    baseline: str  # What you're comparing against

# Example:
research_assistant_criteria = CapstoneSuccessCriteria(
    must_have=[
        "Search multiple academic sources",
        "Synthesize findings into coherent summaries",
        "Track source citations accurately",
        "Handle ambiguous queries by asking clarifying questions"
    ],
    should_have=[
        "Learn user's research interests over time",
        "Identify gaps in the literature",
        "Suggest related topics"
    ],
    nice_to_have=[
        "Generate formatted bibliography",
        "Create visual knowledge maps"
    ],
    reliability="Successfully completes 85% of research tasks",
    performance="Delivers initial synthesis within 5 minutes",
    safety="Never fabricates citations; always provides sources",
    who_is_the_user="Graduate students doing literature reviews",
    what_do_they_get="4x faster literature review with better coverage",
    how_will_they_feel="Confident they haven't missed important papers",
    test_cases=[
        "Well-defined topic with abundant literature",
        "Niche topic with sparse literature",
        "Interdisciplinary topic spanning multiple fields",
        "Rapidly evolving topic with recent publications"
    ],
    metrics=[
        "Papers found vs. expert baseline",
        "Synthesis quality (human evaluation)",
        "Time to completion",
        "User satisfaction"
    ],
    baseline="Manual search using Google Scholar"
)
```

### Phase 2: Architecture Design

#### The Architecture Canvas

```
CAPSTONE ARCHITECTURE CANVAS
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. CORE CAPABILITY                                                       │
│    What is the one thing this system must do extraordinarily well?       │
│                                                                          │
│    [Your answer here]                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. AGENT ARCHITECTURE                                                    │
│    Which pattern best fits?                                              │
│                                                                          │
│    □ Single agent with tools                                            │
│    □ ReAct loop                                                          │
│    □ Multi-agent with coordinator                                        │
│    □ Hierarchical agents                                                 │
│    □ Other: _____________                                                │
│                                                                          │
│    Why this pattern?                                                     │
│    [Your justification]                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. TOOLS NEEDED                                                          │
│                                                                          │
│    Tool Name          │  Purpose              │  Build vs Buy           │
│    ─────────────────────────────────────────────────────────────────    │
│    [tool 1]           │  [purpose]            │  [build/buy/adapt]      │
│    [tool 2]           │  [purpose]            │  [build/buy/adapt]      │
│    [tool 3]           │  [purpose]            │  [build/buy/adapt]      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 4. MEMORY REQUIREMENTS                                                   │
│                                                                          │
│    Short-term (context): [what needs to be in context?]                 │
│    Long-term (persisted): [what needs to survive sessions?]             │
│    Knowledge base: [what domain knowledge is needed?]                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 5. GUARDRAILS & SAFETY                                                   │
│                                                                          │
│    What could go wrong?              │  How do you prevent it?          │
│    ─────────────────────────────────────────────────────────────────    │
│    [failure mode 1]                  │  [mitigation]                    │
│    [failure mode 2]                  │  [mitigation]                    │
│    [failure mode 3]                  │  [mitigation]                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 6. DATA FLOW                                                             │
│                                                                          │
│    Draw the flow from user input to final output:                       │
│                                                                          │
│    User Input                                                            │
│        ↓                                                                 │
│    [step 1]                                                              │
│        ↓                                                                 │
│    [step 2]                                                              │
│        ↓                                                                 │
│    ...                                                                   │
│        ↓                                                                 │
│    Final Output                                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 7. WHAT I'M NOT BUILDING                                                 │
│                                                                          │
│    Explicit scope boundaries:                                            │
│    • [thing that's out of scope 1]                                      │
│    • [thing that's out of scope 2]                                      │
│    • [thing that's out of scope 3]                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Implementation

#### The Build Sequence

```
IMPLEMENTATION ORDER
═══════════════════════════════════════════════════════════════════════════

WEEK 1: Foundation
─────────────────────────────────────────────────────────────────────────
Day 1-2: Environment setup, dependencies, basic project structure
Day 3-4: Core data models, types, and interfaces
Day 5-7: Simplest possible end-to-end flow (happy path only)

Milestone: Can run the system and see basic output

WEEK 2: Core Functionality
─────────────────────────────────────────────────────────────────────────
Day 8-10: Implement all must-have tools
Day 11-12: Add memory layer
Day 13-14: Basic reasoning/planning

Milestone: System handles main use cases

WEEK 3: Robustness
─────────────────────────────────────────────────────────────────────────
Day 15-16: Error handling and edge cases
Day 17-18: Guardrails and safety checks
Day 19-21: Testing and debugging

Milestone: System fails gracefully, handles edge cases

WEEK 4: Polish & Documentation
─────────────────────────────────────────────────────────────────────────
Day 22-23: Performance optimization
Day 24-25: Documentation and README
Day 26-27: Demo preparation
Day 28: Final testing and cleanup

Milestone: Portfolio-ready project
```

---

## Capstone Implementation Patterns

### Pattern A: The Research Agent

An agent that assists with knowledge work by gathering, synthesizing, and organizing information.

```python
"""
Research Agent Capstone Template
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import asyncio

# ═══════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResearchState:
    """State for research agent."""
    query: str
    research_plan: List[str] = None
    sources_found: List[Dict] = None
    key_findings: List[str] = None
    synthesis: str = None
    confidence: float = 0.0
    iterations: int = 0
    max_iterations: int = 5

# ═══════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def academic_search(query: str, limit: int = 10) -> List[Dict]:
    """Search academic databases for papers matching query."""
    # Implementation: Call Semantic Scholar, arXiv, etc.
    pass

@tool
def web_search(query: str, limit: int = 10) -> List[Dict]:
    """Search the web for relevant information."""
    # Implementation: Call search API
    pass

@tool
def read_document(url: str) -> Dict:
    """Read and extract content from a document."""
    # Implementation: Fetch and parse document
    pass

@tool
def save_finding(finding: str, source: str, confidence: float) -> bool:
    """Save a research finding with its source."""
    # Implementation: Store in research memory
    pass

# ═══════════════════════════════════════════════════════════════════════════
# AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

async def plan_research(state: ResearchState) -> ResearchState:
    """Create a research plan based on the query."""

    llm = ChatOpenAI(model="gpt-4o")

    prompt = f"""
    Research Query: {state.query}

    Create a research plan with 3-5 specific search strategies.
    Consider:
    - Different angles on the topic
    - Academic vs. practical sources
    - Recent vs. foundational works

    Return as a numbered list of search strategies.
    """

    response = await llm.ainvoke(prompt)
    state.research_plan = _parse_plan(response.content)
    return state


async def gather_sources(state: ResearchState) -> ResearchState:
    """Execute searches and gather sources."""

    all_sources = []

    for strategy in state.research_plan:
        # Execute appropriate search based on strategy
        if "academic" in strategy.lower():
            sources = await academic_search.ainvoke({"query": strategy})
        else:
            sources = await web_search.ainvoke({"query": strategy})

        all_sources.extend(sources)

    # Deduplicate and rank
    state.sources_found = _dedupe_and_rank(all_sources)
    return state


async def extract_findings(state: ResearchState) -> ResearchState:
    """Extract key findings from sources."""

    llm = ChatOpenAI(model="gpt-4o")
    findings = []

    for source in state.sources_found[:10]:  # Top 10 sources
        content = await read_document.ainvoke({"url": source["url"]})

        prompt = f"""
        Research Query: {state.query}
        Source: {source['title']}
        Content: {content['text'][:5000]}

        Extract 2-3 key findings relevant to the query.
        For each finding, assess confidence (high/medium/low).
        """

        response = await llm.ainvoke(prompt)
        findings.extend(_parse_findings(response.content, source))

    state.key_findings = findings
    return state


async def synthesize(state: ResearchState) -> ResearchState:
    """Synthesize findings into coherent summary."""

    llm = ChatOpenAI(model="gpt-4o")

    prompt = f"""
    Research Query: {state.query}

    Findings:
    {_format_findings(state.key_findings)}

    Synthesize these findings into a coherent summary that:
    1. Answers the original query
    2. Notes areas of consensus
    3. Identifies disagreements or gaps
    4. Cites sources appropriately
    5. Suggests areas for further investigation

    Use an academic tone but remain accessible.
    """

    response = await llm.ainvoke(prompt)
    state.synthesis = response.content

    # Calculate confidence based on source quality and agreement
    state.confidence = _calculate_confidence(state.key_findings)

    return state


async def evaluate_completeness(state: ResearchState) -> str:
    """Decide if research is complete or needs more iteration."""

    state.iterations += 1

    if state.iterations >= state.max_iterations:
        return "complete"

    if state.confidence >= 0.8:
        return "complete"

    if len(state.key_findings) < 5:
        return "need_more_sources"

    return "complete"

# ═══════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_research_graph():
    """Construct the research agent graph."""

    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("plan", plan_research)
    graph.add_node("gather", gather_sources)
    graph.add_node("extract", extract_findings)
    graph.add_node("synthesize", synthesize)

    # Add edges
    graph.add_edge("plan", "gather")
    graph.add_edge("gather", "extract")
    graph.add_edge("extract", "synthesize")

    # Conditional edge based on completeness
    graph.add_conditional_edges(
        "synthesize",
        evaluate_completeness,
        {
            "complete": END,
            "need_more_sources": "plan"  # Loop back with refined plan
        }
    )

    graph.set_entry_point("plan")

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class ResearchAgent:
    """Complete research agent interface."""

    def __init__(self):
        self.graph = build_research_graph()

    async def research(self, query: str) -> Dict[str, Any]:
        """Conduct research on a query."""

        initial_state = ResearchState(query=query)

        final_state = await self.graph.ainvoke(initial_state)

        return {
            "query": query,
            "synthesis": final_state.synthesis,
            "sources": final_state.sources_found,
            "findings": final_state.key_findings,
            "confidence": final_state.confidence,
            "iterations": final_state.iterations
        }
```

### Pattern B: The Workflow Agent

An agent that orchestrates complex multi-step workflows with human oversight.

```python
"""
Workflow Agent Capstone Template

First Principles: Complex work requires decomposition, execution, and oversight.
Analogical: Like a skilled project manager with expert assistants.
Emergence: Workflow optimization emerges from learning which paths succeed.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowTask:
    """A single task in a workflow."""
    id: str
    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    requires_approval: bool = False
    assigned_agent: Optional[str] = None

@dataclass
class WorkflowState:
    """Complete state of a workflow execution."""
    workflow_id: str
    goal: str
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    execution_log: List[Dict] = field(default_factory=list)
    approvals_needed: List[str] = field(default_factory=list)
    completed: bool = False


class WorkflowAgent:
    """
    Orchestrates complex workflows with human-in-the-loop.

    Key capabilities:
    - Decomposes goals into task graphs
    - Assigns tasks to specialist agents
    - Handles dependencies and parallelism
    - Requests human approval at critical points
    - Learns from workflow outcomes
    """

    def __init__(self, llm, specialist_agents: Dict[str, Any]):
        self.llm = llm
        self.specialists = specialist_agents
        self.workflow_memory = WorkflowMemory()

    async def execute_workflow(
        self,
        goal: str,
        approval_callback: Callable[[str], bool]
    ) -> WorkflowState:
        """Execute a complete workflow toward a goal."""

        # 1. Plan the workflow
        state = await self._plan_workflow(goal)

        # 2. Execute tasks respecting dependencies
        while not state.completed:
            # Get executable tasks (dependencies met, not blocked)
            ready_tasks = self._get_ready_tasks(state)

            if not ready_tasks:
                if state.approvals_needed:
                    # Wait for approvals
                    for task_id in state.approvals_needed:
                        task = state.tasks[task_id]
                        approved = approval_callback(
                            f"Approve task '{task.name}'?\n{task.description}"
                        )
                        if approved:
                            task.status = TaskStatus.IN_PROGRESS
                            state.approvals_needed.remove(task_id)
                        else:
                            task.status = TaskStatus.FAILED
                            await self._handle_rejection(state, task)
                    continue
                else:
                    # Check if all tasks done
                    if self._is_complete(state):
                        state.completed = True
                    else:
                        # Stuck - need to replan
                        await self._replan(state)
                    continue

            # Execute ready tasks in parallel
            results = await asyncio.gather(*[
                self._execute_task(task, state)
                for task in ready_tasks
            ])

            # Update state with results
            for task, result in zip(ready_tasks, results):
                self._update_task_result(state, task, result)

        # 3. Learn from workflow
        await self._learn_from_workflow(state)

        return state

    async def _plan_workflow(self, goal: str) -> WorkflowState:
        """Decompose goal into task graph."""

        # Check if we've done similar workflows before
        similar = await self.workflow_memory.find_similar(goal)

        if similar and similar["success_rate"] > 0.8:
            # Reuse successful pattern
            return self._adapt_workflow_template(similar["template"], goal)

        # Generate new plan
        prompt = f"""
        Goal: {goal}

        Decompose this into a workflow with:
        1. Individual tasks (5-15 tasks)
        2. Dependencies between tasks
        3. Which tasks need human approval
        4. Which specialist agent should handle each task

        Available specialists: {list(self.specialists.keys())}

        Return as structured task list with dependencies.
        """

        response = await self.llm.ainvoke(prompt)
        return self._parse_workflow_plan(response.content, goal)

    async def _execute_task(
        self,
        task: WorkflowTask,
        state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute a single task."""

        task.status = TaskStatus.IN_PROGRESS

        # Check if approval needed first
        if task.requires_approval:
            task.status = TaskStatus.WAITING_APPROVAL
            state.approvals_needed.append(task.id)
            return {"status": "waiting_approval"}

        # Get specialist agent
        agent = self.specialists.get(task.assigned_agent)

        if not agent:
            return {
                "status": "error",
                "error": f"No specialist found: {task.assigned_agent}"
            }

        # Gather inputs from dependencies
        inputs = self._gather_dependency_outputs(task, state)

        try:
            result = await agent.execute(
                task=task.description,
                inputs=inputs,
                context=state.execution_log[-10:]  # Recent context
            )

            return {"status": "success", "result": result}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _learn_from_workflow(self, state: WorkflowState):
        """Extract lessons from completed workflow."""

        success = all(
            t.status == TaskStatus.COMPLETED
            for t in state.tasks.values()
        )

        lessons = {
            "goal": state.goal,
            "task_count": len(state.tasks),
            "success": success,
            "task_sequence": [t.id for t in self._topological_sort(state.tasks)],
            "bottlenecks": self._identify_bottlenecks(state),
            "failures": [
                {"task": t.name, "reason": t.result.get("error")}
                for t in state.tasks.values()
                if t.status == TaskStatus.FAILED
            ]
        }

        await self.workflow_memory.store(lessons)

    def _get_ready_tasks(self, state: WorkflowState) -> List[WorkflowTask]:
        """Get tasks whose dependencies are satisfied."""
        ready = []

        for task in state.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            deps_met = all(
                state.tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.dependencies
            )

            if deps_met:
                ready.append(task)

        return ready
```

### Pattern C: The Learning Agent

An agent that explicitly improves its performance over time through experience.

```python
"""
Learning Agent Capstone Template

First Principles: Learning requires prediction, action, feedback, and update.
Analogical: Like a student who improves through practice and reflection.
Emergence: Expertise emerges from accumulated experience and adaptation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Experience:
    """A single learning experience."""
    situation: Dict[str, Any]
    action_taken: str
    outcome: Dict[str, Any]
    success: bool
    lessons: List[str]

@dataclass
class Strategy:
    """A learned strategy with performance tracking."""
    name: str
    description: str
    applicable_situations: List[str]
    success_count: int = 0
    failure_count: int = 0
    avg_quality: float = 0.5

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(1, total)

    @property
    def confidence(self) -> float:
        """Confidence increases with more data."""
        total = self.success_count + self.failure_count
        return min(1.0, total / 100)  # Max confidence at 100 experiences


class LearningAgent:
    """
    An agent that explicitly learns from experience.

    Key mechanisms:
    1. Strategy Library: Collection of approaches with tracked performance
    2. Experience Memory: Episodic memory of past situations
    3. Reflection: Explicit analysis of outcomes to extract lessons
    4. Adaptation: Strategy weights updated based on outcomes
    """

    def __init__(self, llm, tools, initial_strategies: List[Strategy]):
        self.llm = llm
        self.tools = tools
        self.strategies = {s.name: s for s in initial_strategies}
        self.experiences: List[Experience] = []
        self.situation_embeddings = {}

    async def act(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Take action in a situation, learning from the outcome."""

        # 1. Analyze situation
        analysis = await self._analyze_situation(situation)

        # 2. Retrieve relevant experiences
        similar_experiences = await self._retrieve_similar(situation)

        # 3. Select strategy based on situation and past performance
        strategy = await self._select_strategy(analysis, similar_experiences)

        # 4. Generate action using selected strategy
        action = await self._generate_action(situation, strategy, similar_experiences)

        # 5. Execute action
        result = await self._execute_action(action)

        # 6. Evaluate outcome
        outcome = await self._evaluate_outcome(situation, action, result)

        # 7. Learn from experience
        experience = await self._learn(situation, strategy, action, outcome)

        return {
            "action": action,
            "result": result,
            "strategy_used": strategy.name,
            "outcome": outcome,
            "learned": experience.lessons
        }

    async def _select_strategy(
        self,
        analysis: Dict[str, Any],
        similar_experiences: List[Experience]
    ) -> Strategy:
        """Select best strategy using UCB (Upper Confidence Bound)."""

        # Calculate UCB score for each applicable strategy
        scores = {}

        for name, strategy in self.strategies.items():
            # Check if strategy applies to this situation
            if not self._strategy_applies(strategy, analysis):
                continue

            # UCB formula: exploitation + exploration
            exploitation = strategy.success_rate * strategy.avg_quality
            exploration = np.sqrt(2 * np.log(len(self.experiences) + 1) /
                                  max(1, strategy.success_count + strategy.failure_count))

            scores[name] = exploitation + 0.5 * exploration

        # Select highest scoring strategy
        if not scores:
            return self._create_new_strategy(analysis)

        best_name = max(scores, key=scores.get)
        return self.strategies[best_name]

    async def _learn(
        self,
        situation: Dict,
        strategy: Strategy,
        action: str,
        outcome: Dict
    ) -> Experience:
        """Learn from the experience."""

        # 1. Update strategy statistics
        if outcome["success"]:
            strategy.success_count += 1
        else:
            strategy.failure_count += 1

        # Update average quality with exponential moving average
        alpha = 0.1
        strategy.avg_quality = (
            (1 - alpha) * strategy.avg_quality +
            alpha * outcome["quality"]
        )

        # 2. Extract lessons through reflection
        lessons = await self._reflect(situation, strategy, action, outcome)

        # 3. Store experience
        experience = Experience(
            situation=situation,
            action_taken=action,
            outcome=outcome,
            success=outcome["success"],
            lessons=lessons
        )
        self.experiences.append(experience)

        # 4. Consider creating new strategies from lessons
        await self._maybe_create_strategy(lessons)

        return experience

    async def _reflect(
        self,
        situation: Dict,
        strategy: Strategy,
        action: str,
        outcome: Dict
    ) -> List[str]:
        """Reflect on experience to extract generalizable lessons."""

        prompt = f"""
        Situation: {situation}
        Strategy Used: {strategy.name} - {strategy.description}
        Action Taken: {action}
        Outcome: {outcome}

        Reflect on this experience:
        1. Why did this {'succeed' if outcome['success'] else 'fail'}?
        2. What could be done differently?
        3. What general principle can be extracted?
        4. When should this strategy be used/avoided?

        Extract 2-3 concise, generalizable lessons.
        """

        response = await self.llm.ainvoke(prompt)
        return self._parse_lessons(response.content)

    async def _maybe_create_strategy(self, lessons: List[str]):
        """Create new strategy if lessons suggest one."""

        prompt = f"""
        Recent lessons learned:
        {lessons}

        Existing strategies:
        {[s.name for s in self.strategies.values()]}

        Should a new strategy be created based on these lessons?
        If yes, define:
        - Name
        - Description
        - When to apply (situations)

        If no, explain why existing strategies are sufficient.
        """

        response = await self.llm.ainvoke(prompt)

        if "new strategy" in response.content.lower():
            new_strategy = self._parse_new_strategy(response.content)
            if new_strategy:
                self.strategies[new_strategy.name] = new_strategy

    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate report on learning progress."""

        return {
            "total_experiences": len(self.experiences),
            "strategies": {
                name: {
                    "success_rate": f"{s.success_rate:.1%}",
                    "confidence": f"{s.confidence:.1%}",
                    "uses": s.success_count + s.failure_count
                }
                for name, s in self.strategies.items()
            },
            "recent_lessons": [
                e.lessons for e in self.experiences[-5:]
            ],
            "overall_success_rate": sum(
                1 for e in self.experiences if e.success
            ) / max(1, len(self.experiences))
        }
```

---

## Your Capstone Checklist

### Technical Requirements

```
CAPSTONE TECHNICAL CHECKLIST
═══════════════════════════════════════════════════════════════════════════

□ Architecture
  □ Clear separation of concerns
  □ Appropriate agent pattern selected and justified
  □ State management implemented properly
  □ Error handling throughout

□ Functionality
  □ All must-have features implemented
  □ Edge cases handled
  □ Graceful degradation under failure
  □ Performance meets requirements

□ Quality
  □ Code is clean and readable
  □ Functions are documented
  □ Tests cover critical paths
  □ No obvious security vulnerabilities

□ Observability
  □ Logging implemented
  □ Key metrics tracked
  □ Traces available for debugging
  □ Health checks if deployed

□ Safety
  □ Guardrails implemented
  □ Human oversight preserved
  □ Failure modes documented
  □ No dangerous capabilities exposed
```

### Documentation Requirements

```
DOCUMENTATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════

□ README.md
  □ Clear problem statement
  □ Solution approach explained
  □ Architecture diagram
  □ Setup instructions
  □ Usage examples
  □ Known limitations

□ Architecture Document
  □ Component diagram
  □ Data flow diagram
  □ Key design decisions with rationale
  □ Alternative approaches considered

□ User Guide
  □ How to use the system
  □ Expected inputs and outputs
  □ Error messages explained
  □ FAQ

□ Technical Reference
  □ API documentation
  □ Configuration options
  □ Extending the system
```

### Presentation Requirements

```
DEMO CHECKLIST
═══════════════════════════════════════════════════════════════════════════

□ Story
  □ Problem is clearly motivated
  □ Solution approach is logical
  □ Demo flows naturally
  □ Value is evident

□ Technical Depth
  □ Architecture explained visually
  □ Key decisions justified
  □ Code quality demonstrated
  □ Metrics/results shown

□ Honesty
  □ Limitations acknowledged
  □ Failures shown (and how handled)
  □ Future improvements identified
  □ What you learned shared
```

---

## Career Guidance

### Your Agentic AI Career Path

```
CAREER TRAJECTORIES IN AGENTIC AI
═══════════════════════════════════════════════════════════════════════════

TECHNICAL PATHS
─────────────────────────────────────────────────────────────────────────

Agent Engineer
├── Build and deploy agent systems
├── Focus: Architecture, implementation, production
├── Next steps: Senior/Staff Engineer, Architect
└── Key skills: LangChain/Graph, deployment, debugging

ML/AI Engineer
├── Train and optimize models for agent tasks
├── Focus: Fine-tuning, evaluation, performance
├── Next steps: Research Engineer, ML Architect
└── Key skills: PyTorch, training pipelines, benchmarking

Platform Engineer
├── Build infrastructure for agent systems
├── Focus: Scaling, reliability, developer experience
├── Next steps: Principal Engineer, CTO
└── Key skills: Kubernetes, observability, APIs

RESEARCH PATHS
─────────────────────────────────────────────────────────────────────────

Research Scientist
├── Push the boundaries of agent capabilities
├── Focus: Novel architectures, alignment, reasoning
├── Next steps: Senior Researcher, Research Lead
└── Key skills: Mathematics, papers, experimentation

Applied Researcher
├── Bring research into production
├── Focus: Translating papers, practical improvements
├── Next steps: Research Lead, Staff Scientist
└── Key skills: Engineering + research synthesis

PRODUCT/BUSINESS PATHS
─────────────────────────────────────────────────────────────────────────

AI Product Manager
├── Define and guide agent products
├── Focus: User needs, market fit, roadmap
├── Next steps: Director of Product, VP
└── Key skills: Technical literacy, user research, strategy

Technical Founder
├── Build AI-first companies
├── Focus: Vision, team building, execution
├── Next steps: CEO, Serial Entrepreneur
└── Key skills: Everything + risk tolerance

AI Consultant
├── Help organizations adopt agent technology
├── Focus: Assessment, implementation, training
├── Next steps: Practice Lead, Partner
└── Key skills: Communication, breadth, business acumen
```

### Building Your Portfolio

```
PORTFOLIO BUILDING STRATEGY
═══════════════════════════════════════════════════════════════════════════

YOUR PORTFOLIO SHOULD DEMONSTRATE:

1. Range: Different types of agents, different domains
2. Depth: At least one project showing deep technical work
3. Impact: Projects that solve real problems
4. Communication: Clear documentation and explanation
5. Growth: Evidence of learning and improvement over time

SUGGESTED PORTFOLIO STRUCTURE:

/portfolio
├── capstone/              # Your Week 52 capstone
│   ├── README.md
│   ├── demo_video.mp4
│   └── src/
│
├── projects/
│   ├── project_1/         # Show range
│   ├── project_2/
│   └── project_3/
│
├── experiments/           # Show curiosity
│   ├── experiment_1/
│   └── experiment_2/
│
├── writing/               # Show communication
│   ├── blog_posts/
│   └── tutorials/
│
└── contributions/         # Show community engagement
    ├── open_source/
    └── collaborations/
```

---

## Key Takeaways

### 1. Build Something Real
The best capstone solves a genuine problem. Start with the pain, not the technology.

### 2. Design Before Code
Time spent on architecture pays dividends in implementation.

### 3. Simple Is Hard
The goal is the simplest system that meets requirements.

### 4. Document Your Thinking
Future you (and others) need to understand your decisions.

### 5. Ship It
A shipped project beats a perfect plan. Get something working, then iterate.

---

## Exercises

### Exercise 1: Problem Discovery
Interview 3 people about their work. Identify 5 problems that could benefit from agentic AI. Evaluate each against the selection criteria.

### Exercise 2: Architecture Design
Complete the Architecture Canvas for your chosen capstone. Have someone else review it and challenge your decisions.

### Exercise 3: MVP Definition
Define the absolute minimum version of your capstone that would demonstrate value. Build that first.

### Exercise 4: Failure Mode Analysis
List 10 ways your capstone could fail. For each, design a mitigation or detection mechanism.

---

## Next: The Frontier and Beyond

In [Module 52.3: The Frontier and Beyond](03_the_frontier_and_beyond.md), we look at where agentic AI is headed and how you can contribute to shaping its future.

Your capstone is a beginning, not an end. The field is young, and there is much to discover.

---

*"Start where you are. Use what you have. Do what you can."* — Arthur Ashe

The best capstone is the one you build. Start today.
