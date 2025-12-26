# Module 5.3: Building Agent Teams

> "Individually, we are one drop. Together, we are an ocean." — Ryunosuke Satoro

## What You'll Learn

- How to design effective agent team compositions
- Orchestrating teams with LangGraph for complex workflows
- Error handling and graceful degradation in multi-agent systems
- Monitoring, debugging, and observability for agent teams
- Production patterns and best practices
- Complete case study: Building a research team from scratch

---

## First Principles: What Makes a Team Effective?

Let's build up from fundamentals. What are the essential elements of an effective agent team?

### The Team Effectiveness Formula

```
Team Effectiveness = Σ(Individual Capability) × Coordination Quality × Shared Context

Where:
├── Individual Capability
│   ├── Specialization depth
│   ├── Tool access
│   └── Prompt engineering
│
├── Coordination Quality
│   ├── Clear handoffs
│   ├── Minimal friction
│   └── Conflict resolution
│
└── Shared Context
    ├── Common goals
    ├── Shared memory
    └── Communication protocols
```

**Key insight**: A team of mediocre individuals with excellent coordination outperforms a team of experts with poor coordination.

### The Five Pillars of Agent Teams

```
                    EFFECTIVE AGENT TEAMS
                            │
    ┌───────────┬───────────┼───────────┬───────────┐
    │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
│ ROLES │  │COMMUN-│  │ COOR- │  │SHARED │  │ACCOUNT│
│       │  │ICATION│  │DINATE │  │CONTEXT│  │ABILITY│
│       │  │       │  │       │  │       │  │       │
│Who    │  │How do │  │Who    │  │What   │  │Who is │
│does   │  │they   │  │decides│  │do     │  │respon-│
│what?  │  │talk?  │  │what?  │  │they   │  │sible  │
│       │  │       │  │       │  │know?  │  │for    │
│       │  │       │  │       │  │       │  │what?  │
└───────┘  └───────┘  └───────┘  └───────┘  └───────┘
```

Let's examine each pillar:

**1. Roles (Specialization)**
- Each agent has a clear purpose
- Responsibilities don't overlap excessively
- Capabilities match assigned tasks

**2. Communication**
- Clear protocols for message passing
- Structured formats for information exchange
- Appropriate channels (direct vs. broadcast)

**3. Coordination**
- Decision-making authority is defined
- Task sequencing is managed
- Conflicts are resolved

**4. Shared Context**
- Common understanding of goals
- Access to shared knowledge
- Awareness of each other's work

**5. Accountability**
- Each output has an owner
- Errors can be traced
- Success/failure is attributable

---

## Analogical Thinking: Teams We Know

### The Film Production Analogy

A film production is an excellent model for agent teams:

```
FILM PRODUCTION                    AGENT TEAM
─────────────────────────────────────────────────────────────

Director                           Supervisor/Orchestrator
├── Creative vision                ├── Task planning
├── Coordinates departments        ├── Coordinates agents
└── Final authority                └── Final decisions

Cinematographer                    Research Agent
├── Captures visual content        ├── Gathers information
├── Technical expertise            ├── Domain expertise
└── Specialized tools              └── Specialized tools

Editor                             Analysis Agent
├── Assembles footage              ├── Synthesizes findings
├── Shapes narrative               ├── Identifies patterns
└── Refines output                 └── Refines understanding

Writer                             Writing Agent
├── Creates dialogue/story         ├── Creates content
├── Ensures coherence              ├── Ensures coherence
└── Revises based on feedback      └── Revises based on review

Producer                           Monitoring/Control
├── Manages resources              ├── Manages costs/tokens
├── Handles logistics              ├── Handles failures
└── Ensures delivery               └── Ensures completion
```

### The Orchestra Analogy

For teams that must work in harmony:

```
ORCHESTRA                          MULTI-AGENT SYSTEM
─────────────────────────────────────────────────────────────

Conductor                          Orchestrator Agent
└── Sets tempo, cues sections      └── Sets pace, triggers agents

Sections (strings, brass, etc.)    Specialized Agent Groups
└── Each has unique role           └── Each has unique capability

Sheet Music                        Shared State/Protocol
└── Everyone reads same score      └── Everyone follows same rules

Rehearsals                         Testing/Debugging
└── Practice coordination          └── Validate coordination

Performance                        Execution
└── Synchronized output            └── Coordinated responses
```

---

## Emergence Thinking: Teams That Self-Organize

The most powerful agent teams don't just follow scripts—they adapt and self-organize.

### Emergent Team Behaviors

```
DESIGNED BEHAVIORS                    EMERGENT BEHAVIORS
────────────────────────────────────────────────────────────────

"Agent A handles research"        →  Agents route work to whoever
"Agent B handles analysis"           is best suited, not just
                                     whoever is designated

"Follow this exact workflow"      →  Team discovers shortcuts
                                     and optimizations

"Report all findings"             →  Team develops shared
                                     vocabulary and conventions

"Ask supervisor when stuck"       →  Agents help each other
                                     before escalating

"Try once, then fail"             →  Team retries with different
                                     approaches automatically
```

### Designing for Emergence

To enable emergent behavior, design with these principles:

```python
# PRINCIPLE 1: Self-Description
# Agents describe their own capabilities

class EmergentAgent:
    def __init__(self):
        self.capabilities = self._discover_capabilities()
        self.can_help_with = self._summarize_strengths()

    def _discover_capabilities(self):
        """Agent introspects to find what it can do."""
        return [tool.name for tool in self.tools]

    def advertise(self):
        """Tell others what I can do."""
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "current_load": self.task_count
        }

# PRINCIPLE 2: Request for Help Protocol
# Any agent can ask any agent for help

    def request_help(self, task, team):
        """Find someone who can help."""
        for agent in team:
            if agent.can_handle(task) and agent != self:
                return agent.assist(task)
        return self.escalate(task)

# PRINCIPLE 3: Learning from Outcomes
# Team remembers what worked

    def record_outcome(self, task, approach, success):
        """Share what worked (or didn't) with the team."""
        self.team_memory.store({
            "task_type": task.type,
            "approach": approach,
            "success": success,
            "agent": self.name
        })
```

---

## Team Composition Patterns

### Pattern 1: The Assembly Line

Sequential processing with handoffs.

```
ASSEMBLY LINE PATTERN

Input → [Agent A] → [Agent B] → [Agent C] → Output
         Research     Analyze     Write

Best for:
- Predictable workflows
- Clear phase transitions
- Independent processing steps

Implementation:
```

```python
# code/06_assembly_line.py

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class AssemblyLineState(TypedDict):
    """State that flows through the assembly line."""
    input_query: str
    research_output: str
    analysis_output: str
    final_output: str
    current_stage: str
    errors: List[str]

class AssemblyLineTeam:
    """Assembly line team pattern."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def research_stage(self, state: AssemblyLineState) -> dict:
        """Stage 1: Research."""
        prompt = f"""
        You are a research specialist.
        Research the following topic and provide findings:

        Topic: {state['input_query']}

        Provide detailed research findings.
        """

        response = self.llm.invoke(prompt)
        return {
            "research_output": response.content,
            "current_stage": "analysis"
        }

    def analysis_stage(self, state: AssemblyLineState) -> dict:
        """Stage 2: Analysis."""
        prompt = f"""
        You are an analyst.
        Analyze these research findings:

        Research: {state['research_output']}

        Provide key insights and patterns.
        """

        response = self.llm.invoke(prompt)
        return {
            "analysis_output": response.content,
            "current_stage": "writing"
        }

    def writing_stage(self, state: AssemblyLineState) -> dict:
        """Stage 3: Writing."""
        prompt = f"""
        You are a writer.
        Create a well-structured report based on:

        Original Query: {state['input_query']}
        Research: {state['research_output']}
        Analysis: {state['analysis_output']}

        Write a comprehensive report.
        """

        response = self.llm.invoke(prompt)
        return {
            "final_output": response.content,
            "current_stage": "complete"
        }

    def build_graph(self):
        """Build the assembly line graph."""
        graph = StateGraph(AssemblyLineState)

        graph.add_node("research", self.research_stage)
        graph.add_node("analysis", self.analysis_stage)
        graph.add_node("writing", self.writing_stage)

        graph.set_entry_point("research")
        graph.add_edge("research", "analysis")
        graph.add_edge("analysis", "writing")
        graph.add_edge("writing", END)

        return graph.compile()
```

### Pattern 2: The Hub and Spoke

Central coordinator with specialized workers.

```
HUB AND SPOKE PATTERN

                    ┌─────────────┐
                    │    HUB      │
                    │ (Supervisor)│
                    └──────┬──────┘
                           │
         ┌─────────┬───────┴───────┬─────────┐
         │         │               │         │
         ▼         ▼               ▼         ▼
    ┌────────┐ ┌────────┐    ┌────────┐ ┌────────┐
    │Spoke A │ │Spoke B │    │Spoke C │ │Spoke D │
    │Research│ │Analysis│    │Writing │ │Review  │
    └────────┘ └────────┘    └────────┘ └────────┘

Best for:
- Complex routing decisions
- Need for central oversight
- Variable workflows

Implementation:
```

```python
# code/06_hub_spoke.py

from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import operator

class HubSpokeState(TypedDict):
    """State for hub and spoke pattern."""
    query: str
    hub_analysis: str
    selected_spokes: List[str]
    spoke_results: Annotated[List[dict], operator.add]
    final_output: str
    iteration: int

class HubSpokeTeam:
    """Hub and spoke team pattern with supervisor."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        self.spokes = {
            "research": "Find and gather information",
            "analysis": "Analyze data and find patterns",
            "writing": "Create written content",
            "coding": "Write or review code",
            "math": "Perform calculations",
        }

    def hub_analyze(self, state: HubSpokeState) -> dict:
        """Hub analyzes the query and selects spokes."""

        spoke_descriptions = "\n".join(
            f"- {name}: {desc}"
            for name, desc in self.spokes.items()
        )

        prompt = f"""
        You are the hub supervisor. Analyze this query and decide which
        specialized workers (spokes) should handle it.

        Available spokes:
        {spoke_descriptions}

        Query: {state['query']}

        List the spokes needed (comma-separated).
        Then briefly describe what each should do.
        """

        response = self.llm.invoke(prompt)

        # Parse selected spokes
        selected = []
        for spoke in self.spokes.keys():
            if spoke in response.content.lower():
                selected.append(spoke)

        # Default to research if none selected
        if not selected:
            selected = ["research"]

        return {
            "hub_analysis": response.content,
            "selected_spokes": selected,
            "iteration": 0
        }

    def execute_spoke(self, spoke_name: str, state: HubSpokeState) -> dict:
        """Execute a single spoke's work."""

        prompt = f"""
        You are the {spoke_name} specialist.
        Your job: {self.spokes[spoke_name]}

        Query: {state['query']}
        Hub instructions: {state['hub_analysis']}

        Previous results from other specialists:
        {self._format_previous_results(state['spoke_results'])}

        Provide your specialized contribution.
        """

        response = self.llm.invoke(prompt)

        return {
            "spoke_results": [{
                "spoke": spoke_name,
                "result": response.content
            }]
        }

    def _format_previous_results(self, results: List[dict]) -> str:
        if not results:
            return "None yet."
        return "\n".join(
            f"- {r['spoke']}: {r['result'][:200]}..."
            for r in results
        )

    def hub_compile(self, state: HubSpokeState) -> dict:
        """Hub compiles results from all spokes."""

        results = self._format_previous_results(state['spoke_results'])

        prompt = f"""
        You are the hub supervisor. Compile the specialists' work into
        a coherent final response.

        Original query: {state['query']}

        Specialist results:
        {results}

        Create a comprehensive final response.
        """

        response = self.llm.invoke(prompt)

        return {"final_output": response.content}

    def should_continue(self, state: HubSpokeState) -> str:
        """Decide next step based on remaining spokes."""
        iteration = state.get("iteration", 0)
        selected = state.get("selected_spokes", [])
        completed = [r["spoke"] for r in state.get("spoke_results", [])]

        remaining = [s for s in selected if s not in completed]

        if remaining:
            return remaining[0]
        return "compile"

    def build_graph(self):
        """Build the hub and spoke graph."""
        graph = StateGraph(HubSpokeState)

        # Hub nodes
        graph.add_node("hub_analyze", self.hub_analyze)
        graph.add_node("compile", self.hub_compile)

        # Spoke nodes
        for spoke_name in self.spokes.keys():
            graph.add_node(
                spoke_name,
                lambda s, name=spoke_name: self.execute_spoke(name, s)
            )

        # Edges
        graph.set_entry_point("hub_analyze")

        # Dynamic routing from hub analysis
        graph.add_conditional_edges(
            "hub_analyze",
            self.should_continue,
            {**{spoke: spoke for spoke in self.spokes.keys()}, "compile": "compile"}
        )

        # Each spoke can go to next spoke or compile
        for spoke in self.spokes.keys():
            graph.add_conditional_edges(
                spoke,
                self.should_continue,
                {**{s: s for s in self.spokes.keys()}, "compile": "compile"}
            )

        graph.add_edge("compile", END)

        return graph.compile()
```

### Pattern 3: The Debate Team

Agents with different perspectives that debate to consensus.

```
DEBATE TEAM PATTERN

        Query
          │
          ▼
    ┌───────────────────────────────────────┐
    │           DEBATE ARENA                 │
    │                                        │
    │   ┌─────────┐     ┌─────────┐         │
    │   │  Pro    │ ◄─► │  Con    │         │
    │   │ Agent   │     │ Agent   │         │
    │   └─────────┘     └─────────┘         │
    │        │               │              │
    │        └───────┬───────┘              │
    │                ▼                      │
    │         ┌───────────┐                 │
    │         │Synthesizer│                 │
    │         │  Agent    │                 │
    │         └───────────┘                 │
    │                                        │
    └───────────────────────────────────────┘
          │
          ▼
     Balanced Output

Best for:
- Complex decisions
- Need for multiple perspectives
- Risk assessment

Implementation:
```

```python
# code/06_debate_team.py

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import operator

class DebateState(TypedDict):
    """State for debate team pattern."""
    question: str
    round_number: int
    pro_arguments: Annotated[List[str], operator.add]
    con_arguments: Annotated[List[str], operator.add]
    synthesis: str
    max_rounds: int

class DebateTeam:
    """Debate team that explores multiple perspectives."""

    def __init__(self, max_rounds: int = 2):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.max_rounds = max_rounds

    def pro_agent(self, state: DebateState) -> dict:
        """Agent that argues FOR the proposition."""

        previous_con = state.get("con_arguments", [])
        previous_pro = state.get("pro_arguments", [])

        context = ""
        if previous_con:
            context = f"Counter-arguments to address: {previous_con[-1]}"
        if previous_pro:
            context += f"\nYour previous arguments: {previous_pro[-1]}"

        prompt = f"""
        You are arguing IN FAVOR of this proposition:
        "{state['question']}"

        {context}

        Round {state['round_number'] + 1} of {state['max_rounds']}.

        Provide your strongest arguments FOR this proposition.
        If there were counter-arguments, address them.
        """

        response = self.llm.invoke(prompt)

        return {
            "pro_arguments": [response.content],
            "round_number": state["round_number"] + 1
        }

    def con_agent(self, state: DebateState) -> dict:
        """Agent that argues AGAINST the proposition."""

        previous_pro = state.get("pro_arguments", [])
        previous_con = state.get("con_arguments", [])

        context = ""
        if previous_pro:
            context = f"Arguments to counter: {previous_pro[-1]}"
        if previous_con:
            context += f"\nYour previous arguments: {previous_con[-1]}"

        prompt = f"""
        You are arguing AGAINST this proposition:
        "{state['question']}"

        {context}

        Round {state['round_number']} of {state['max_rounds']}.

        Provide your strongest arguments AGAINST this proposition.
        Address the pro-side's arguments directly.
        """

        response = self.llm.invoke(prompt)

        return {"con_arguments": [response.content]}

    def synthesizer(self, state: DebateState) -> dict:
        """Agent that synthesizes the debate into balanced output."""

        pro_summary = "\n\n".join(state["pro_arguments"])
        con_summary = "\n\n".join(state["con_arguments"])

        prompt = f"""
        You are a neutral synthesizer. The debate on "{state['question']}"
        has concluded after {state['round_number']} rounds.

        ARGUMENTS IN FAVOR:
        {pro_summary}

        ARGUMENTS AGAINST:
        {con_summary}

        Synthesize these perspectives into a balanced analysis:
        1. Strongest points from each side
        2. Where they agree
        3. Key trade-offs to consider
        4. A nuanced conclusion
        """

        response = self.llm.invoke(prompt)

        return {"synthesis": response.content}

    def should_continue_debate(self, state: DebateState) -> str:
        """Check if debate should continue."""
        if state["round_number"] >= state["max_rounds"]:
            return "synthesize"
        return "pro"

    def build_graph(self):
        """Build the debate graph."""
        graph = StateGraph(DebateState)

        graph.add_node("pro", self.pro_agent)
        graph.add_node("con", self.con_agent)
        graph.add_node("synthesize", self.synthesizer)

        graph.set_entry_point("pro")
        graph.add_edge("pro", "con")
        graph.add_conditional_edges(
            "con",
            self.should_continue_debate,
            {"pro": "pro", "synthesize": "synthesize"}
        )
        graph.add_edge("synthesize", END)

        return graph.compile()

    def run(self, question: str) -> str:
        """Run a debate on a question."""
        graph = self.build_graph()

        result = graph.invoke({
            "question": question,
            "round_number": 0,
            "pro_arguments": [],
            "con_arguments": [],
            "synthesis": "",
            "max_rounds": self.max_rounds
        })

        return result["synthesis"]
```

---

## Error Handling and Graceful Degradation

Production systems must handle failures gracefully.

### Error Handling Patterns

```python
# code/06_error_handling.py

from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import operator
import traceback

class RobustState(TypedDict):
    """State with error tracking."""
    query: str
    results: Annotated[List[dict], operator.add]
    errors: Annotated[List[dict], operator.add]
    retries: dict
    final_output: str
    status: str  # "running", "completed", "failed", "degraded"

class RobustTeam:
    """Team with comprehensive error handling."""

    def __init__(self, max_retries: int = 2):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.max_retries = max_retries

    def safe_execute(self, agent_name: str, agent_fn, state: RobustState) -> dict:
        """Execute an agent with error handling."""

        retries = state.get("retries", {}).get(agent_name, 0)

        try:
            result = agent_fn(state)
            return {
                "results": [{"agent": agent_name, "output": result, "success": True}],
                "status": "running"
            }

        except Exception as e:
            error_info = {
                "agent": agent_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "retry_count": retries
            }

            if retries < self.max_retries:
                # Retry
                new_retries = {**state.get("retries", {}), agent_name: retries + 1}
                return {
                    "errors": [error_info],
                    "retries": new_retries,
                    "status": "running"
                }
            else:
                # Give up on this agent, try to continue
                return {
                    "errors": [error_info],
                    "results": [{
                        "agent": agent_name,
                        "output": f"Agent failed after {retries} retries",
                        "success": False
                    }],
                    "status": "degraded"
                }

    def research_agent(self, state: RobustState) -> str:
        """Research agent that might fail."""
        # Simulate potential failure
        response = self.llm.invoke(f"Research: {state['query']}")
        return response.content

    def fallback_agent(self, state: RobustState) -> dict:
        """Fallback when primary agents fail."""

        failed_agents = [
            e["agent"] for e in state.get("errors", [])
            if e.get("retry_count", 0) >= self.max_retries
        ]

        prompt = f"""
        Some agents failed: {failed_agents}
        Original query: {state['query']}

        Provide a best-effort response with limited capabilities.
        Be transparent about limitations.
        """

        response = self.llm.invoke(prompt)

        return {
            "results": [{"agent": "fallback", "output": response.content, "success": True}],
            "status": "degraded"
        }

    def should_use_fallback(self, state: RobustState) -> str:
        """Decide if we need to use fallback."""
        if state.get("status") == "degraded":
            return "fallback"
        return "continue"


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = {}
        self.state = {}  # "closed", "open", "half-open"

    def record_success(self, agent_name: str):
        """Record a successful call."""
        self.failures[agent_name] = 0
        self.state[agent_name] = "closed"

    def record_failure(self, agent_name: str):
        """Record a failed call."""
        self.failures[agent_name] = self.failures.get(agent_name, 0) + 1

        if self.failures[agent_name] >= self.failure_threshold:
            self.state[agent_name] = "open"

    def allow_request(self, agent_name: str) -> bool:
        """Check if a request should be allowed."""
        current_state = self.state.get(agent_name, "closed")

        if current_state == "closed":
            return True
        elif current_state == "open":
            # In production, check if reset_timeout has passed
            return False
        else:  # half-open
            return True

    def call(self, agent_name: str, fn, *args, **kwargs):
        """Call with circuit breaker protection."""
        if not self.allow_request(agent_name):
            raise CircuitBreakerOpen(f"Circuit breaker open for {agent_name}")

        try:
            result = fn(*args, **kwargs)
            self.record_success(agent_name)
            return result
        except Exception as e:
            self.record_failure(agent_name)
            raise


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass
```

### Graceful Degradation Strategies

```
GRACEFUL DEGRADATION

Level 0: FULL CAPABILITY
All agents working normally
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ A ✓│ │ B ✓│ │ C ✓│ │ D ✓│
└────┘ └────┘ └────┘ └────┘

Level 1: REDUCED CAPABILITY
One agent failed, others compensate
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ A ✓│ │ B ✗│ │ C ✓│ │ D ✓│
└────┘ └────┘ └────┘ └────┘
        A & C handle B's tasks

Level 2: MINIMAL CAPABILITY
Multiple agents failed, core functionality only
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ A ✗│ │ B ✗│ │ C ✓│ │ D ✓│
└────┘ └────┘ └────┘ └────┘
        Only C & D available

Level 3: FALLBACK MODE
Most agents failed, single generalist handles all
┌────┐ ┌────┐ ┌────┐ ┌────────────┐
│ A ✗│ │ B ✗│ │ C ✗│ │ FALLBACK ✓ │
└────┘ └────┘ └────┘ └────────────┘
```

---

## Monitoring and Observability

You can't improve what you can't measure.

### Key Metrics for Agent Teams

```python
# code/06_monitoring.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import time

@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    agent_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0
    token_usage: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0
        return self.successful_calls / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0
        return self.total_latency_ms / self.total_calls


@dataclass
class TeamMetrics:
    """Metrics for the entire team."""
    team_name: str
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    task_latencies: List[float] = field(default_factory=list)

    @property
    def task_completion_rate(self) -> float:
        if self.total_tasks == 0:
            return 0
        return self.completed_tasks / self.total_tasks

    @property
    def avg_task_latency_ms(self) -> float:
        if not self.task_latencies:
            return 0
        return sum(self.task_latencies) / len(self.task_latencies)


class TeamMonitor:
    """Monitor for multi-agent teams."""

    def __init__(self, team_name: str):
        self.metrics = TeamMetrics(team_name=team_name)

    def start_task(self) -> str:
        """Start tracking a task."""
        task_id = f"task_{datetime.now().timestamp()}"
        self.metrics.total_tasks += 1
        return task_id

    def end_task(self, task_id: str, success: bool, latency_ms: float):
        """End tracking a task."""
        if success:
            self.metrics.completed_tasks += 1
        else:
            self.metrics.failed_tasks += 1
        self.metrics.task_latencies.append(latency_ms)

    def record_agent_call(
        self,
        agent_name: str,
        success: bool,
        latency_ms: float,
        tokens: int = 0,
        error: Optional[str] = None
    ):
        """Record an agent call."""
        if agent_name not in self.metrics.agent_metrics:
            self.metrics.agent_metrics[agent_name] = AgentMetrics(agent_name)

        agent = self.metrics.agent_metrics[agent_name]
        agent.total_calls += 1
        agent.total_latency_ms += latency_ms
        agent.token_usage += tokens

        if success:
            agent.successful_calls += 1
        else:
            agent.failed_calls += 1
            if error:
                agent.errors.append(error)

    def get_report(self) -> str:
        """Generate a metrics report."""
        lines = [
            f"=== Team: {self.metrics.team_name} ===",
            f"Total Tasks: {self.metrics.total_tasks}",
            f"Completed: {self.metrics.completed_tasks}",
            f"Failed: {self.metrics.failed_tasks}",
            f"Completion Rate: {self.metrics.task_completion_rate:.1%}",
            f"Avg Task Latency: {self.metrics.avg_task_latency_ms:.0f}ms",
            "",
            "--- Agent Performance ---"
        ]

        for name, agent in self.metrics.agent_metrics.items():
            lines.extend([
                f"\n{name}:",
                f"  Calls: {agent.total_calls}",
                f"  Success Rate: {agent.success_rate:.1%}",
                f"  Avg Latency: {agent.avg_latency_ms:.0f}ms",
                f"  Tokens Used: {agent.token_usage}",
            ])

            if agent.errors:
                lines.append(f"  Recent Errors: {agent.errors[-3:]}")

        return "\n".join(lines)


class MonitoredAgent:
    """Base class for monitored agents."""

    def __init__(self, name: str, llm, monitor: TeamMonitor):
        self.name = name
        self.llm = llm
        self.monitor = monitor

    def invoke(self, prompt: str) -> str:
        """Invoke with monitoring."""
        start_time = time.time()
        success = True
        error = None

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_agent_call(
                self.name,
                success=success,
                latency_ms=latency_ms,
                error=error
            )
```

### Tracing with LangSmith

```python
# code/06_tracing.py

import os
from langsmith import traceable
from langsmith.run_trees import RunTree

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-team"

class TracedTeam:
    """Team with LangSmith tracing."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    @traceable(name="research_agent", run_type="chain")
    def research_agent(self, query: str) -> str:
        """Traced research agent."""
        response = self.llm.invoke(f"Research: {query}")
        return response.content

    @traceable(name="analysis_agent", run_type="chain")
    def analysis_agent(self, research: str) -> str:
        """Traced analysis agent."""
        response = self.llm.invoke(f"Analyze: {research}")
        return response.content

    @traceable(name="team_execute", run_type="chain")
    def execute(self, query: str) -> str:
        """Traced team execution."""
        research = self.research_agent(query)
        analysis = self.analysis_agent(research)
        return analysis
```

---

## Complete Case Study: Research Team

Let's build a production-ready research team from scratch.

```python
# code/06_complete_research_team.py

"""
Complete Multi-Agent Research Team
===================================
A production-ready team for research tasks with:
- Specialized agents (planner, researcher, analyst, writer, reviewer)
- Shared memory system
- Error handling and retries
- Monitoring and observability
- Graceful degradation
"""

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from dataclasses import dataclass, field
from datetime import datetime
import operator
import uuid
import time


# ============================================================
# STATE DEFINITION
# ============================================================

class ResearchTeamState(TypedDict):
    """Complete state for the research team."""
    # Input
    task_id: str
    query: str

    # Planning
    research_plan: List[str]
    current_step: int

    # Research
    research_findings: Annotated[List[dict], operator.add]
    sources: Annotated[List[str], operator.add]

    # Analysis
    analysis: str
    key_insights: List[str]

    # Writing
    draft: str

    # Review
    review_feedback: str
    final_report: str

    # Control
    current_phase: str
    errors: Annotated[List[dict], operator.add]
    retry_counts: dict
    status: str  # running, completed, failed, degraded

    # Metrics
    start_time: float
    phase_timings: dict


# ============================================================
# SHARED MEMORY
# ============================================================

class TeamMemory:
    """Shared memory system for the team."""

    def __init__(self, team_id: str):
        self.team_id = team_id
        self.embeddings = OpenAIEmbeddings()

        # Semantic memory for past research
        self.knowledge_store = Chroma(
            collection_name=f"team_{team_id}_knowledge",
            embedding_function=self.embeddings
        )

        # Working memory for current task
        self.working_memory: dict = {}

    def store_finding(self, finding: str, source: str, metadata: dict = None):
        """Store a research finding."""
        self.knowledge_store.add_texts(
            texts=[finding],
            metadatas=[{
                "source": source,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }]
        )

    def retrieve_relevant(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant past findings."""
        try:
            docs = self.knowledge_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except:
            return []

    def set_working(self, key: str, value):
        """Set working memory."""
        self.working_memory[key] = value

    def get_working(self, key: str, default=None):
        """Get from working memory."""
        return self.working_memory.get(key, default)


# ============================================================
# AGENT DEFINITIONS
# ============================================================

class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, llm, memory: TeamMemory, monitor=None):
        self.name = name
        self.llm = llm
        self.memory = memory
        self.monitor = monitor

    def invoke(self, prompt: str) -> str:
        """Invoke with monitoring."""
        start = time.time()
        try:
            response = self.llm.invoke(prompt)
            if self.monitor:
                self.monitor.record_agent_call(
                    self.name, True, (time.time() - start) * 1000
                )
            return response.content
        except Exception as e:
            if self.monitor:
                self.monitor.record_agent_call(
                    self.name, False, (time.time() - start) * 1000, error=str(e)
                )
            raise


class PlannerAgent(BaseAgent):
    """Plans the research approach."""

    def plan(self, state: ResearchTeamState) -> dict:
        """Create a research plan."""

        # Check for relevant past research
        past_research = self.memory.retrieve_relevant(state["query"], k=2)
        past_context = ""
        if past_research:
            past_context = f"Relevant past research:\n" + "\n".join(past_research)

        prompt = f"""
        You are a research planner. Create a detailed research plan.

        Query: {state['query']}

        {past_context}

        Create a step-by-step research plan with 3-5 specific questions to investigate.
        Format each step as a clear, searchable question.
        Number each step.
        """

        response = self.invoke(prompt)

        # Parse steps
        steps = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                steps.append(line.lstrip("0123456789.-) "))

        if not steps:
            steps = [state["query"]]

        return {
            "research_plan": steps,
            "current_step": 0,
            "current_phase": "research",
            "phase_timings": {
                **state.get("phase_timings", {}),
                "planning": time.time() - state["start_time"]
            }
        }


class ResearcherAgent(BaseAgent):
    """Conducts research."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_tool = DuckDuckGoSearchRun()

    def research(self, state: ResearchTeamState) -> dict:
        """Conduct research on current step."""

        current_step = state["current_step"]
        if current_step >= len(state["research_plan"]):
            return {"current_phase": "analysis"}

        question = state["research_plan"][current_step]

        # Search for information
        try:
            search_results = self.search_tool.run(question)
        except:
            search_results = "Search failed. Using general knowledge."

        # Process with LLM
        prompt = f"""
        You are a researcher. Analyze these search results for:
        Question: {question}

        Search Results:
        {search_results[:2000]}

        Provide:
        1. Key findings (bullet points)
        2. Reliability assessment (high/medium/low)
        3. Gaps in information
        """

        response = self.invoke(prompt)

        # Store in memory
        self.memory.store_finding(
            response,
            source="web_search",
            metadata={"question": question}
        )

        return {
            "research_findings": [{
                "question": question,
                "findings": response,
                "step": current_step
            }],
            "sources": [f"Web search: {question}"],
            "current_step": current_step + 1
        }


class AnalystAgent(BaseAgent):
    """Analyzes research findings."""

    def analyze(self, state: ResearchTeamState) -> dict:
        """Analyze all research findings."""

        findings_text = "\n\n".join(
            f"Q: {f['question']}\nFindings: {f['findings']}"
            for f in state["research_findings"]
        )

        prompt = f"""
        You are a research analyst. Synthesize these findings:

        Original Query: {state['query']}

        Research Findings:
        {findings_text}

        Provide:
        1. Executive Summary (2-3 sentences)
        2. Key Insights (numbered list)
        3. Patterns and Themes
        4. Contradictions or Uncertainties
        5. Conclusions
        """

        response = self.invoke(prompt)

        # Extract key insights
        insights = []
        for line in response.split("\n"):
            if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-", "•")):
                insights.append(line.strip())

        return {
            "analysis": response,
            "key_insights": insights[:5],
            "current_phase": "writing",
            "phase_timings": {
                **state.get("phase_timings", {}),
                "analysis": time.time() - state["start_time"]
            }
        }


class WriterAgent(BaseAgent):
    """Writes the research report."""

    def write(self, state: ResearchTeamState) -> dict:
        """Write the draft report."""

        prompt = f"""
        You are a technical writer. Create a research report.

        Query: {state['query']}

        Research Findings:
        {self._format_findings(state['research_findings'])}

        Analysis:
        {state['analysis']}

        Write a well-structured report with:
        - Title
        - Executive Summary
        - Key Findings
        - Detailed Analysis
        - Conclusions
        - Recommendations

        Use clear headings and professional tone.
        """

        response = self.invoke(prompt)

        return {
            "draft": response,
            "current_phase": "review",
            "phase_timings": {
                **state.get("phase_timings", {}),
                "writing": time.time() - state["start_time"]
            }
        }

    def _format_findings(self, findings: List[dict]) -> str:
        return "\n".join(
            f"- {f['question']}: {f['findings'][:300]}..."
            for f in findings
        )


class ReviewerAgent(BaseAgent):
    """Reviews and improves the report."""

    def review(self, state: ResearchTeamState) -> dict:
        """Review and finalize the report."""

        prompt = f"""
        You are an editor. Review and improve this research report.

        Original Query: {state['query']}

        Draft Report:
        {state['draft']}

        Review for:
        1. Accuracy and completeness
        2. Clarity and structure
        3. Grammar and style
        4. Missing information

        Provide:
        1. Brief feedback (what's good, what needs improvement)
        2. The improved final report

        Separate feedback and report with "---FINAL REPORT---"
        """

        response = self.invoke(prompt)

        # Split feedback and final report
        if "---FINAL REPORT---" in response:
            parts = response.split("---FINAL REPORT---")
            feedback = parts[0].strip()
            final_report = parts[1].strip()
        else:
            feedback = "Report approved with minor edits."
            final_report = response

        return {
            "review_feedback": feedback,
            "final_report": final_report,
            "current_phase": "complete",
            "status": "completed",
            "phase_timings": {
                **state.get("phase_timings", {}),
                "review": time.time() - state["start_time"]
            }
        }


# ============================================================
# TEAM ORCHESTRATION
# ============================================================

class ResearchTeam:
    """Complete research team with all agents."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory = TeamMemory(team_id="research_team")

        # Create agents
        self.planner = PlannerAgent("planner", self.llm, self.memory)
        self.researcher = ResearcherAgent("researcher", self.llm, self.memory)
        self.analyst = AnalystAgent("analyst", self.llm, self.memory)
        self.writer = WriterAgent("writer", self.llm, self.memory)
        self.reviewer = ReviewerAgent("reviewer", self.llm, self.memory)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the team workflow graph."""

        graph = StateGraph(ResearchTeamState)

        # Add nodes
        graph.add_node("plan", self.planner.plan)
        graph.add_node("research", self.researcher.research)
        graph.add_node("analyze", self.analyst.analyze)
        graph.add_node("write", self.writer.write)
        graph.add_node("review", self.reviewer.review)

        # Add edges
        graph.set_entry_point("plan")
        graph.add_edge("plan", "research")

        # Research loop
        graph.add_conditional_edges(
            "research",
            self._should_continue_research,
            {"research": "research", "analyze": "analyze"}
        )

        graph.add_edge("analyze", "write")
        graph.add_edge("write", "review")
        graph.add_edge("review", END)

        return graph.compile()

    def _should_continue_research(self, state: ResearchTeamState) -> str:
        """Check if more research needed."""
        if state["current_step"] < len(state["research_plan"]):
            return "research"
        return "analyze"

    def run(self, query: str) -> dict:
        """Run the research team on a query."""

        initial_state = {
            "task_id": str(uuid.uuid4()),
            "query": query,
            "research_plan": [],
            "current_step": 0,
            "research_findings": [],
            "sources": [],
            "analysis": "",
            "key_insights": [],
            "draft": "",
            "review_feedback": "",
            "final_report": "",
            "current_phase": "planning",
            "errors": [],
            "retry_counts": {},
            "status": "running",
            "start_time": time.time(),
            "phase_timings": {}
        }

        result = self.graph.invoke(initial_state)

        return {
            "task_id": result["task_id"],
            "query": result["query"],
            "final_report": result["final_report"],
            "key_insights": result["key_insights"],
            "sources": result["sources"],
            "status": result["status"],
            "total_time": time.time() - result["start_time"],
            "phase_timings": result["phase_timings"]
        }


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 70)
    print("MULTI-AGENT RESEARCH TEAM")
    print("=" * 70)

    team = ResearchTeam()

    query = "What are the latest advancements in sustainable aviation fuel and their potential to reduce aviation's carbon footprint?"

    print(f"\nQuery: {query}")
    print("-" * 70)
    print("\nStarting research...\n")

    result = team.run(query)

    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)

    print(f"\nTask ID: {result['task_id']}")
    print(f"Status: {result['status']}")
    print(f"Total Time: {result['total_time']:.2f}s")

    print("\n--- Key Insights ---")
    for i, insight in enumerate(result['key_insights'][:5], 1):
        print(f"{i}. {insight}")

    print("\n--- Sources ---")
    for source in result['sources'][:5]:
        print(f"  • {source}")

    print("\n--- Final Report ---")
    print(result['final_report'][:2000])
    if len(result['final_report']) > 2000:
        print("\n... [truncated]")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Multi-agent teams require clear roles and responsibilities
    2. Shared memory enables context passing between agents
    3. Sequential workflows are easier to debug than parallel
    4. Error handling should be built in from the start
    5. Monitoring helps identify bottlenecks and failures
    """)
```

---

## Key Takeaways

### 1. Team Effectiveness = Capability × Coordination × Context
Individual agent quality matters, but coordination and shared context are multipliers.

### 2. Choose Patterns Based on Problem Type
- **Assembly Line**: Predictable, sequential workflows
- **Hub and Spoke**: Complex routing, need for oversight
- **Debate Team**: Multiple perspectives, risk assessment
- **Peer-to-Peer**: Emergent coordination, flexible roles

### 3. Design for Failure
Production systems must handle:
- Individual agent failures (retry, fallback)
- Cascading failures (circuit breakers)
- Graceful degradation (reduced capability)

### 4. Observe Everything
You can't improve what you can't measure:
- Track success rates, latencies, token usage
- Use tracing for debugging
- Monitor team-level and agent-level metrics

### 5. Emergence Requires Simple, Robust Rules
Complex team behavior should emerge from simple interaction rules, not be explicitly programmed.

---

## What's Next?

Congratulations! You've completed Week 5 on Multi-Agent Systems. You now understand:
- How to design agents that collaborate and delegate
- Short-term and long-term memory systems
- Building production-ready agent teams

In **Week 6: Guardrails & Safety**, we'll learn:
- Compliance and responsible AI practices
- Guardrails to prevent bias, hallucinations, and errors
- Monitoring and observability for production systems

The agents we've built are powerful—Week 6 ensures they're also safe and reliable.

[Continue to Week 6 →](../week6/README.md)

---

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Multi-Agent Patterns Paper](https://arxiv.org/abs/2308.00352)
- [LangSmith Tracing](https://docs.smith.langchain.com/)
- [Anthropic's Constitutional AI](https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback)
