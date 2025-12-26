# Module 5.1: Designing Collaborative Agents

> "If you want to go fast, go alone. If you want to go far, go together." — African Proverb

## What You'll Learn

- Why single agents have fundamental limitations
- How to design agents that specialize and collaborate
- Communication patterns: direct messaging, broadcast, and blackboard systems
- Delegation strategies: when and how to hand off work
- Supervisor vs. peer-to-peer architectural patterns
- Implementing agent collaboration with LangGraph

---

## First Principles: Why Multi-Agent Systems?

Let's start from the fundamentals. Why would we ever want multiple agents instead of one powerful agent?

### The Single Agent Paradox

A single agent faces an impossible trade-off:

```
                    SINGLE AGENT DILEMMA

     Generalist                        Specialist
    ┌──────────────┐                 ┌──────────────┐
    │ Knows a bit  │                 │ Knows deeply │
    │ about        │                 │ about ONE    │
    │ EVERYTHING   │                 │ thing        │
    └──────────────┘                 └──────────────┘
          │                                │
          ▼                                ▼
    Can attempt                      Can excel at
    any task...                      one task...
          │                                │
          ▼                                ▼
    ...but masters                   ...but fails at
    none                             everything else
```

**The insight**: No single system can be both maximally general AND maximally competent. This is why human civilization developed specialization and trade.

### The Multi-Agent Solution

Multi-agent systems resolve this paradox through **division of labor**:

```
MULTI-AGENT ARCHITECTURE

User Query: "Research quantum computing and write a report"
                            │
                            ▼
                    ┌───────────────┐
                    │  Coordinator  │
                    │    Agent      │
                    └───────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   Research    │ │   Analysis    │ │   Writing     │
    │    Agent      │ │    Agent      │ │    Agent      │
    │               │ │               │ │               │
    │ - Web search  │ │ - Synthesize  │ │ - Structure   │
    │ - Document    │ │ - Compare     │ │ - Draft       │
    │   retrieval   │ │ - Evaluate    │ │ - Edit        │
    └───────────────┘ └───────────────┘ └───────────────┘
```

Each agent can be:
- **Deeply specialized** in its domain
- **Optimized** for its specific task
- **Smaller** (less context, cheaper, faster)
- **Independently testable** and improvable

### Fundamental Requirements for Collaboration

At the atomic level, agent collaboration requires exactly three things:

```python
# The Three Pillars of Multi-Agent Collaboration

class CollaborativeAgent:
    def __init__(self):
        # 1. IDENTITY: The agent knows itself
        self.name = "research_agent"
        self.capabilities = ["web_search", "document_analysis"]
        self.limitations = ["cannot_write_code", "no_math_tools"]

        # 2. COMMUNICATION: The agent can send/receive messages
        self.inbox = []
        self.outbox = []

        # 3. PROTOCOLS: The agent knows the rules of engagement
        self.protocols = {
            "request_help": self.request_from_peer,
            "delegate_task": self.hand_off_to_specialist,
            "report_status": self.update_coordinator
        }
```

Without identity, agents don't know their role. Without communication, they can't coordinate. Without protocols, they can't work together effectively.

---

## Analogical Thinking: Agents as Organizations

Understanding multi-agent systems becomes intuitive when we map them to human organizations we already understand.

### The Organization-Agent Mapping

| Organization Type | Agent Pattern | Communication | Best For |
|-------------------|---------------|---------------|----------|
| **Military** | Hierarchical | Command chain | Clear authority, critical ops |
| **Startup** | Flat/Peer-to-peer | Direct communication | Speed, flexibility |
| **Hospital ER** | Specialist routing | Triage system | Expertise matching |
| **Factory** | Pipeline | Sequential handoff | Predictable workflows |
| **Research Lab** | Collaborative | Shared workspace | Creative problems |
| **Bee Colony** | Swarm | Pheromone signals | Emergent coordination |

### Deep Dive: The Hospital ER Analogy

A hospital emergency room is an excellent model for multi-agent systems:

```
HOSPITAL ER                          MULTI-AGENT SYSTEM
────────────────────────────────────────────────────────────────

Patient arrives                      User query arrives
      │                                    │
      ▼                                    ▼
┌─────────────┐                     ┌─────────────┐
│   Triage    │ ← Assess severity   │   Router    │ ← Classify query
│    Nurse    │                     │   Agent     │
└──────┬──────┘                     └──────┬──────┘
       │                                   │
       │ Route to specialist               │ Route to specialist
       │                                   │
   ┌───┴───┬───────┬───────┐          ┌───┴───┬───────┬───────┐
   ▼       ▼       ▼       ▼          ▼       ▼       ▼       ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│Cardi│ │Neuro│ │Ortho│ │ ER  │   │ SQL │ │Write│ │Code │ │Math │
│ olo │ │logi │ │pedi │ │ Doc │   │Agent│ │Agent│ │Agent│ │Agent│
└─────┘ └─────┘ └─────┘ └─────┘   └─────┘ └─────┘ └─────┘ └─────┘
   │       │       │       │         │       │       │       │
   └───────┴───────┴───────┘         └───────┴───────┴───────┘
           │                                 │
           ▼                                 ▼
    Specialists can                   Agents can consult
    consult each other                each other
           │                                 │
           ▼                                 ▼
    Patient gets                      User gets
    comprehensive care                complete answer
```

**Key insights from this analogy:**

1. **Triage is crucial**: The routing decision determines everything
2. **Specialists must know their limits**: A cardiologist calls orthopedics for bone issues
3. **Handoffs require information transfer**: Medical records = shared state
4. **Complex cases need teams**: Multi-disciplinary = multi-agent

### Deep Dive: The Research Lab Analogy

For creative, open-ended problems, the research lab model works better:

```
RESEARCH LAB MODEL

                    ┌─────────────────────────────────────┐
                    │         SHARED WHITEBOARD           │
                    │   (Working hypotheses, findings,    │
                    │    questions, partial results)      │
                    └─────────────────────────────────────┘
                           ▲      ▲      ▲      ▲
                           │      │      │      │
        ┌──────────────────┼──────┼──────┼──────┼──────────────────┐
        │                  │      │      │      │                  │
   ┌────┴────┐       ┌────┴────┐ │ ┌────┴────┐ │          ┌────┴────┐
   │ Senior  │       │ Data    │ │ │ Domain  │ │          │ Junior  │
   │ Researchr│       │ Analyst │ │ │ Expert  │ │          │ Researchr│
   │         │       │         │ │ │         │ │          │         │
   │ Sets    │       │ Crunches│ │ │ Provides│ │          │ Explores│
   │ direction│       │ numbers │ │ │ context │ │          │ ideas   │
   └─────────┘       └─────────┘ │ └─────────┘ │          └─────────┘
                                 │             │
                           ┌─────┴─────────────┴─────┐
                           │      Principal          │
                           │      Investigator       │
                           │                         │
                           │  Reviews, guides,       │
                           │  makes final calls      │
                           └─────────────────────────┘
```

**Key insights:**

1. **Shared workspace enables parallel work**: Everyone sees the same whiteboard
2. **No strict hierarchy for ideas**: Junior researchers can have breakthroughs
3. **PI provides direction, not dictation**: Guidance without micromanagement
4. **Ideas evolve through iteration**: Build on each other's work

---

## Emergence Thinking: Intelligence from Interaction

The most powerful aspect of multi-agent systems is **emergence**—complex, intelligent behavior that arises from simple agent interactions, but isn't explicitly programmed.

### How Emergence Works

```
INDIVIDUAL RULES                    EMERGENT BEHAVIOR
─────────────────────────────────────────────────────────────────

Rule: "Report when uncertain"       →  System admits limitations
Rule: "Verify important claims"     →  Self-fact-checking
Rule: "Ask specialists for help"    →  Expertise aggregation
Rule: "Share useful findings"       →  Collective memory
Rule: "Flag disagreements"          →  Debate and resolution

                    These simple rules produce:

                    ┌────────────────────────────────────┐
                    │                                    │
                    │   COLLECTIVE INTELLIGENCE          │
                    │                                    │
                    │   - Self-correcting                │
                    │   - Adaptive to new problems       │
                    │   - More robust than any single    │
                    │     agent                          │
                    │   - Gracefully handles failures    │
                    │                                    │
                    └────────────────────────────────────┘
```

### The Ant Colony Example

Ant colonies demonstrate emergence perfectly:

```
INDIVIDUAL ANT RULES:
1. If you find food, leave a pheromone trail back to nest
2. If you smell pheromones, follow them
3. Pheromones evaporate over time

EMERGENT BEHAVIOR:
- Shortest paths to food emerge automatically
- Colony adapts to new food sources
- Dead ends are abandoned
- No ant "knows" the optimal path—the colony does
```

**Applying this to agents:**

```python
# Individual agent rule: Share useful findings
class ResearchAgent:
    def process_query(self, query, shared_memory):
        result = self.search(query)

        # Simple rule: If I found something useful, share it
        if result.is_relevant and result.confidence > 0.7:
            shared_memory.add(
                key=query,
                value=result,
                tags=["research", "verified"]
            )

        return result

# Emergent behavior: The team builds collective knowledge
# without any agent explicitly managing the knowledge base
```

### Designing for Emergence

To enable emergence, design agents with:

1. **Local awareness, global impact**
   - Each agent sees only its local context
   - But actions affect the shared environment

2. **Simple, robust rules**
   - Complex rules are fragile
   - Simple rules compose into complex behavior

3. **Feedback loops**
   - Agents can see effects of their actions
   - Success reinforces, failure adjusts

---

## Communication Patterns

How agents talk to each other determines what kinds of collaboration are possible.

### Pattern 1: Direct Messaging

The simplest pattern—agents send messages directly to each other.

```
DIRECT MESSAGING

Agent A                              Agent B
┌─────────────────┐                 ┌─────────────────┐
│                 │   "Help me      │                 │
│  Research       │   analyze this" │   Analysis      │
│  Agent          │ ───────────────►│   Agent         │
│                 │                 │                 │
│                 │◄─────────────── │                 │
│                 │   "Here's my    │                 │
│                 │   analysis"     │                 │
└─────────────────┘                 └─────────────────┘
```

**Implementation:**

```python
# code/01_direct_messaging.py

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class AgentMessage(TypedDict):
    sender: str
    receiver: str
    content: str
    message_type: str  # "request", "response", "info"

class DirectMessagingState(TypedDict):
    messages: list[AgentMessage]
    current_agent: str
    task_complete: bool

def research_agent(state: DirectMessagingState) -> DirectMessagingState:
    """Research agent that can request help from analysis agent."""

    # Check for responses from other agents
    my_messages = [m for m in state["messages"]
                   if m["receiver"] == "research"]

    if not my_messages:
        # Initial research, then request analysis help
        return {
            "messages": state["messages"] + [{
                "sender": "research",
                "receiver": "analysis",
                "content": "Please analyze these findings: [data]",
                "message_type": "request"
            }],
            "current_agent": "analysis",
            "task_complete": False
        }
    else:
        # Got response, complete task
        return {
            **state,
            "task_complete": True
        }
```

**When to use:**
- Clear sender-receiver relationships
- Simple request-response patterns
- Low number of agents (2-4)

### Pattern 2: Broadcast / Pub-Sub

Agents publish to channels, others subscribe to channels of interest.

```
BROADCAST / PUBLISH-SUBSCRIBE

                    ┌──────────────────────────────────┐
                    │         MESSAGE CHANNELS          │
                    ├──────────────────────────────────┤
                    │  #research    #analysis   #final  │
                    └────────┬─────────┬─────────┬─────┘
                             │         │         │
        ┌────────────────────┼─────────┼─────────┼────────────────────┐
        │                    │         │         │                    │
        ▼                    ▼         ▼         ▼                    ▼
 ┌─────────────┐     ┌─────────────┐        ┌─────────────┐  ┌─────────────┐
 │  Research   │     │  Analysis   │        │   Writing   │  │   Review    │
 │   Agent     │     │   Agent     │        │   Agent     │  │   Agent     │
 │             │     │             │        │             │  │             │
 │ Publishes:  │     │ Publishes:  │        │ Publishes:  │  │ Subscribes: │
 │ #research   │     │ #analysis   │        │ #final      │  │ ALL         │
 │             │     │             │        │             │  │             │
 │ Subscribes: │     │ Subscribes: │        │ Subscribes: │  │             │
 │ (none)      │     │ #research   │        │ #analysis   │  │             │
 └─────────────┘     └─────────────┘        └─────────────┘  └─────────────┘
```

**Implementation:**

```python
# code/01_pubsub_messaging.py

from collections import defaultdict
from typing import Callable

class MessageBus:
    """Simple publish-subscribe message bus for agents."""

    def __init__(self):
        self.subscribers: dict[str, list[Callable]] = defaultdict(list)
        self.message_log: list[dict] = []

    def subscribe(self, channel: str, handler: Callable):
        """Subscribe to a channel."""
        self.subscribers[channel].append(handler)

    def publish(self, channel: str, message: dict, sender: str):
        """Publish a message to a channel."""
        full_message = {
            "channel": channel,
            "sender": sender,
            "content": message,
            "timestamp": time.time()
        }
        self.message_log.append(full_message)

        # Notify all subscribers
        for handler in self.subscribers[channel]:
            handler(full_message)

class PubSubAgent:
    """Agent that communicates via publish-subscribe."""

    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus = bus
        self.received_messages = []

    def subscribe_to(self, channel: str):
        """Subscribe to a channel."""
        self.bus.subscribe(channel, self._handle_message)

    def publish_to(self, channel: str, message: dict):
        """Publish to a channel."""
        self.bus.publish(channel, message, self.name)

    def _handle_message(self, message: dict):
        """Handle incoming message."""
        if message["sender"] != self.name:  # Don't process own messages
            self.received_messages.append(message)
            self.on_message(message)

    def on_message(self, message: dict):
        """Override in subclass to handle messages."""
        pass
```

**When to use:**
- Many agents that need to stay informed
- Loose coupling between agents
- Event-driven architectures

### Pattern 3: Blackboard / Shared State

All agents read from and write to a shared knowledge structure.

```
BLACKBOARD ARCHITECTURE

┌──────────────────────────────────────────────────────────────────┐
│                        BLACKBOARD (Shared State)                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   task_status: "in_progress"                                      │
│   research_findings: ["finding1", "finding2"]                    │
│   analysis_results: {"summary": "...", "confidence": 0.85}       │
│   draft_report: "..."                                            │
│   review_comments: []                                            │
│   final_output: null                                             │
│                                                                   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
    READ │ WRITE            READ │ WRITE            READ │ WRITE
         │                       │                       │
         ▼                       ▼                       ▼
 ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
 │   Research    │      │   Analysis    │      │    Writing    │
 │    Agent      │      │    Agent      │      │    Agent      │
 │               │      │               │      │               │
 │ Reads: task   │      │ Reads: task,  │      │ Reads: all    │
 │ Writes:       │      │  research     │      │ Writes:       │
 │  research     │      │ Writes:       │      │  draft,       │
 │  findings     │      │  analysis     │      │  final        │
 └───────────────┘      └───────────────┘      └───────────────┘
```

**Implementation with LangGraph:**

```python
# code/01_blackboard_pattern.py

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator

class BlackboardState(TypedDict):
    """Shared state (blackboard) for all agents."""
    task: str
    research_findings: Annotated[List[str], operator.add]
    analysis_results: dict
    draft_report: str
    review_comments: Annotated[List[str], operator.add]
    final_output: str
    current_phase: str

def research_agent(state: BlackboardState) -> dict:
    """Research agent reads task, writes findings."""
    task = state["task"]

    # Simulate research
    findings = [
        f"Finding 1 about {task}",
        f"Finding 2 about {task}",
    ]

    return {
        "research_findings": findings,
        "current_phase": "analysis"
    }

def analysis_agent(state: BlackboardState) -> dict:
    """Analysis agent reads findings, writes analysis."""
    findings = state["research_findings"]

    # Simulate analysis
    analysis = {
        "summary": f"Analyzed {len(findings)} findings",
        "key_themes": ["theme1", "theme2"],
        "confidence": 0.85
    }

    return {
        "analysis_results": analysis,
        "current_phase": "writing"
    }

def writing_agent(state: BlackboardState) -> dict:
    """Writing agent reads all, writes draft."""
    findings = state["research_findings"]
    analysis = state["analysis_results"]

    # Simulate writing
    draft = f"""
    Report on: {state['task']}

    Key Findings: {findings}

    Analysis: {analysis['summary']}

    Conclusion: Based on our analysis...
    """

    return {
        "draft_report": draft,
        "current_phase": "review"
    }

# Build the graph
def build_blackboard_graph():
    graph = StateGraph(BlackboardState)

    graph.add_node("research", research_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("writing", writing_agent)

    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "writing")
    graph.add_edge("writing", END)

    graph.set_entry_point("research")

    return graph.compile()
```

**When to use:**
- Complex state that multiple agents need
- Phases where agents build on each other's work
- Need for audit trail / history

---

## Delegation Patterns

How does an agent decide to hand off work to another agent?

### Pattern 1: Capability-Based Delegation

Agents know their capabilities and limitations explicitly.

```python
# code/02_capability_delegation.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AgentCapability:
    name: str
    description: str
    keywords: List[str]
    confidence_threshold: float = 0.7

class CapabilityAwareAgent:
    """Agent that knows what it can and cannot do."""

    def __init__(self, name: str, capabilities: List[AgentCapability]):
        self.name = name
        self.capabilities = capabilities
        self.peer_registry: dict[str, 'CapabilityAwareAgent'] = {}

    def register_peer(self, peer: 'CapabilityAwareAgent'):
        """Register a peer agent."""
        self.peer_registry[peer.name] = peer

    def can_handle(self, task: str) -> tuple[bool, float]:
        """Check if this agent can handle a task."""
        task_lower = task.lower()

        for capability in self.capabilities:
            keyword_matches = sum(
                1 for kw in capability.keywords
                if kw in task_lower
            )
            if keyword_matches > 0:
                confidence = min(1.0, keyword_matches * 0.3 + 0.4)
                return True, confidence

        return False, 0.0

    def find_best_peer(self, task: str) -> Optional['CapabilityAwareAgent']:
        """Find the best peer to handle a task."""
        best_peer = None
        best_confidence = 0.0

        for peer in self.peer_registry.values():
            can_handle, confidence = peer.can_handle(task)
            if can_handle and confidence > best_confidence:
                best_peer = peer
                best_confidence = confidence

        return best_peer

    def process(self, task: str) -> str:
        """Process a task, delegating if necessary."""
        can_handle, confidence = self.can_handle(task)

        if can_handle and confidence > 0.6:
            return self._do_task(task)
        else:
            # Try to delegate
            peer = self.find_best_peer(task)
            if peer:
                return f"[Delegated to {peer.name}] {peer.process(task)}"
            else:
                return f"[{self.name}] Unable to handle: {task}"

    def _do_task(self, task: str) -> str:
        """Actually do the task (override in subclass)."""
        return f"[{self.name}] Completed: {task}"

# Example usage
research_agent = CapabilityAwareAgent(
    name="researcher",
    capabilities=[
        AgentCapability(
            name="web_search",
            description="Search the web for information",
            keywords=["search", "find", "look up", "research"]
        ),
        AgentCapability(
            name="summarize",
            description="Summarize documents",
            keywords=["summarize", "summary", "overview"]
        )
    ]
)

code_agent = CapabilityAwareAgent(
    name="coder",
    capabilities=[
        AgentCapability(
            name="write_code",
            description="Write code in various languages",
            keywords=["code", "program", "function", "implement", "python"]
        ),
        AgentCapability(
            name="debug",
            description="Debug and fix code issues",
            keywords=["debug", "fix", "error", "bug"]
        )
    ]
)

# Register as peers
research_agent.register_peer(code_agent)
code_agent.register_peer(research_agent)
```

### Pattern 2: Supervisor-Based Delegation

A supervisor agent decides who handles what.

```
SUPERVISOR DELEGATION

                    User Query
                         │
                         ▼
              ┌─────────────────────┐
              │    SUPERVISOR       │
              │                     │
              │  1. Analyze query   │
              │  2. Select worker   │
              │  3. Monitor result  │
              │  4. Validate output │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Worker  │    │ Worker  │    │ Worker  │
    │    A    │    │    B    │    │    C    │
    └─────────┘    └─────────┘    └─────────┘
```

**Implementation:**

```python
# code/02_supervisor_delegation.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import json

# Define worker descriptions
WORKERS = {
    "researcher": {
        "description": "Searches for information, gathers data, finds facts",
        "keywords": ["search", "find", "research", "look up", "information"]
    },
    "analyst": {
        "description": "Analyzes data, compares options, evaluates evidence",
        "keywords": ["analyze", "compare", "evaluate", "assess", "review"]
    },
    "writer": {
        "description": "Writes content, drafts documents, creates text",
        "keywords": ["write", "draft", "create", "compose", "document"]
    }
}

class SupervisorState(TypedDict):
    query: str
    selected_worker: str
    worker_result: str
    final_answer: str
    messages: list

def supervisor_agent(state: SupervisorState) -> dict:
    """Supervisor decides which worker to use."""

    llm = ChatOpenAI(model="gpt-4o-mini")

    worker_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in WORKERS.items()
    ])

    prompt = ChatPromptTemplate.from_template("""
    You are a supervisor agent. Your job is to route tasks to the right worker.

    Available workers:
    {workers}

    User query: {query}

    Which worker should handle this query? Respond with just the worker name.
    """)

    response = llm.invoke(
        prompt.format(workers=worker_descriptions, query=state["query"])
    )

    selected = response.content.strip().lower()

    # Validate selection
    if selected not in WORKERS:
        selected = "researcher"  # Default fallback

    return {"selected_worker": selected}

def route_to_worker(state: SupervisorState) -> str:
    """Route to the selected worker."""
    return state["selected_worker"]

def researcher_worker(state: SupervisorState) -> dict:
    """Research worker implementation."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(
        f"Research the following query and provide findings: {state['query']}"
    )

    return {"worker_result": response.content}

def analyst_worker(state: SupervisorState) -> dict:
    """Analysis worker implementation."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(
        f"Analyze the following and provide insights: {state['query']}"
    )

    return {"worker_result": response.content}

def writer_worker(state: SupervisorState) -> dict:
    """Writing worker implementation."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(
        f"Write content based on: {state['query']}"
    )

    return {"worker_result": response.content}

def compile_response(state: SupervisorState) -> dict:
    """Supervisor compiles final response."""
    return {
        "final_answer": f"[Handled by {state['selected_worker']}]\n\n{state['worker_result']}"
    }

def build_supervisor_graph():
    """Build the supervisor delegation graph."""

    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("researcher", researcher_worker)
    graph.add_node("analyst", analyst_worker)
    graph.add_node("writer", writer_worker)
    graph.add_node("compile", compile_response)

    # Add edges
    graph.add_edge("supervisor", route_to_worker)
    graph.add_conditional_edges(
        "supervisor",
        route_to_worker,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer"
        }
    )

    # All workers go to compile
    graph.add_edge("researcher", "compile")
    graph.add_edge("analyst", "compile")
    graph.add_edge("writer", "compile")
    graph.add_edge("compile", END)

    graph.set_entry_point("supervisor")

    return graph.compile()
```

### Pattern 3: Self-Delegation (Recursive)

An agent can delegate to itself for sub-tasks.

```python
# code/02_self_delegation.py

class RecursiveAgent:
    """Agent that can break down tasks and delegate to itself."""

    def __init__(self, llm, max_depth: int = 3):
        self.llm = llm
        self.max_depth = max_depth

    def process(self, task: str, depth: int = 0) -> str:
        """Process a task, possibly breaking it down recursively."""

        if depth >= self.max_depth:
            return self._direct_solve(task)

        # Decide if task should be broken down
        subtasks = self._decompose_if_needed(task)

        if len(subtasks) == 1:
            # No decomposition needed, solve directly
            return self._direct_solve(task)
        else:
            # Recursively solve subtasks
            results = []
            for subtask in subtasks:
                result = self.process(subtask, depth + 1)
                results.append(result)

            # Combine results
            return self._combine_results(task, results)

    def _decompose_if_needed(self, task: str) -> list[str]:
        """Break task into subtasks if complex."""
        prompt = f"""
        Task: {task}

        Should this task be broken into smaller subtasks?
        If yes, list the subtasks (one per line).
        If no, respond with just: NO_DECOMPOSITION
        """

        response = self.llm.invoke(prompt)

        if "NO_DECOMPOSITION" in response.content:
            return [task]
        else:
            return [line.strip() for line in response.content.split("\n") if line.strip()]

    def _direct_solve(self, task: str) -> str:
        """Directly solve a task without decomposition."""
        response = self.llm.invoke(f"Complete this task: {task}")
        return response.content

    def _combine_results(self, original_task: str, results: list[str]) -> str:
        """Combine subtask results into final answer."""
        prompt = f"""
        Original task: {original_task}

        Subtask results:
        {chr(10).join(f'- {r}' for r in results)}

        Combine these results into a coherent final answer.
        """

        response = self.llm.invoke(prompt)
        return response.content
```

---

## Architectural Patterns

### Pattern 1: Supervisor Architecture

A central supervisor coordinates all workers.

```
SUPERVISOR ARCHITECTURE

                         ┌──────────────────┐
                         │                  │
                         │   SUPERVISOR     │
                         │                  │
                         │  - Routes tasks  │
                         │  - Monitors work │
                         │  - Validates     │
                         │    output        │
                         │  - Handles       │
                         │    failures      │
                         │                  │
                         └────────┬─────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
         ┌──────────┐      ┌──────────┐      ┌──────────┐
         │ Worker A │      │ Worker B │      │ Worker C │
         │          │      │          │      │          │
         │ Domain   │      │ Domain   │      │ Domain   │
         │ Expert 1 │      │ Expert 2 │      │ Expert 3 │
         └──────────┘      └──────────┘      └──────────┘

Advantages:
- Clear authority and control
- Easy to debug (one decision-maker)
- Predictable execution flow

Disadvantages:
- Supervisor is a bottleneck
- Single point of failure
- Workers can't help each other directly
```

**Full Implementation:**

```python
# code/03_supervisor_architecture.py

from typing import TypedDict, Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import operator

class SupervisorSystemState(TypedDict):
    """State for supervisor-based system."""
    user_query: str
    plan: List[str]
    current_step: int
    worker_outputs: Annotated[List[dict], operator.add]
    final_response: str
    iteration_count: int

class SupervisorSystem:
    """Complete supervisor-based multi-agent system."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.workers = self._create_workers()
        self.graph = self._build_graph()

    def _create_workers(self) -> dict:
        """Create specialized worker agents."""
        return {
            "researcher": self._make_worker(
                "You are a research specialist. Find information and facts."
            ),
            "analyst": self._make_worker(
                "You are an analyst. Analyze data and provide insights."
            ),
            "writer": self._make_worker(
                "You are a writer. Create clear, well-structured content."
            ),
            "critic": self._make_worker(
                "You are a critic. Review work and suggest improvements."
            )
        }

    def _make_worker(self, system_prompt: str):
        """Create a worker function."""
        def worker(task: str) -> str:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{task}")
            ])
            chain = prompt | self.llm
            return chain.invoke({"task": task}).content
        return worker

    def _supervisor_plan(self, state: SupervisorSystemState) -> dict:
        """Supervisor creates a plan for the task."""

        prompt = ChatPromptTemplate.from_template("""
        You are a supervisor agent. Create a plan to complete this task.

        Available workers: researcher, analyst, writer, critic

        Task: {query}

        Create a step-by-step plan. Each step should specify:
        - Which worker to use
        - What they should do

        Format each step as: WORKER: task description

        Keep the plan to 3-5 steps.
        """)

        response = self.llm.invoke(prompt.format(query=state["user_query"]))

        # Parse plan
        plan = []
        for line in response.content.split("\n"):
            line = line.strip()
            if ":" in line and any(w in line.lower() for w in self.workers.keys()):
                plan.append(line)

        return {"plan": plan, "current_step": 0}

    def _execute_step(self, state: SupervisorSystemState) -> dict:
        """Execute the current step in the plan."""

        current_step = state["current_step"]
        if current_step >= len(state["plan"]):
            return state

        step = state["plan"][current_step]

        # Parse worker and task
        worker_name = None
        for name in self.workers.keys():
            if name in step.lower():
                worker_name = name
                break

        if not worker_name:
            worker_name = "researcher"  # Default

        # Include context from previous steps
        context = ""
        if state["worker_outputs"]:
            context = "Previous work:\n" + "\n".join(
                f"- {o['worker']}: {o['output'][:200]}..."
                for o in state["worker_outputs"]
            )

        task_with_context = f"{step}\n\n{context}"

        # Execute
        output = self.workers[worker_name](task_with_context)

        return {
            "worker_outputs": [{"worker": worker_name, "output": output}],
            "current_step": current_step + 1
        }

    def _should_continue(self, state: SupervisorSystemState) -> str:
        """Check if we should continue executing steps."""
        if state["current_step"] >= len(state["plan"]):
            return "compile"
        return "execute"

    def _compile_response(self, state: SupervisorSystemState) -> dict:
        """Compile worker outputs into final response."""

        outputs = "\n\n".join(
            f"### {o['worker'].title()}'s Work:\n{o['output']}"
            for o in state["worker_outputs"]
        )

        prompt = ChatPromptTemplate.from_template("""
        Compile these worker outputs into a coherent final response.

        Original query: {query}

        Worker outputs:
        {outputs}

        Create a well-structured final response.
        """)

        response = self.llm.invoke(
            prompt.format(query=state["user_query"], outputs=outputs)
        )

        return {"final_response": response.content}

    def _build_graph(self):
        """Build the supervisor graph."""

        graph = StateGraph(SupervisorSystemState)

        graph.add_node("plan", self._supervisor_plan)
        graph.add_node("execute", self._execute_step)
        graph.add_node("compile", self._compile_response)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute",
            self._should_continue,
            {"execute": "execute", "compile": "compile"}
        )
        graph.add_edge("compile", END)

        return graph.compile()

    def run(self, query: str) -> str:
        """Run the multi-agent system."""
        initial_state = {
            "user_query": query,
            "plan": [],
            "current_step": 0,
            "worker_outputs": [],
            "final_response": "",
            "iteration_count": 0
        }

        result = self.graph.invoke(initial_state)
        return result["final_response"]
```

### Pattern 2: Peer-to-Peer Architecture

Agents communicate directly without a central coordinator.

```
PEER-TO-PEER ARCHITECTURE

         ┌──────────────────────────────────────────────┐
         │                                              │
         │             SHARED MESSAGE BUS              │
         │                                              │
         └──────────┬──────────┬──────────┬────────────┘
                    │          │          │
                    ▼          ▼          ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Agent A  │ │ Agent B  │ │ Agent C  │
              │          │ │          │ │          │
              │ Can      │ │ Can      │ │ Can      │
              │ initiate │ │ initiate │ │ initiate │
              │ work     │ │ work     │ │ work     │
              │          │ │          │ │          │
              │ Can      │ │ Can      │ │ Can      │
              │ request  │ │ request  │ │ request  │
              │ help     │ │ help     │ │ help     │
              └──────────┘ └──────────┘ └──────────┘
                    ▲          ▲          ▲
                    │          │          │
                    └──────────┴──────────┘
                    Direct communication

Advantages:
- No bottleneck
- More resilient (no single point of failure)
- Agents can self-organize

Disadvantages:
- Harder to debug
- Can lead to circular dependencies
- Need conflict resolution
```

**Implementation:**

```python
# code/03_peer_to_peer.py

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator

class PeerToPeerState(TypedDict):
    """State for peer-to-peer system."""
    task: str
    messages: Annotated[List[dict], operator.add]
    contributions: Annotated[List[dict], operator.add]
    consensus_reached: bool
    final_output: str
    round_number: int

class PeerToPeerAgent:
    """An agent in a peer-to-peer system."""

    def __init__(self, name: str, specialty: str, llm):
        self.name = name
        self.specialty = specialty
        self.llm = llm

    def contribute(self, state: PeerToPeerState) -> dict:
        """Make a contribution to the shared task."""

        # See what others have contributed
        other_contributions = [
            c for c in state["contributions"]
            if c["agent"] != self.name
        ]

        context = ""
        if other_contributions:
            context = "Other agents' contributions:\n" + "\n".join(
                f"- {c['agent']}: {c['content'][:200]}"
                for c in other_contributions
            )

        prompt = f"""
        You are {self.name}, a specialist in {self.specialty}.

        Task: {state['task']}

        {context}

        Provide your unique contribution based on your specialty.
        Build on others' work where relevant.
        """

        response = self.llm.invoke(prompt)

        return {
            "contributions": [{
                "agent": self.name,
                "specialty": self.specialty,
                "content": response.content,
                "round": state["round_number"]
            }]
        }

    def vote_on_consensus(self, state: PeerToPeerState) -> bool:
        """Vote on whether consensus has been reached."""

        prompt = f"""
        Task: {state['task']}

        All contributions:
        {self._format_contributions(state['contributions'])}

        Do we have enough information to complete this task?
        Respond with YES or NO.
        """

        response = self.llm.invoke(prompt)
        return "YES" in response.content.upper()

    def _format_contributions(self, contributions: List[dict]) -> str:
        return "\n".join(
            f"- {c['agent']} ({c['specialty']}): {c['content']}"
            for c in contributions
        )


class PeerToPeerSystem:
    """Peer-to-peer multi-agent system."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.agents = [
            PeerToPeerAgent("Alex", "research and information gathering", self.llm),
            PeerToPeerAgent("Blake", "analysis and critical thinking", self.llm),
            PeerToPeerAgent("Casey", "synthesis and writing", self.llm),
        ]
        self.max_rounds = 3

    def _contribution_round(self, state: PeerToPeerState) -> dict:
        """All agents contribute in parallel."""

        all_contributions = []
        for agent in self.agents:
            contribution = agent.contribute(state)
            all_contributions.extend(contribution["contributions"])

        return {
            "contributions": all_contributions,
            "round_number": state["round_number"] + 1
        }

    def _check_consensus(self, state: PeerToPeerState) -> dict:
        """Check if consensus is reached."""

        if state["round_number"] >= self.max_rounds:
            return {"consensus_reached": True}

        votes = [agent.vote_on_consensus(state) for agent in self.agents]
        consensus = sum(votes) > len(votes) / 2

        return {"consensus_reached": consensus}

    def _should_continue(self, state: PeerToPeerState) -> str:
        """Decide whether to continue or synthesize."""
        if state["consensus_reached"]:
            return "synthesize"
        return "contribute"

    def _synthesize(self, state: PeerToPeerState) -> dict:
        """Synthesize all contributions into final output."""

        prompt = f"""
        Task: {state['task']}

        Agent contributions from {state['round_number']} rounds:
        {self._format_all_contributions(state['contributions'])}

        Synthesize these contributions into a coherent final response.
        """

        response = self.llm.invoke(prompt)
        return {"final_output": response.content}

    def _format_all_contributions(self, contributions: List[dict]) -> str:
        by_round = {}
        for c in contributions:
            r = c.get("round", 0)
            if r not in by_round:
                by_round[r] = []
            by_round[r].append(c)

        result = []
        for r in sorted(by_round.keys()):
            result.append(f"\n### Round {r + 1}")
            for c in by_round[r]:
                result.append(f"- {c['agent']}: {c['content']}")

        return "\n".join(result)

    def build_graph(self):
        """Build the peer-to-peer graph."""

        graph = StateGraph(PeerToPeerState)

        graph.add_node("contribute", self._contribution_round)
        graph.add_node("check_consensus", self._check_consensus)
        graph.add_node("synthesize", self._synthesize)

        graph.set_entry_point("contribute")
        graph.add_edge("contribute", "check_consensus")
        graph.add_conditional_edges(
            "check_consensus",
            self._should_continue,
            {"contribute": "contribute", "synthesize": "synthesize"}
        )
        graph.add_edge("synthesize", END)

        return graph.compile()
```

### Pattern 3: Hierarchical Architecture

Multiple layers of supervision.

```
HIERARCHICAL ARCHITECTURE

                    ┌────────────────────┐
                    │  EXECUTIVE AGENT   │
                    │                    │
                    │  High-level        │
                    │  decisions         │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌────────────┐  ┌────────────┐  ┌────────────┐
       │ MANAGER A  │  │ MANAGER B  │  │ MANAGER C  │
       │            │  │            │  │            │
       │ Research   │  │ Analysis   │  │ Output     │
       │ Division   │  │ Division   │  │ Division   │
       └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
             │               │               │
         ┌───┴───┐       ┌───┴───┐       ┌───┴───┐
         ▼       ▼       ▼       ▼       ▼       ▼
       ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
       │W1 │   │W2 │   │W3 │   │W4 │   │W5 │   │W6 │
       └───┘   └───┘   └───┘   └───┘   └───┘   └───┘

Advantages:
- Scales to many agents
- Clear chain of command
- Managers can parallelize work

Disadvantages:
- Complex to implement
- Can be slow (many hops)
- Communication overhead
```

---

## Hands-On Exercise: Building a Research Team

Let's build a complete multi-agent research team that demonstrates these patterns.

```python
# code/03_research_team.py

from typing import TypedDict, Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
import operator

# === State Definition ===

class ResearchTeamState(TypedDict):
    """State for the research team."""
    query: str
    research_plan: str
    raw_research: Annotated[List[str], operator.add]
    analysis: str
    draft: str
    review_comments: List[str]
    final_report: str
    current_phase: str

# === Agent Definitions ===

class ResearchTeamAgents:
    """Collection of agents for the research team."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.search_tool = DuckDuckGoSearchRun()

    def planner(self, state: ResearchTeamState) -> dict:
        """Plans the research approach."""

        prompt = ChatPromptTemplate.from_template("""
        You are a research planner. Create a research plan for this query.

        Query: {query}

        Create a plan with 3-5 specific research questions to investigate.
        Format as a numbered list.
        """)

        response = self.llm.invoke(prompt.format(query=state["query"]))

        return {
            "research_plan": response.content,
            "current_phase": "research"
        }

    def researcher(self, state: ResearchTeamState) -> dict:
        """Conducts research based on the plan."""

        # Extract research questions from plan
        questions = state["research_plan"].split("\n")
        questions = [q.strip() for q in questions if q.strip() and q[0].isdigit()]

        findings = []
        for question in questions[:3]:  # Limit to 3 for demo
            # Search for information
            search_results = self.search_tool.run(question)
            findings.append(f"Q: {question}\nFindings: {search_results[:500]}")

        return {
            "raw_research": findings,
            "current_phase": "analysis"
        }

    def analyst(self, state: ResearchTeamState) -> dict:
        """Analyzes research findings."""

        prompt = ChatPromptTemplate.from_template("""
        You are a research analyst. Analyze these findings.

        Original query: {query}

        Research findings:
        {findings}

        Provide:
        1. Key themes and patterns
        2. Conflicting information (if any)
        3. Gaps in the research
        4. Main conclusions
        """)

        response = self.llm.invoke(
            prompt.format(
                query=state["query"],
                findings="\n\n".join(state["raw_research"])
            )
        )

        return {
            "analysis": response.content,
            "current_phase": "writing"
        }

    def writer(self, state: ResearchTeamState) -> dict:
        """Writes the draft report."""

        prompt = ChatPromptTemplate.from_template("""
        You are a technical writer. Write a research report.

        Query: {query}

        Research findings:
        {findings}

        Analysis:
        {analysis}

        Write a well-structured report with:
        - Executive summary
        - Key findings
        - Analysis
        - Conclusions
        - Recommendations
        """)

        response = self.llm.invoke(
            prompt.format(
                query=state["query"],
                findings="\n".join(state["raw_research"]),
                analysis=state["analysis"]
            )
        )

        return {
            "draft": response.content,
            "current_phase": "review"
        }

    def reviewer(self, state: ResearchTeamState) -> dict:
        """Reviews and improves the draft."""

        prompt = ChatPromptTemplate.from_template("""
        You are an editor. Review this draft report.

        Original query: {query}

        Draft:
        {draft}

        Provide:
        1. What's good about this draft
        2. What needs improvement
        3. Specific suggestions

        Then provide an improved version of the report.
        """)

        response = self.llm.invoke(
            prompt.format(query=state["query"], draft=state["draft"])
        )

        return {
            "final_report": response.content,
            "current_phase": "complete"
        }

# === Graph Construction ===

def build_research_team():
    """Build the research team graph."""

    agents = ResearchTeamAgents()

    graph = StateGraph(ResearchTeamState)

    # Add nodes
    graph.add_node("planner", agents.planner)
    graph.add_node("researcher", agents.researcher)
    graph.add_node("analyst", agents.analyst)
    graph.add_node("writer", agents.writer)
    graph.add_node("reviewer", agents.reviewer)

    # Add edges (linear flow for this example)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()

# === Main Execution ===

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("MULTI-AGENT RESEARCH TEAM")
    print("=" * 60)

    team = build_research_team()

    query = "What are the latest developments in quantum computing and their potential impact on cryptography?"

    print(f"\nQuery: {query}\n")
    print("-" * 60)

    result = team.invoke({
        "query": query,
        "research_plan": "",
        "raw_research": [],
        "analysis": "",
        "draft": "",
        "review_comments": [],
        "final_report": "",
        "current_phase": "planning"
    })

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_report"])
```

---

## Key Takeaways

### 1. Multi-Agent Systems Solve the Generalist-Specialist Trade-off
No single agent can be both maximally general and maximally competent. Multi-agent systems enable specialization while maintaining broad capability.

### 2. Three Communication Patterns Cover Most Cases
- **Direct messaging**: Simple, clear, good for small teams
- **Pub/Sub**: Loose coupling, good for event-driven systems
- **Blackboard**: Shared state, good for iterative refinement

### 3. Delegation Requires Explicit Design
Agents need:
- **Self-awareness**: Know capabilities and limitations
- **Peer awareness**: Know who can help
- **Protocols**: Know how to ask for help

### 4. Choose Architecture Based on Problem Type
- **Supervisor**: When you need control and predictability
- **Peer-to-peer**: When you need flexibility and resilience
- **Hierarchical**: When you need to scale

### 5. Emergence is the Goal
Design simple, robust interaction rules. Complex, intelligent behavior should emerge from these interactions, not be explicitly programmed.

---

## What's Next?

In **Module 5.2: Agent Memory Systems**, we'll explore:
- How to give agents short-term and long-term memory
- Memory architectures for multi-agent systems
- How memory enables learning and adaptation

The agents we've built so far forget everything after each task. Memory changes everything.

[Continue to Module 5.2 →](02_agent_memory_systems.md)
