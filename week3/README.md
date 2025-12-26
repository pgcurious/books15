# Week 3: LangGraph Workflows

> "The whole is greater than the sum of its parts." — Aristotle

Welcome to Week 3! This week we transcend simple chains and enter the world of **graph-based orchestration**. LangGraph gives you explicit control over how your agent thinks, decides, and acts—turning implicit behaviors into explicit, debuggable, and extensible workflows.

---

## Learning Objectives

By the end of this week, you will:
- Understand why graphs are the natural representation for complex agent workflows
- Master LangGraph's core primitives: nodes, edges, and state
- Build multi-step agents with conditional branching and cycles
- Implement event-driven architectures for responsive agents
- Design real-world pipelines that handle complexity gracefully

---

## Contents

| Module | Topic | Duration |
|--------|-------|----------|
| 3.1 | [Orchestrating Multi-step Tasks](01_orchestrating_multistep_tasks.md) | 90 min |
| 3.2 | [Event-driven Workflows with LangGraph](02_event_driven_workflows.md) | 75 min |
| 3.3 | [Real-world Pipeline Examples](03_real_world_pipelines.md) | 90 min |

---

## The Three Thinking Frameworks Applied to LangGraph

### First Principles: Why Graphs?

Let's reason from first principles about agent execution:

**First Principle #1:** Agent behavior is a sequence of steps with decision points.

```
Input → Process → Decide → Branch A or Branch B → ... → Output
```

**First Principle #2:** Not all sequences are linear. Complex tasks require:
- **Conditionals**: "If X, then do Y, else do Z"
- **Loops**: "Repeat until condition is met"
- **Parallelism**: "Do A and B simultaneously"
- **Cycles**: "Return to previous step based on feedback"

**First Principle #3:** Graphs are the mathematical structure that captures all these patterns.

```
Linear Chain:    A → B → C → D

Conditional:     A → B ─┬→ C
                        └→ D

Loop:            A → B → C ─┐
                     ↑      │
                     └──────┘

Parallel:        A ─┬→ B ─┬→ D
                    └→ C ─┘
```

**Conclusion:** To build sophisticated agents, we need graph-based orchestration. That's LangGraph.

---

### Analogical Thinking: The Factory Floor

Think of a LangGraph workflow as a **modern factory assembly line**:

| Factory Concept | LangGraph Equivalent |
|-----------------|---------------------|
| Workstations | Nodes (functions that process) |
| Conveyor belts | Edges (connections between nodes) |
| Product being assembled | State (data flowing through) |
| Quality control checkpoints | Conditional edges |
| Rework loops | Cycles in the graph |
| Parallel assembly lines | Parallel branches |
| Factory floor layout | Graph definition |
| Production manager | LangGraph runtime |

Just as a well-designed factory:
- Routes products efficiently between stations
- Makes decisions based on product state
- Can handle exceptions and rework
- Produces consistent, quality output

A LangGraph workflow:
- Routes data efficiently between processing nodes
- Makes decisions based on state conditions
- Handles errors and retries gracefully
- Produces reliable, predictable agent behavior

---

### Emergence Thinking: From Simple Rules to Complex Behavior

Here's the beautiful insight: **complex agent behaviors emerge from simple graph primitives**.

```
Simple Primitives:
├── Nodes: Execute a function, update state
├── Edges: Move from one node to another
├── Conditional Edges: Choose path based on state
└── State: Shared data structure

Combined Create:
├── Multi-turn conversations
├── Tool selection and execution
├── Self-reflection and correction
├── Human-in-the-loop workflows
├── Multi-agent collaboration
└── Autonomous research assistants
```

**The emergence pattern:**
1. Each node does one simple thing
2. Edges define possible paths
3. Conditions determine which path to take
4. State accumulates knowledge
5. Complex, adaptive behavior emerges

No single node is "intelligent." Intelligence emerges from the structure.

---

## Why LangGraph Over Simple Chains?

| Feature | LangChain Chains | LangGraph |
|---------|-----------------|-----------|
| Execution | Linear, sequential | Graph-based, flexible |
| Branching | Limited | Native conditionals |
| Cycles | Not supported | First-class support |
| State | Implicit | Explicit, typed |
| Debugging | Opaque | Transparent, traceable |
| Human-in-loop | Difficult | Built-in support |
| Persistence | Add-on | Native checkpointing |

**When to use Chains:** Simple, linear transformations
**When to use LangGraph:** Anything with decisions, loops, or complexity

---

## The LangGraph Mental Model

```
┌─────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                         STATE                            │   │
│   │   TypedDict or Pydantic model holding all workflow data  │   │
│   │   • Messages, results, flags, counters, etc.             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │  START   │────►│  NODE A  │────►│  NODE B  │               │
│   │          │     │(process) │     │(decide)  │               │
│   └──────────┘     └──────────┘     └────┬─────┘               │
│                                          │                       │
│                         ┌────────────────┼────────────────┐     │
│                         │                │                │     │
│                         ▼                ▼                ▼     │
│                  ┌──────────┐     ┌──────────┐     ┌──────────┐│
│                  │  NODE C  │     │  NODE D  │     │   END    ││
│                  │(action1) │     │(action2) │     │          ││
│                  └──────────┘     └──────────┘     └──────────┘│
│                         │                │                       │
│                         └───────┬────────┘                      │
│                                 ▼                                │
│                          ┌──────────┐                           │
│                          │  NODE E  │──────► END                │
│                          │(combine) │                           │
│                          └──────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before starting Week 3, ensure you have:

1. ✅ Completed Week 1 & 2 (conceptual foundation + LangChain basics)
2. ✅ Python 3.9+
3. ✅ Understanding of basic graph concepts (nodes, edges)

```bash
# Install LangGraph
pip install langgraph

# Verify installation
python -c "import langgraph; print('LangGraph installed successfully!')"
```

---

## Setup for Week 3

```bash
# Navigate to week 3
cd week3/code

# Install dependencies
pip install langgraph langchain-openai langchain-core

# Create .env if not exists
echo "OPENAI_API_KEY=your_key_here" > .env
```

Test your setup:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str

def hello(state: State) -> State:
    return {"message": "Hello, LangGraph!"}

# Build a minimal graph
builder = StateGraph(State)
builder.add_node("hello", hello)
builder.add_edge(START, "hello")
builder.add_edge("hello", END)

graph = builder.compile()
result = graph.invoke({"message": ""})
print(result["message"])  # "Hello, LangGraph!"
```

---

## What You'll Build

By the end of Week 3, you'll have built:

### 1. A Multi-step Research Agent
An agent that:
- Plans research approach
- Gathers information from multiple sources
- Synthesizes findings
- Self-corrects when information is insufficient

### 2. An Event-driven Document Processor
A workflow that:
- Responds to document events
- Routes based on document type
- Processes in parallel when possible
- Handles failures gracefully

### 3. A Production-ready RAG Pipeline
A complete retrieval-augmented generation system with:
- Query understanding
- Multi-source retrieval
- Answer synthesis
- Quality verification

---

## Module Overview

### Module 3.1: Orchestrating Multi-step Tasks
- Graph fundamentals: nodes, edges, state
- Building your first LangGraph workflow
- Conditional branching and cycles
- The ReAct pattern in graph form

### Module 3.2: Event-driven Workflows
- State management and reducers
- Checkpointing and persistence
- Human-in-the-loop patterns
- Streaming and real-time updates

### Module 3.3: Real-world Pipeline Examples
- Production RAG pipelines
- Multi-agent orchestration
- Error handling patterns
- Testing and debugging strategies

---

## Key Concepts Preview

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| StateGraph | Main graph builder class | Defines workflow structure |
| State | TypedDict holding workflow data | Enables typed, traceable execution |
| Node | Function that processes state | Encapsulates processing logic |
| Edge | Connection between nodes | Defines flow paths |
| Conditional Edge | Edge with routing logic | Enables decisions |
| Checkpoint | Saved state snapshot | Enables persistence, replay |
| Reducer | State merge strategy | Handles concurrent updates |

---

## Let's Orchestrate!

Start with [Module 3.1: Orchestrating Multi-step Tasks](01_orchestrating_multistep_tasks.md)

---

## The Journey So Far

```
Week 1: Foundations
       │
       ▼  (What is an agent?)
Week 2: Build Your First Agent
       │
       ▼  (How to build with LangChain?)
Week 3: LangGraph Workflows  ◄── YOU ARE HERE
       │
       ▼  (How to orchestrate complex behavior?)
Week 4: APIs & Real Data
       │
       ▼  (How to connect to the real world?)
```

---

*"Complexity is the enemy of execution. Graphs tame complexity by making it explicit."*
