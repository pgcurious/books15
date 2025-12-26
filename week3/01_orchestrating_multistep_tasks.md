# Module 3.1: Orchestrating Multi-step Tasks

## What You'll Learn
- Why sequential chains break down for complex tasks
- LangGraph fundamentals: StateGraph, nodes, edges, and state
- Building workflows with conditional branching
- Implementing cycles for iterative refinement
- The ReAct pattern expressed as a graph

---

## First Principles: The Limits of Linear Execution

Let's start with a fundamental question: **Why do chains break?**

### The Chain Assumption

LangChain's LCEL assumes:
```
Input → Step 1 → Step 2 → Step 3 → Output
```

**First Principle #1:** This works when every step is guaranteed to succeed and the path is predetermined.

But real-world agent tasks often require:

```python
# Pseudocode for a real agent task
def research_topic(topic):
    plan = create_plan(topic)
    results = []

    while not sufficient(results):      # Loop until done
        if need_web_search(plan):       # Conditional
            results += web_search(...)
        elif need_database(plan):       # Conditional
            results += query_db(...)

        if quality_check_fails(results): # Conditional
            plan = revise_plan(plan)     # Go back!

    return synthesize(results)
```

**First Principle #2:** Complex tasks have:
- **Decisions** (if/else)
- **Iterations** (while loops)
- **Backtracking** (revise and retry)

**First Principle #3:** Graphs are the mathematical structure that naturally represents decisions, iterations, and backtracking.

---

## Analogical Thinking: The Choose-Your-Own-Adventure Book

Remember those books where you make choices and jump to different pages?

```
Page 1: You enter a dark cave.
        - If you light a torch, go to page 15
        - If you proceed in darkness, go to page 23

Page 15: The torch reveals a treasure chest!
         - If you open it, go to page 42
         - If you search for traps first, go to page 31
```

This is exactly how LangGraph works:

| Book Concept | LangGraph Equivalent |
|--------------|---------------------|
| Current page | Current state |
| Page content | Node function |
| Choice options | Conditional edges |
| "Go to page X" | Edge to next node |
| Your progress | State accumulation |
| Reaching "The End" | END node |

The story (agent behavior) emerges from the structure of choices (graph), not a predetermined path.

---

## Part 1: LangGraph Fundamentals

### The Three Pillars

Every LangGraph workflow has three essential components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE THREE PILLARS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. STATE                 2. NODES                3. EDGES       │
│  ┌───────────────┐        ┌───────────────┐      ─────────►     │
│  │ {             │        │               │      ────┬────►     │
│  │   messages: []│        │   function    │          │          │
│  │   result: ""  │        │   that        │      ────┴────►     │
│  │   step: 0     │        │   transforms  │                     │
│  │ }             │        │   state       │      Conditional    │
│  └───────────────┘        └───────────────┘                     │
│                                                                  │
│  What flows through       What processes         How nodes      │
│  the workflow             the data               connect        │
└─────────────────────────────────────────────────────────────────┘
```

### Pillar 1: State

State is a TypedDict or Pydantic model that holds all data flowing through your workflow.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State that flows through our agent graph."""

    # Messages use a special reducer to append, not replace
    messages: Annotated[list, add_messages]

    # Simple fields replace on update
    current_step: str
    research_results: list[str]
    is_complete: bool
```

**Key Insight:** State is immutable. Nodes return updates, not modified state.

### Pillar 2: Nodes

Nodes are functions that take state, perform some processing, and return state updates.

```python
def research_node(state: AgentState) -> dict:
    """A node that performs research."""
    # Access current state
    messages = state["messages"]

    # Do some processing
    result = perform_research(messages[-1].content)

    # Return state UPDATE (not full state!)
    return {
        "research_results": state["research_results"] + [result],
        "current_step": "synthesize"
    }
```

**Key Insight:** Nodes return partial state updates. LangGraph handles merging.

### Pillar 3: Edges

Edges define how nodes connect. They can be:

1. **Static edges:** Always go from A to B
2. **Conditional edges:** Route based on state

```python
# Static edge
builder.add_edge("node_a", "node_b")

# Conditional edge
def route_decision(state: AgentState) -> str:
    """Decide which node to go to next."""
    if state["is_complete"]:
        return "synthesize"
    elif len(state["research_results"]) < 3:
        return "research"
    else:
        return "review"

builder.add_conditional_edges(
    "decide",
    route_decision,
    {
        "synthesize": "synthesize_node",
        "research": "research_node",
        "review": "review_node"
    }
)
```

---

## Part 2: Your First LangGraph Workflow

Let's build a simple research workflow step by step.

```python
# code/01_first_graph.py

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STEP 1: Define State
# =============================================================================

class ResearchState(TypedDict):
    """State for our research workflow."""
    messages: Annotated[list, add_messages]  # Conversation history
    topic: str                                # What to research
    findings: list[str]                       # Research results
    synthesis: str                            # Final answer
    iteration: int                            # Current iteration

# =============================================================================
# STEP 2: Define Nodes
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def plan_research(state: ResearchState) -> dict:
    """Plan what aspects of the topic to research."""
    topic = state["topic"]

    response = llm.invoke([
        SystemMessage(content="You are a research planner. Break down topics into 3 key questions."),
        HumanMessage(content=f"What are the 3 most important questions to answer about: {topic}")
    ])

    return {
        "messages": [AIMessage(content=f"Research plan:\n{response.content}")],
        "iteration": 1
    }

def gather_information(state: ResearchState) -> dict:
    """Gather information about the topic."""
    topic = state["topic"]
    iteration = state["iteration"]

    # In production, this would call actual search APIs
    response = llm.invoke([
        SystemMessage(content="You are a research assistant. Provide detailed, factual information."),
        HumanMessage(content=f"Provide key facts and insights about: {topic} (Focus area {iteration})")
    ])

    return {
        "findings": [response.content],
        "iteration": iteration + 1
    }

def synthesize_findings(state: ResearchState) -> dict:
    """Synthesize all findings into a coherent response."""
    findings = state["findings"]
    topic = state["topic"]

    findings_text = "\n\n".join([f"Finding {i+1}:\n{f}" for i, f in enumerate(findings)])

    response = llm.invoke([
        SystemMessage(content="You are a research synthesizer. Create clear, comprehensive summaries."),
        HumanMessage(content=f"Synthesize these findings about '{topic}' into a coherent summary:\n\n{findings_text}")
    ])

    return {
        "synthesis": response.content,
        "messages": [AIMessage(content=response.content)]
    }

# =============================================================================
# STEP 3: Define Routing Logic
# =============================================================================

def should_continue_research(state: ResearchState) -> str:
    """Decide whether to continue gathering or synthesize."""
    if state["iteration"] >= 3:
        return "synthesize"
    return "gather"

# =============================================================================
# STEP 4: Build the Graph
# =============================================================================

def build_research_graph():
    """Build and compile the research workflow graph."""

    # Create the graph builder with our state type
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("plan", plan_research)
    builder.add_node("gather", gather_information)
    builder.add_node("synthesize", synthesize_findings)

    # Add edges
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "gather")

    # Conditional edge: gather more or synthesize?
    builder.add_conditional_edges(
        "gather",
        should_continue_research,
        {
            "gather": "gather",      # Loop back
            "synthesize": "synthesize" # Move forward
        }
    )

    builder.add_edge("synthesize", END)

    # Compile and return
    return builder.compile()

# =============================================================================
# STEP 5: Run the Workflow
# =============================================================================

if __name__ == "__main__":
    # Build the graph
    graph = build_research_graph()

    # Visualize (optional, requires graphviz)
    # print(graph.get_graph().draw_mermaid())

    # Run the workflow
    initial_state = {
        "messages": [],
        "topic": "The impact of Large Language Models on software development",
        "findings": [],
        "synthesis": "",
        "iteration": 0
    }

    print("=" * 60)
    print("RESEARCH WORKFLOW")
    print("=" * 60)
    print(f"Topic: {initial_state['topic']}\n")

    # Execute with streaming to see progress
    for step in graph.stream(initial_state):
        node_name = list(step.keys())[0]
        print(f"[{node_name}] Completed")

        if node_name == "gather":
            print(f"  Iteration: {step[node_name].get('iteration', '?')}")
        elif node_name == "synthesize":
            print(f"\n{'='*60}")
            print("FINAL SYNTHESIS:")
            print("="*60)
            print(step[node_name].get("synthesis", "No synthesis"))
```

### The Graph Visualization

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │  plan   │
                    └────┬────┘
                         │
                         ▼
               ┌─────────────────┐
          ┌───►│     gather      │◄───┐
          │    └────────┬────────┘    │
          │             │             │
          │    ┌────────┴────────┐    │
          │    │  continue?      │    │
          │    └────────┬────────┘    │
          │             │             │
          │    ┌────────┼────────┐    │
          │    │        │        │    │
          │    ▼        │        ▼    │
          │  "gather"   │   "synthesize"
          │    │        │        │
          └────┘        │        ▼
                        │   ┌─────────┐
                        │   │synthesize│
                        │   └────┬────┘
                        │        │
                        │        ▼
                        │   ┌─────────┐
                        └──►│   END   │
                            └─────────┘
```

---

## Part 3: Understanding State Reducers

### The Problem: Concurrent Updates

What happens when two nodes try to update the same field?

```python
# Node A returns: {"messages": [msg1]}
# Node B returns: {"messages": [msg2]}
# What should final state be?
```

### The Solution: Reducers

Reducers define how to merge updates. The `add_messages` reducer appends instead of replacing:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # With reducer: messages get APPENDED
    messages: Annotated[list, add_messages]

    # Without reducer: value gets REPLACED
    current_node: str
```

### Common Reducers

```python
from operator import add

class State(TypedDict):
    # Append lists
    messages: Annotated[list, add_messages]
    items: Annotated[list, add]  # Python's add for lists

    # Sum numbers
    count: Annotated[int, lambda a, b: a + b]

    # Keep latest (default behavior, no annotation needed)
    status: str
```

---

## Part 4: Conditional Branching Patterns

### Pattern 1: Simple If/Else

```python
def route_by_type(state: State) -> str:
    if state["document_type"] == "email":
        return "process_email"
    elif state["document_type"] == "pdf":
        return "process_pdf"
    else:
        return "process_generic"

builder.add_conditional_edges(
    "classify",
    route_by_type,
    {
        "process_email": "email_node",
        "process_pdf": "pdf_node",
        "process_generic": "generic_node"
    }
)
```

### Pattern 2: Multiple Conditions (Complex Routing)

```python
def complex_router(state: State) -> str:
    """Route based on multiple conditions."""
    has_errors = len(state["errors"]) > 0
    needs_review = state["confidence"] < 0.8
    is_complete = state["is_complete"]

    if has_errors:
        return "error_handler"
    elif is_complete and not needs_review:
        return "finalize"
    elif needs_review:
        return "human_review"
    else:
        return "continue_processing"
```

### Pattern 3: Parallel Branching

```python
from langgraph.graph import StateGraph

# Send to multiple nodes simultaneously
builder.add_conditional_edges(
    "start",
    lambda x: ["branch_a", "branch_b", "branch_c"],  # Return list!
    {
        "branch_a": "node_a",
        "branch_b": "node_b",
        "branch_c": "node_c"
    }
)
```

---

## Part 5: Cycles and Iterative Refinement

### The Self-Correcting Agent

One of LangGraph's superpowers is native support for cycles—allowing agents to reflect and improve.

```python
# code/02_self_correcting_agent.py

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini")

class WriterState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    draft: str
    critique: str
    revision_count: int
    is_approved: bool

def write_draft(state: WriterState) -> dict:
    """Write initial draft or revision."""
    task = state["task"]
    critique = state.get("critique", "")
    revision = state["revision_count"]

    if critique:
        prompt = f"""Revise this draft based on the critique.

Original task: {task}
Previous critique: {critique}
Current draft: {state['draft']}

Write an improved version addressing all critique points."""
    else:
        prompt = f"Write a concise, professional response for: {task}"

    response = llm.invoke([
        SystemMessage(content="You are an expert writer. Be concise and clear."),
        HumanMessage(content=prompt)
    ])

    return {
        "draft": response.content,
        "revision_count": revision + 1,
        "messages": [AIMessage(content=f"Draft v{revision + 1}:\n{response.content}")]
    }

def critique_draft(state: WriterState) -> dict:
    """Critique the current draft."""
    draft = state["draft"]
    task = state["task"]

    response = llm.invoke([
        SystemMessage(content="""You are a critical editor. Evaluate if the draft:
1. Fully addresses the task
2. Is clear and concise
3. Has proper structure

If it passes all criteria, respond with just: APPROVED
Otherwise, provide specific, actionable feedback."""),
        HumanMessage(content=f"Task: {task}\n\nDraft:\n{draft}")
    ])

    critique = response.content
    is_approved = "APPROVED" in critique.upper()

    return {
        "critique": critique,
        "is_approved": is_approved,
        "messages": [AIMessage(content=f"Critique: {critique}")]
    }

def should_revise(state: WriterState) -> str:
    """Decide whether to revise or finalize."""
    if state["is_approved"]:
        return "finalize"
    elif state["revision_count"] >= 3:
        return "finalize"  # Max revisions reached
    else:
        return "revise"

def finalize(state: WriterState) -> dict:
    """Finalize the draft."""
    return {
        "messages": [AIMessage(content=f"Final version (after {state['revision_count']} revision(s)):\n\n{state['draft']}")]
    }

def build_writer_graph():
    builder = StateGraph(WriterState)

    builder.add_node("write", write_draft)
    builder.add_node("critique", critique_draft)
    builder.add_node("finalize", finalize)

    # Start with writing
    builder.add_edge(START, "write")

    # After writing, always critique
    builder.add_edge("write", "critique")

    # After critique, decide: revise or finalize
    builder.add_conditional_edges(
        "critique",
        should_revise,
        {
            "revise": "write",    # CYCLE back to write!
            "finalize": "finalize"
        }
    )

    builder.add_edge("finalize", END)

    return builder.compile()

# Demo
if __name__ == "__main__":
    graph = build_writer_graph()

    result = graph.invoke({
        "messages": [],
        "task": "Write a professional email declining a meeting invitation due to a scheduling conflict",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "is_approved": False
    })

    print("Final Draft:")
    print(result["draft"])
    print(f"\nTotal revisions: {result['revision_count']}")
```

### The Cycle Pattern Visualized

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                ┌─────────────────┐
            ┌──►│     write       │
            │   └────────┬────────┘
            │            │
            │            ▼
            │   ┌─────────────────┐
            │   │    critique     │
            │   └────────┬────────┘
            │            │
            │   ┌────────┴────────┐
            │   │                 │
            │   ▼                 ▼
            │ "revise"        "finalize"
            │   │                 │
            └───┘                 ▼
                         ┌─────────────────┐
                         │    finalize     │
                         └────────┬────────┘
                                  │
                                  ▼
                            ┌─────────┐
                            │   END   │
                            └─────────┘
```

---

## Part 6: The ReAct Pattern as a Graph

The ReAct (Reason + Act) pattern is fundamental to tool-using agents. Let's express it explicitly as a LangGraph workflow.

```python
# code/03_react_agent.py

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import json

load_dotenv()

# =============================================================================
# TOOLS
# =============================================================================

@tool
def search(query: str) -> str:
    """Search for information on a topic."""
    # Mock search results
    return f"Search results for '{query}': Found relevant information about {query}."

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        import math
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    # Mock weather
    return f"Weather in {location}: 72°F, Sunny"

tools = [search, calculate, get_weather]
tool_map = {t.name: t for t in tools}

# =============================================================================
# STATE
# =============================================================================

class ReActState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: list
    tool_results: list

# =============================================================================
# NODES
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

def reason(state: ReActState) -> dict:
    """The 'Reason' step: LLM decides what to do next."""
    messages = state["messages"]

    # Add system message if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [
            SystemMessage(content="""You are a helpful assistant with access to tools.
Think step by step. Use tools when needed to find information or perform calculations.
When you have enough information to answer, provide a final response without calling tools.""")
        ] + messages

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "tool_calls": response.tool_calls if response.tool_calls else []
    }

def act(state: ReActState) -> dict:
    """The 'Act' step: Execute tool calls."""
    tool_calls = state["tool_calls"]
    results = []

    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
        else:
            result = f"Unknown tool: {tool_name}"

        results.append(ToolMessage(
            content=str(result),
            tool_call_id=call["id"]
        ))

    return {
        "messages": results,
        "tool_results": [r.content for r in results]
    }

# =============================================================================
# ROUTING
# =============================================================================

def should_act(state: ReActState) -> Literal["act", "end"]:
    """Decide whether to execute tools or finish."""
    tool_calls = state.get("tool_calls", [])

    if tool_calls:
        return "act"
    return "end"

# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_react_graph():
    builder = StateGraph(ReActState)

    builder.add_node("reason", reason)
    builder.add_node("act", act)

    builder.add_edge(START, "reason")

    builder.add_conditional_edges(
        "reason",
        should_act,
        {
            "act": "act",
            "end": END
        }
    )

    # After acting, go back to reasoning
    builder.add_edge("act", "reason")

    return builder.compile()

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    graph = build_react_graph()

    # Test queries
    queries = [
        "What is 15% of 230?",
        "What's the weather in San Francisco?",
        "Search for recent advances in AI and summarize"
    ]

    for query in queries:
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        result = graph.invoke({
            "messages": [HumanMessage(content=query)],
            "tool_calls": [],
            "tool_results": []
        })

        # Get final AI message
        final_message = result["messages"][-1]
        print(f"Answer: {final_message.content}\n")
```

### The ReAct Loop Visualized

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
               ┌─────────────────┐
          ┌───►│     reason      │◄───┐
          │    │  (LLM thinks)   │    │
          │    └────────┬────────┘    │
          │             │             │
          │    ┌────────┴────────┐    │
          │    │   tool_calls?   │    │
          │    └────────┬────────┘    │
          │             │             │
          │    ┌────────┼────────┐    │
          │    │        │        │    │
          │    ▼        │        ▼    │
          │   yes       │        no   │
          │    │        │        │    │
          │    ▼        │        ▼    │
          │ ┌──────┐    │   ┌────────┐│
          │ │  act │    │   │  END   ││
          │ │(tools)│   │   └────────┘│
          │ └──┬───┘    │             │
          │    │        │             │
          └────┘        │             │
                        │             │
     ReAct Loop         │             │
     (Reason → Act →    │             │
      Reason → ...)     │             │
                        └─────────────┘
```

**Emergence Insight:** The ReAct pattern creates emergent problem-solving behavior:
1. Each node is simple (reason or act)
2. The cycle allows unlimited iterations
3. Complex multi-tool sequences emerge naturally

---

## Summary: Orchestration Patterns

| Pattern | When to Use | Key Feature |
|---------|-------------|-------------|
| Linear | Simple transformations | `A → B → C → END` |
| Conditional | Decisions based on content | Router function |
| Cycle | Iterative refinement | Edge back to earlier node |
| ReAct | Tool-using agents | Reason → Act → Reason loop |
| Parallel | Independent operations | Multiple edges from one node |

---

## Exercises

1. **Add error handling**: Modify the ReAct agent to handle tool failures gracefully

2. **Implement timeout**: Add a maximum iteration count to prevent infinite loops

3. **Add reflection**: Create a "reflect" node that summarizes what the agent learned

4. **Multi-agent**: Create two specialized agents (researcher + writer) that pass work between them

---

## Key Takeaways

1. **Graphs capture complex control flow** that chains cannot express
2. **State is immutable**—nodes return updates, not modified state
3. **Reducers handle concurrent updates** (e.g., `add_messages` for appending)
4. **Conditional edges enable decisions** based on runtime state
5. **Cycles enable iteration and self-correction**—a superpower for agents
6. **ReAct is a graph pattern**: Reason → Act → Reason → ...

---

## Next Steps

In [Module 3.2: Event-driven Workflows with LangGraph](02_event_driven_workflows.md), we'll explore:
- State management and persistence
- Checkpointing for long-running workflows
- Human-in-the-loop patterns
- Streaming and real-time updates

---

*"A graph is not just a data structure. It's a way of thinking about complex systems."*
