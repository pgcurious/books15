# Module 3.2: Event-driven Workflows with LangGraph

## What You'll Learn
- Deep dive into state management and reducer patterns
- Checkpointing for persistence and recovery
- Human-in-the-loop (HITL) workflow design
- Streaming for real-time user feedback
- Interrupts and breakpoints for workflow control

---

## First Principles: What Is Event-driven Architecture?

### The Fundamental Insight

**First Principle #1:** Traditional programming is request-response. Event-driven programming is react-to-change.

```
Request-Response:
  User Request → Process → Response → Done

Event-Driven:
  Event₁ → React → State Change → Event₂ → React → State Change → ...
```

**First Principle #2:** In event-driven systems, components don't call each other directly. They emit events and react to events.

**First Principle #3:** This decoupling enables:
- **Persistence**: Save state between events
- **Resilience**: Resume after failures
- **Flexibility**: Add/remove handlers without changing others
- **Human-in-loop**: Wait for human events indefinitely

### Why Event-Driven for Agents?

Agent workflows need:
- To wait for tool responses (async events)
- To pause for human approval (external events)
- To survive server restarts (persistence)
- To handle long-running operations (state management)

Event-driven architecture provides all of these naturally.

---

## Analogical Thinking: The Restaurant Kitchen

Think of LangGraph as a **restaurant kitchen during service**:

| Restaurant | LangGraph |
|------------|-----------|
| Order ticket | Initial event/message |
| Kitchen stations | Nodes (prep, grill, plate) |
| Order state (cooking, ready) | Graph state |
| Ticket spike | Checkpointer |
| Expo calling orders | Streaming updates |
| Chef tasting before serving | Human-in-the-loop |
| Rush hour pause | Interrupt points |

The kitchen doesn't process orders linearly. It:
- Handles multiple orders concurrently
- Moves dishes between stations based on readiness
- Pauses when chef approval is needed
- Survives shift changes (state persisted on tickets)

---

## Part 1: Advanced State Management

### State Design Principles

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from datetime import datetime
from pydantic import BaseModel, Field

# Principle 1: Use TypedDict for simple state
class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]
    status: str

# Principle 2: Use Pydantic for complex, validated state
class DocumentState(BaseModel):
    """State for document processing workflow."""

    # Metadata
    document_id: str = Field(description="Unique document identifier")
    created_at: datetime = Field(default_factory=datetime.now)

    # Processing state
    content: str = Field(default="")
    extracted_entities: list[dict] = Field(default_factory=list)
    classification: Optional[str] = Field(default=None)

    # Workflow control
    current_stage: str = Field(default="intake")
    requires_review: bool = Field(default=False)
    error_message: Optional[str] = Field(default=None)

    # Audit trail
    processing_history: list[dict] = Field(default_factory=list)
```

### Custom Reducers

Reducers determine how state updates merge. Let's build custom ones:

```python
# code/04_custom_reducers.py

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# Custom reducer: Keep the higher value
def max_reducer(current: int, update: int) -> int:
    """Keep the maximum value."""
    return max(current, update)

# Custom reducer: Merge dictionaries
def merge_dicts(current: dict, update: dict) -> dict:
    """Deep merge two dictionaries."""
    result = current.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

# Custom reducer: Append unique items
def append_unique(current: list, update: list) -> list:
    """Append only unique items."""
    seen = set(current)
    return current + [item for item in update if item not in seen]

# Custom reducer: Running average
class RunningAverage:
    """Reducer that maintains a running average."""
    def __init__(self):
        self.count = 0
        self.total = 0

    def __call__(self, current: float, update: float) -> float:
        self.count += 1
        self.total += update
        return self.total / self.count

# Apply to state
class AnalyticsState(TypedDict):
    max_confidence: Annotated[int, max_reducer]
    metadata: Annotated[dict, merge_dicts]
    unique_topics: Annotated[list, append_unique]
    # Note: Running average would need special handling
```

### State Channels and Aggregation

```python
from typing import Sequence
from langchain_core.messages import BaseMessage

# Messages channel with custom aggregation
def message_aggregator(
    left: Sequence[BaseMessage],
    right: Sequence[BaseMessage]
) -> Sequence[BaseMessage]:
    """
    Aggregate messages with deduplication and ordering.
    """
    seen_ids = set()
    result = []

    for msg in list(left) + list(right):
        msg_id = getattr(msg, 'id', None) or hash(msg.content)
        if msg_id not in seen_ids:
            seen_ids.add(msg_id)
            result.append(msg)

    return sorted(result, key=lambda m: getattr(m, 'timestamp', 0))

class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], message_aggregator]
```

---

## Part 2: Checkpointing and Persistence

### Why Checkpointing Matters

```
Without Checkpointing:
  Server crash during long operation → All progress lost

With Checkpointing:
  Server crash → Resume from last checkpoint → Continue processing
```

### Memory Checkpointer (Development)

```python
# code/05_checkpointing.py

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

llm = ChatOpenAI(model="gpt-4o-mini")

def chat(state: ConversationState) -> dict:
    """Process a conversation turn."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def summarize(state: ConversationState) -> dict:
    """Summarize if conversation gets long."""
    messages = state["messages"]
    if len(messages) > 10:
        summary = llm.invoke(f"Summarize this conversation:\n{messages}")
        return {"summary": summary.content}
    return {}

# Build graph
builder = StateGraph(ConversationState)
builder.add_node("chat", chat)
builder.add_node("summarize", summarize)
builder.add_edge(START, "chat")
builder.add_edge("chat", "summarize")
builder.add_edge("summarize", END)

# Create checkpointer
memory = MemorySaver()

# Compile with checkpointer
graph = builder.compile(checkpointer=memory)

# Use with thread_id for conversation persistence
config = {"configurable": {"thread_id": "user-123-conversation"}}

# First message
result1 = graph.invoke(
    {"messages": [HumanMessage(content="Hi, my name is Alice")], "summary": ""},
    config=config
)
print(f"Response 1: {result1['messages'][-1].content}")

# Second message - remembers context!
result2 = graph.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config
)
print(f"Response 2: {result2['messages'][-1].content}")

# Check conversation state
state = graph.get_state(config)
print(f"Total messages: {len(state.values['messages'])}")
```

### SQLite Checkpointer (Production)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Create SQLite checkpointer for persistence
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
db_checkpointer = SqliteSaver(conn)

# Compile with persistent checkpointer
graph = builder.compile(checkpointer=db_checkpointer)

# Now conversations survive server restarts!
```

### PostgreSQL Checkpointer (Scale)

```python
from langgraph.checkpoint.postgres import PostgresSaver

# For production at scale
postgres_checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)

graph = builder.compile(checkpointer=postgres_checkpointer)
```

### Checkpoint Operations

```python
# Get current state
state = graph.get_state(config)
print(f"Current node: {state.next}")
print(f"Values: {state.values}")

# Get state history
for state in graph.get_state_history(config):
    print(f"Step: {state.config['configurable']['checkpoint_id']}")
    print(f"Node: {state.next}")

# Replay from specific checkpoint
old_config = {"configurable": {
    "thread_id": "user-123",
    "checkpoint_id": "specific-checkpoint-id"
}}
result = graph.invoke({"messages": [...]}, old_config)

# Update state manually
graph.update_state(
    config,
    {"messages": [AIMessage(content="Manually injected message")]}
)
```

---

## Part 3: Human-in-the-Loop Patterns

### Why Human-in-the-Loop?

**Emergence Insight:** Even the smartest agents need human oversight for:
- High-stakes decisions
- Ambiguous situations
- Quality assurance
- Legal/compliance requirements
- Learning from corrections

### Pattern 1: Approval Gates

```python
# code/06_human_approval.py

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    draft_response: str
    approved: bool
    feedback: str

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_response(state: ApprovalState) -> dict:
    """Generate a draft response."""
    messages = state["messages"]
    response = llm.invoke([
        SystemMessage(content="Generate a professional response."),
        *messages
    ])
    return {"draft_response": response.content}

def check_approval(state: ApprovalState) -> Literal["send", "revise"]:
    """Route based on approval status."""
    if state.get("approved", False):
        return "send"
    return "revise"

def await_approval(state: ApprovalState) -> dict:
    """This node is where we pause for human input."""
    # The graph will interrupt here
    # Human provides feedback via graph.update_state()
    return {}

def revise_response(state: ApprovalState) -> dict:
    """Revise based on feedback."""
    feedback = state.get("feedback", "")
    draft = state["draft_response"]

    response = llm.invoke([
        SystemMessage(content="Revise the draft based on feedback."),
        HumanMessage(content=f"Draft: {draft}\n\nFeedback: {feedback}")
    ])

    return {
        "draft_response": response.content,
        "approved": False,  # Reset for next review
        "feedback": ""
    }

def send_response(state: ApprovalState) -> dict:
    """Send the approved response."""
    return {
        "messages": [AIMessage(content=f"[SENT] {state['draft_response']}")]
    }

# Build graph with interrupt
builder = StateGraph(ApprovalState)

builder.add_node("generate", generate_response)
builder.add_node("await_approval", await_approval)
builder.add_node("revise", revise_response)
builder.add_node("send", send_response)

builder.add_edge(START, "generate")
builder.add_edge("generate", "await_approval")

builder.add_conditional_edges(
    "await_approval",
    check_approval,
    {"send": "send", "revise": "revise"}
)

builder.add_edge("revise", "await_approval")  # Back to approval
builder.add_edge("send", END)

# Compile with interrupt point
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["await_approval"]  # Pause BEFORE this node
)

# === Usage ===

config = {"configurable": {"thread_id": "approval-demo"}}

# Start the workflow
result = graph.invoke(
    {
        "messages": [HumanMessage(content="Write an email to decline a job offer")],
        "draft_response": "",
        "approved": False,
        "feedback": ""
    },
    config=config
)

# Get current state (paused at await_approval)
state = graph.get_state(config)
print(f"Draft: {state.values['draft_response']}")
print(f"Waiting at: {state.next}")

# Simulate human review - REJECT with feedback
graph.update_state(
    config,
    {"approved": False, "feedback": "Make it more grateful and less formal"}
)

# Continue execution
result = graph.invoke(None, config=config)

# Check new draft
state = graph.get_state(config)
print(f"Revised draft: {state.values['draft_response']}")

# Human approves this time
graph.update_state(config, {"approved": True})

# Final execution
result = graph.invoke(None, config=config)
print(f"Final: {result['messages'][-1].content}")
```

### Pattern 2: Human as a Tool

```python
from langchain_core.tools import tool
from typing import Optional

# Human input as a tool the agent can invoke
@tool
def ask_human(question: str) -> str:
    """
    Ask a human for input when you need clarification or approval.
    Use this when you're uncertain or need human judgment.

    Args:
        question: The specific question to ask the human
    """
    # In production, this would:
    # 1. Send notification to human
    # 2. Wait for response (via webhook, polling, etc.)
    # 3. Return the human's answer

    # For demo, we simulate
    print(f"\n[HUMAN INPUT REQUESTED]: {question}")
    return input("Human response: ")

# Add to agent's toolset
tools = [search, calculate, ask_human]
```

### Pattern 3: Escalation Workflow

```python
def should_escalate(state: State) -> Literal["continue", "escalate"]:
    """Determine if human escalation is needed."""
    conditions = [
        state.get("confidence", 1.0) < 0.7,
        state.get("sensitive_topic", False),
        state.get("high_value", False),
        "urgent" in state.get("flags", [])
    ]

    if any(conditions):
        return "escalate"
    return "continue"

def escalate_to_human(state: State) -> dict:
    """Prepare escalation package for human review."""
    return {
        "escalation": {
            "reason": state.get("escalation_reason", "Low confidence"),
            "context": state.get("messages", [])[-5:],  # Last 5 messages
            "suggested_action": state.get("draft_response", ""),
            "priority": "high" if state.get("urgent", False) else "normal"
        }
    }
```

---

## Part 4: Streaming and Real-time Updates

### Why Streaming Matters

**User Experience Principle:** Perceived latency is as important as actual latency.

```
Without Streaming:
  [Wait 10 seconds] → [Full response appears]
  User experience: "Is it working? This is slow..."

With Streaming:
  [Immediate] → [Token] → [Token] → [Token] → ...
  User experience: "It's responding! I can see it thinking..."
```

### Basic Streaming

```python
# code/07_streaming.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

def chat(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()

# Stream graph events
print("Streaming graph events:")
for event in graph.stream(
    {"messages": [HumanMessage(content="Explain quantum computing")]},
    stream_mode="values"
):
    if "messages" in event:
        print(f"Messages count: {len(event['messages'])}")

# Stream with updates mode (more detailed)
print("\nStreaming with updates:")
for event in graph.stream(
    {"messages": [HumanMessage(content="Explain AI")]},
    stream_mode="updates"
):
    for node, updates in event.items():
        print(f"Node '{node}' produced updates: {list(updates.keys())}")
```

### Streaming LLM Tokens

```python
# code/08_token_streaming.py

async def stream_tokens():
    """Stream individual LLM tokens."""

    config = {"configurable": {"thread_id": "stream-demo"}}

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="Write a haiku about coding")]},
        config=config,
        version="v2"
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # Individual token from LLM
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)

        elif kind == "on_chain_start":
            print(f"\n[Starting: {event['name']}]")

        elif kind == "on_chain_end":
            print(f"\n[Completed: {event['name']}]")

# Run with asyncio
import asyncio
asyncio.run(stream_tokens())
```

### Custom Stream Events

```python
from langchain_core.callbacks import dispatch_custom_event

def processing_node(state: State) -> dict:
    """Node that emits custom stream events."""

    # Emit progress event
    dispatch_custom_event(
        "progress",
        {"step": "starting", "percentage": 0}
    )

    # Do work...
    result = process_data(state["data"])

    dispatch_custom_event(
        "progress",
        {"step": "processing", "percentage": 50}
    )

    # More work...
    final = finalize(result)

    dispatch_custom_event(
        "progress",
        {"step": "complete", "percentage": 100}
    )

    return {"result": final}

# Listen for custom events
async for event in graph.astream_events(...):
    if event["event"] == "on_custom_event":
        if event["name"] == "progress":
            data = event["data"]
            print(f"Progress: {data['step']} ({data['percentage']}%)")
```

---

## Part 5: Interrupts and Breakpoints

### Understanding Interrupts

Interrupts let you pause execution at specific points:

```python
# Interrupt BEFORE a node runs
graph = builder.compile(
    interrupt_before=["sensitive_operation"]
)

# Interrupt AFTER a node runs
graph = builder.compile(
    interrupt_after=["generate_draft"]
)

# Multiple interrupt points
graph = builder.compile(
    interrupt_before=["human_review"],
    interrupt_after=["generate", "analyze"]
)
```

### Breakpoint Pattern

```python
# code/09_breakpoints.py

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class DebugState(TypedDict):
    value: int
    history: list
    breakpoint_hit: bool

def step_a(state: DebugState) -> dict:
    new_value = state["value"] * 2
    return {
        "value": new_value,
        "history": state["history"] + [f"step_a: {state['value']} -> {new_value}"]
    }

def step_b(state: DebugState) -> dict:
    new_value = state["value"] + 10
    return {
        "value": new_value,
        "history": state["history"] + [f"step_b: {state['value']} -> {new_value}"]
    }

def step_c(state: DebugState) -> dict:
    new_value = state["value"] // 3
    return {
        "value": new_value,
        "history": state["history"] + [f"step_c: {state['value']} -> {new_value}"]
    }

# Build graph
builder = StateGraph(DebugState)
builder.add_node("a", step_a)
builder.add_node("b", step_b)
builder.add_node("c", step_c)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", END)

# Compile with breakpoint after step_b
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_after=["b"]  # Pause after b, before c
)

config = {"configurable": {"thread_id": "debug-session"}}

# Run until breakpoint
result = graph.invoke(
    {"value": 5, "history": [], "breakpoint_hit": False},
    config=config
)

# Check state at breakpoint
state = graph.get_state(config)
print(f"Paused at: {state.next}")
print(f"Current value: {state.values['value']}")
print(f"History: {state.values['history']}")

# Optionally modify state before continuing
graph.update_state(config, {"value": 100})  # Inject different value

# Continue execution
final = graph.invoke(None, config=config)
print(f"Final value: {final['value']}")
print(f"Full history: {final['history']}")
```

### Dynamic Interrupts

```python
from langgraph.types import interrupt

def node_with_dynamic_interrupt(state: State) -> dict:
    """Node that can decide at runtime whether to interrupt."""

    result = analyze(state["input"])

    if result["needs_human_review"]:
        # Dynamically interrupt and wait for human
        human_input = interrupt({
            "question": "Please review this result",
            "data": result
        })
        # Execution pauses here until human responds
        result = incorporate_feedback(result, human_input)

    return {"result": result}
```

---

## Putting It All Together: Event-Driven Agent

```python
# code/10_event_driven_agent.py

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import json

# =============================================================================
# STATE
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_actions: list
    completed_actions: list
    requires_approval: bool
    approval_granted: bool
    error: str | None

# =============================================================================
# TOOLS (representing external events)
# =============================================================================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Requires human approval for external recipients."""
    return f"Email sent to {to}: {subject}"

@tool
def update_database(table: str, data: dict) -> str:
    """Update a database record. Requires approval for production tables."""
    return f"Updated {table} with {json.dumps(data)}"

@tool
def search_documents(query: str) -> str:
    """Search internal documents. No approval needed."""
    return f"Found 3 documents matching '{query}'"

tools = [send_email, update_database, search_documents]
tool_map = {t.name: t for t in tools}

# Tools requiring approval
SENSITIVE_TOOLS = {"send_email", "update_database"}

# =============================================================================
# NODES
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def plan_actions(state: AgentState) -> dict:
    """Plan what actions to take."""
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Plan actions to help the user."),
        *state["messages"]
    ])

    pending = []
    requires_approval = False

    if response.tool_calls:
        for call in response.tool_calls:
            pending.append(call)
            if call["name"] in SENSITIVE_TOOLS:
                requires_approval = True

    return {
        "messages": [response],
        "pending_actions": pending,
        "requires_approval": requires_approval
    }

def execute_actions(state: AgentState) -> dict:
    """Execute approved actions."""
    results = []

    for action in state["pending_actions"]:
        tool_name = action["name"]
        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(action["args"])
            results.append({"action": tool_name, "result": result})

    return {
        "completed_actions": state["completed_actions"] + results,
        "pending_actions": []
    }

def synthesize_results(state: AgentState) -> dict:
    """Create final response based on action results."""
    completed = state["completed_actions"]

    response = llm.invoke([
        SystemMessage(content="Summarize what was accomplished."),
        HumanMessage(content=f"Completed actions: {json.dumps(completed)}")
    ])

    return {"messages": [response]}

def handle_rejection(state: AgentState) -> dict:
    """Handle when actions are rejected."""
    return {
        "messages": [AIMessage(content="I understand. The requested actions have been cancelled. Is there anything else I can help with?")],
        "pending_actions": []
    }

# =============================================================================
# ROUTING
# =============================================================================

def route_after_plan(state: AgentState) -> Literal["await_approval", "execute", "respond"]:
    """Route based on planned actions."""
    if not state["pending_actions"]:
        return "respond"
    if state["requires_approval"]:
        return "await_approval"
    return "execute"

def route_after_approval(state: AgentState) -> Literal["execute", "reject"]:
    """Route based on approval decision."""
    if state.get("approval_granted", False):
        return "execute"
    return "reject"

# =============================================================================
# BUILD GRAPH
# =============================================================================

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("plan", plan_actions)
builder.add_node("await_approval", lambda x: {})  # Checkpoint node
builder.add_node("execute", execute_actions)
builder.add_node("synthesize", synthesize_results)
builder.add_node("reject", handle_rejection)

# Add edges
builder.add_edge(START, "plan")

builder.add_conditional_edges(
    "plan",
    route_after_plan,
    {
        "await_approval": "await_approval",
        "execute": "execute",
        "respond": "synthesize"
    }
)

builder.add_conditional_edges(
    "await_approval",
    route_after_approval,
    {
        "execute": "execute",
        "reject": "reject"
    }
)

builder.add_edge("execute", "synthesize")
builder.add_edge("synthesize", END)
builder.add_edge("reject", END)

# Compile with checkpointing and interrupt
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["await_approval"]
)

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "event-driven-demo"}}

    # User request requiring approval
    result = graph.invoke({
        "messages": [HumanMessage(content="Send an email to bob@example.com about the project update")],
        "pending_actions": [],
        "completed_actions": [],
        "requires_approval": False,
        "approval_granted": False,
        "error": None
    }, config=config)

    # Check if waiting for approval
    state = graph.get_state(config)
    if state.next == ("await_approval",):
        print("=== APPROVAL REQUIRED ===")
        print(f"Pending actions: {state.values['pending_actions']}")

        # Simulate human approval
        approve = input("Approve? (y/n): ").lower() == 'y'
        graph.update_state(config, {"approval_granted": approve})

        # Continue execution
        result = graph.invoke(None, config=config)

    print("\n=== FINAL RESULT ===")
    print(result["messages"][-1].content)
```

---

## Summary: Event-Driven Patterns

| Pattern | Use Case | Key Feature |
|---------|----------|-------------|
| Checkpointing | Persistence, recovery | `MemorySaver`, `SqliteSaver` |
| Human Approval | High-stakes decisions | `interrupt_before` + `update_state` |
| Streaming | Real-time UX | `stream()`, `astream_events()` |
| Dynamic Interrupt | Runtime decisions | `interrupt()` function |
| State Reducers | Concurrent updates | Annotated types |

---

## Key Takeaways

1. **Event-driven architecture** enables persistence, resilience, and human-in-the-loop
2. **Checkpointing** allows workflows to survive restarts and enables replay
3. **Interrupts** pause execution at defined points for human input
4. **Streaming** provides real-time feedback for better user experience
5. **Custom reducers** handle complex state merge scenarios
6. **The patterns compose**: checkpointing + interrupts + streaming = production-ready agents

---

## Exercises

1. **Build a moderation pipeline**: Create a workflow where all external communications require approval

2. **Add timeout handling**: Implement a pattern where pending approvals expire after a time limit

3. **Multi-approver workflow**: Design a system requiring approval from multiple stakeholders

4. **Stream to WebSocket**: Connect streaming output to a WebSocket for real-time UI updates

---

## Next Steps

In [Module 3.3: Real-world Pipeline Examples](03_real_world_pipelines.md), we'll build production-grade:
- RAG pipelines with quality checks
- Multi-agent orchestration
- Error recovery patterns

---

*"The best systems are not those that never fail, but those that recover gracefully."*
