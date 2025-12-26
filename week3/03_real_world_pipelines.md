# Module 3.3: Real-world Pipeline Examples

## What You'll Learn
- Building production-grade RAG pipelines with quality gates
- Multi-agent orchestration patterns
- Error handling and recovery strategies
- Testing and debugging LangGraph workflows
- Performance optimization techniques

---

## First Principles: What Makes a Pipeline "Production-Ready"?

Let's reason from first principles about what separates demos from production systems.

**First Principle #1:** Production systems must handle failure gracefully.

```
Demo: "Happy path" works
Production: Every path works (or fails informatively)
```

**First Principle #2:** Production systems must be observable.

```
Demo: print() statements
Production: Structured logging, tracing, metrics
```

**First Principle #3:** Production systems must be testable.

```
Demo: Manual verification
Production: Automated tests, CI/CD integration
```

**First Principle #4:** Production systems must be maintainable.

```
Demo: Single file, clever code
Production: Modular, documented, boring code
```

---

## Analogical Thinking: The Industrial Pipeline

Think of a production LangGraph workflow like an **oil refinery**:

| Oil Refinery | LangGraph Pipeline |
|--------------|-------------------|
| Crude oil input | Raw user query |
| Distillation towers | Processing nodes |
| Quality sensors | Validation checks |
| Safety valves | Error handlers |
| Control room monitors | Observability |
| Maintenance schedules | Testing |
| Emergency procedures | Recovery patterns |
| Refined products | Quality responses |

A refinery doesn't stop when one sensor fails. It has redundancy, monitoring, and graceful degradation. Your pipelines should too.

---

## Part 1: Production RAG Pipeline

### The Complete RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION RAG PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Query          ┌─────────┐     ┌─────────┐     ┌─────────┐        │
│   ────────────► │ Understand│────►│ Retrieve │────►│ Validate │       │
│                  │  Query   │     │ Docs     │     │ Results │       │
│                  └─────────┘     └─────────┘     └────┬────┘        │
│                                                       │              │
│                        ┌──────────────────────────────┤              │
│                        │                              │              │
│                        ▼                              ▼              │
│                  ┌─────────┐                    ┌─────────┐         │
│                  │ Re-query │                    │ Generate │         │
│                  │(if needed)│                   │ Answer  │         │
│                  └─────────┘                    └────┬────┘         │
│                                                      │               │
│                                                      ▼               │
│                                                ┌─────────┐          │
│                                                │ Verify  │          │
│                                                │ Answer  │          │
│                                                └────┬────┘          │
│                                                     │                │
│                         ┌───────────────────────────┤                │
│                         │                           │                │
│                         ▼                           ▼                │
│                   ┌─────────┐                 ┌─────────┐           │
│                   │ Refine  │                 │  Return │           │
│                   │(if poor) │                │  Answer │           │
│                   └─────────┘                 └─────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# code/11_production_rag.py

from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
import json
from datetime import datetime

# =============================================================================
# STATE DEFINITION
# =============================================================================

class RetrievalResult(BaseModel):
    """Structured retrieval result."""
    document_id: str
    content: str
    relevance_score: float
    source: str

class QualityMetrics(BaseModel):
    """Quality assessment metrics."""
    relevance: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    coherence: float = Field(ge=0, le=1)
    factuality: float = Field(ge=0, le=1)

    @property
    def overall(self) -> float:
        return (self.relevance + self.completeness + self.coherence + self.factuality) / 4

class RAGState(TypedDict):
    """State for the RAG pipeline."""
    # Input
    original_query: str
    messages: Annotated[list, add_messages]

    # Query understanding
    query_intent: str
    query_entities: list[str]
    reformulated_queries: list[str]

    # Retrieval
    retrieved_docs: list[dict]
    retrieval_scores: list[float]

    # Generation
    draft_answer: str
    final_answer: str
    citations: list[dict]

    # Quality control
    quality_metrics: Optional[dict]
    iteration_count: int

    # Observability
    trace_id: str
    step_timings: list[dict]
    errors: list[dict]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log_step(state: RAGState, step_name: str, duration_ms: float) -> dict:
    """Log step timing for observability."""
    return {
        "step_timings": state.get("step_timings", []) + [{
            "step": step_name,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }]
    }

def log_error(state: RAGState, step_name: str, error: str) -> dict:
    """Log error for debugging."""
    return {
        "errors": state.get("errors", []) + [{
            "step": step_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }]
    }

# =============================================================================
# MOCK VECTOR STORE (replace with real one in production)
# =============================================================================

# In production, initialize with real embeddings and documents
def get_vector_store():
    """Get or create vector store."""
    # Mock documents for demo
    docs = [
        Document(page_content="LangGraph is a library for building stateful, multi-actor applications.", metadata={"source": "docs"}),
        Document(page_content="LangGraph extends LangChain with graph-based workflows.", metadata={"source": "docs"}),
        Document(page_content="Nodes in LangGraph are functions that transform state.", metadata={"source": "tutorial"}),
        Document(page_content="Edges define the flow between nodes in LangGraph.", metadata={"source": "tutorial"}),
        Document(page_content="Checkpointing enables persistence in LangGraph workflows.", metadata={"source": "advanced"}),
    ]

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# =============================================================================
# NODES
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def understand_query(state: RAGState) -> dict:
    """Understand and reformulate the query."""
    import time
    start = time.time()

    query = state["original_query"]

    # Extract intent and entities
    analysis = llm.invoke([
        SystemMessage(content="""Analyze this query and respond with JSON:
{
    "intent": "question|command|clarification",
    "entities": ["entity1", "entity2"],
    "reformulated_queries": ["query1", "query2", "query3"]
}

Generate 2-3 reformulated queries that might retrieve different relevant information."""),
        HumanMessage(content=query)
    ])

    try:
        result = json.loads(analysis.content)
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        result = {
            "intent": "question",
            "entities": [],
            "reformulated_queries": [query]
        }

    duration = (time.time() - start) * 1000

    return {
        "query_intent": result.get("intent", "question"),
        "query_entities": result.get("entities", []),
        "reformulated_queries": result.get("reformulated_queries", [query]),
        **log_step(state, "understand_query", duration)
    }

def retrieve_documents(state: RAGState) -> dict:
    """Retrieve relevant documents."""
    import time
    start = time.time()

    queries = state["reformulated_queries"]
    if not queries:
        queries = [state["original_query"]]

    # Get vector store
    try:
        vectorstore = get_vector_store()
    except Exception as e:
        return {
            "retrieved_docs": [],
            "retrieval_scores": [],
            **log_error(state, "retrieve_documents", str(e))
        }

    # Retrieve for each query
    all_docs = []
    all_scores = []

    for query in queries[:3]:  # Limit to 3 queries
        results = vectorstore.similarity_search_with_score(query, k=3)
        for doc, score in results:
            all_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score)
            })
            all_scores.append(float(score))

    # Deduplicate by content
    seen = set()
    unique_docs = []
    unique_scores = []
    for doc, score in zip(all_docs, all_scores):
        if doc["content"] not in seen:
            seen.add(doc["content"])
            unique_docs.append(doc)
            unique_scores.append(score)

    duration = (time.time() - start) * 1000

    return {
        "retrieved_docs": unique_docs[:5],  # Top 5
        "retrieval_scores": unique_scores[:5],
        **log_step(state, "retrieve_documents", duration)
    }

def validate_retrieval(state: RAGState) -> dict:
    """Validate retrieval quality."""
    docs = state["retrieved_docs"]
    query = state["original_query"]

    if not docs:
        return {"quality_metrics": {"retrieval_quality": 0.0}}

    # Check if documents are relevant
    validation = llm.invoke([
        SystemMessage(content="""Rate the relevance of these documents to the query.
Respond with a single number from 0.0 to 1.0."""),
        HumanMessage(content=f"Query: {query}\n\nDocuments:\n" +
                    "\n---\n".join([d["content"] for d in docs]))
    ])

    try:
        relevance = float(validation.content.strip())
    except ValueError:
        relevance = 0.5

    return {
        "quality_metrics": {
            **(state.get("quality_metrics") or {}),
            "retrieval_quality": relevance
        }
    }

def should_requery(state: RAGState) -> Literal["generate", "requery", "no_docs"]:
    """Decide if we need to requery."""
    docs = state["retrieved_docs"]
    metrics = state.get("quality_metrics", {})
    iteration = state.get("iteration_count", 0)

    if not docs:
        if iteration >= 2:
            return "no_docs"
        return "requery"

    retrieval_quality = metrics.get("retrieval_quality", 0)
    if retrieval_quality < 0.5 and iteration < 2:
        return "requery"

    return "generate"

def requery(state: RAGState) -> dict:
    """Generate alternative queries."""
    original = state["original_query"]
    tried = state.get("reformulated_queries", [])

    response = llm.invoke([
        SystemMessage(content="Generate 3 alternative search queries to find relevant information."),
        HumanMessage(content=f"Original: {original}\nAlready tried: {tried}\n\nNew queries (one per line):")
    ])

    new_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    return {
        "reformulated_queries": new_queries[:3],
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def generate_answer(state: RAGState) -> dict:
    """Generate answer from retrieved documents."""
    import time
    start = time.time()

    docs = state["retrieved_docs"]
    query = state["original_query"]

    context = "\n\n".join([
        f"[Source: {d['source']}]\n{d['content']}"
        for d in docs
    ])

    response = llm.invoke([
        SystemMessage(content="""Answer the question based on the provided context.
Include citations in [Source: X] format.
If the context doesn't contain the answer, say so clearly."""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ])

    # Extract citations
    citations = []
    for doc in docs:
        if doc["source"] in response.content:
            citations.append({
                "source": doc["source"],
                "content_snippet": doc["content"][:100]
            })

    duration = (time.time() - start) * 1000

    return {
        "draft_answer": response.content,
        "citations": citations,
        **log_step(state, "generate_answer", duration)
    }

def verify_answer(state: RAGState) -> dict:
    """Verify the generated answer."""
    answer = state["draft_answer"]
    docs = state["retrieved_docs"]
    query = state["original_query"]

    # Verify factuality against sources
    verification = llm.invoke([
        SystemMessage(content="""Evaluate this answer for:
1. Relevance (0-1): Does it answer the question?
2. Completeness (0-1): Does it cover all aspects?
3. Coherence (0-1): Is it well-structured?
4. Factuality (0-1): Is it supported by the sources?

Respond with JSON: {"relevance": X, "completeness": X, "coherence": X, "factuality": X}"""),
        HumanMessage(content=f"Question: {query}\n\nAnswer: {answer}\n\nSources: {[d['content'] for d in docs]}")
    ])

    try:
        metrics = json.loads(verification.content)
        quality = QualityMetrics(**metrics)
    except (json.JSONDecodeError, ValueError):
        quality = QualityMetrics(relevance=0.7, completeness=0.7, coherence=0.7, factuality=0.7)

    return {
        "quality_metrics": {
            **(state.get("quality_metrics") or {}),
            **quality.model_dump(),
            "overall": quality.overall
        }
    }

def should_refine(state: RAGState) -> Literal["refine", "return"]:
    """Decide if answer needs refinement."""
    metrics = state.get("quality_metrics", {})
    overall = metrics.get("overall", 0)
    iteration = state.get("iteration_count", 0)

    if overall < 0.7 and iteration < 2:
        return "refine"
    return "return"

def refine_answer(state: RAGState) -> dict:
    """Refine the answer based on quality feedback."""
    answer = state["draft_answer"]
    metrics = state.get("quality_metrics", {})

    # Identify weak areas
    weak_areas = []
    if metrics.get("relevance", 1) < 0.7:
        weak_areas.append("relevance to the question")
    if metrics.get("completeness", 1) < 0.7:
        weak_areas.append("completeness")
    if metrics.get("coherence", 1) < 0.7:
        weak_areas.append("structure and clarity")
    if metrics.get("factuality", 1) < 0.7:
        weak_areas.append("factual accuracy")

    improvement_focus = ", ".join(weak_areas) if weak_areas else "overall quality"

    response = llm.invoke([
        SystemMessage(content=f"Improve this answer, focusing on: {improvement_focus}"),
        HumanMessage(content=f"Original answer:\n{answer}\n\nImproved version:")
    ])

    return {
        "draft_answer": response.content,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def format_response(state: RAGState) -> dict:
    """Format the final response."""
    answer = state["draft_answer"]
    citations = state.get("citations", [])
    metrics = state.get("quality_metrics", {})

    # Build final response with metadata
    final = f"{answer}\n\n---\nQuality Score: {metrics.get('overall', 'N/A'):.2f}"

    if citations:
        final += "\n\nSources:\n"
        for c in citations:
            final += f"- {c['source']}\n"

    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)]
    }

def handle_no_docs(state: RAGState) -> dict:
    """Handle case when no relevant documents found."""
    return {
        "final_answer": "I couldn't find relevant information to answer your question. Could you rephrase or provide more context?",
        "messages": [AIMessage(content="I couldn't find relevant information to answer your question. Could you rephrase or provide more context?")]
    }

# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_rag_pipeline():
    """Build the production RAG pipeline."""
    builder = StateGraph(RAGState)

    # Add nodes
    builder.add_node("understand", understand_query)
    builder.add_node("retrieve", retrieve_documents)
    builder.add_node("validate_retrieval", validate_retrieval)
    builder.add_node("requery", requery)
    builder.add_node("generate", generate_answer)
    builder.add_node("verify", verify_answer)
    builder.add_node("refine", refine_answer)
    builder.add_node("format", format_response)
    builder.add_node("no_docs", handle_no_docs)

    # Define flow
    builder.add_edge(START, "understand")
    builder.add_edge("understand", "retrieve")
    builder.add_edge("retrieve", "validate_retrieval")

    builder.add_conditional_edges(
        "validate_retrieval",
        should_requery,
        {
            "generate": "generate",
            "requery": "requery",
            "no_docs": "no_docs"
        }
    )

    builder.add_edge("requery", "retrieve")  # Cycle back

    builder.add_edge("generate", "verify")

    builder.add_conditional_edges(
        "verify",
        should_refine,
        {
            "refine": "refine",
            "return": "format"
        }
    )

    builder.add_edge("refine", "verify")  # Cycle back

    builder.add_edge("format", END)
    builder.add_edge("no_docs", END)

    return builder.compile()

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    import uuid

    pipeline = build_rag_pipeline()

    # Test query
    result = pipeline.invoke({
        "original_query": "How do nodes work in LangGraph?",
        "messages": [],
        "query_intent": "",
        "query_entities": [],
        "reformulated_queries": [],
        "retrieved_docs": [],
        "retrieval_scores": [],
        "draft_answer": "",
        "final_answer": "",
        "citations": [],
        "quality_metrics": None,
        "iteration_count": 0,
        "trace_id": str(uuid.uuid4()),
        "step_timings": [],
        "errors": []
    })

    print("=" * 60)
    print("RAG PIPELINE RESULT")
    print("=" * 60)
    print(f"\nAnswer:\n{result['final_answer']}")
    print(f"\nTimings:")
    for timing in result.get("step_timings", []):
        print(f"  {timing['step']}: {timing['duration_ms']:.1f}ms")
    if result.get("errors"):
        print(f"\nErrors: {result['errors']}")
```

---

## Part 2: Multi-Agent Orchestration

### The Supervisor Pattern

Multiple specialized agents coordinated by a supervisor:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR PATTERN                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌──────────────┐                           │
│                      │  SUPERVISOR  │                           │
│                      │   (Router)   │                           │
│                      └──────┬───────┘                           │
│                             │                                    │
│              ┌──────────────┼──────────────┐                    │
│              │              │              │                     │
│              ▼              ▼              ▼                     │
│       ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│       │ RESEARCH │   │  WRITER  │   │  CRITIC  │               │
│       │  AGENT   │   │  AGENT   │   │  AGENT   │               │
│       └──────────┘   └──────────┘   └──────────┘               │
│              │              │              │                     │
│              └──────────────┼──────────────┘                    │
│                             │                                    │
│                             ▼                                    │
│                      ┌──────────────┐                           │
│                      │    OUTPUT    │                           │
│                      └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# code/12_multi_agent.py

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

# =============================================================================
# STATE
# =============================================================================

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    research_notes: str
    draft: str
    critique: str
    final_output: str
    next_agent: str
    iteration: int

# =============================================================================
# SPECIALIZED AGENTS
# =============================================================================

research_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def research_agent(state: MultiAgentState) -> dict:
    """Research agent: Gathers and organizes information."""
    task = state["task"]
    existing_notes = state.get("research_notes", "")
    critique = state.get("critique", "")

    prompt = f"Task: {task}"
    if existing_notes:
        prompt += f"\n\nPrevious research:\n{existing_notes}"
    if critique:
        prompt += f"\n\nFeedback to address:\n{critique}"

    response = research_llm.invoke([
        SystemMessage(content="""You are a research specialist. Your job is to:
1. Gather relevant information and facts
2. Organize findings clearly
3. Note any gaps or uncertainties

Provide well-structured research notes."""),
        HumanMessage(content=prompt)
    ])

    return {
        "research_notes": response.content,
        "messages": [AIMessage(content=f"[RESEARCH] {response.content[:200]}...")]
    }

def writer_agent(state: MultiAgentState) -> dict:
    """Writer agent: Creates polished content."""
    task = state["task"]
    research = state["research_notes"]
    critique = state.get("critique", "")
    previous_draft = state.get("draft", "")

    prompt = f"Task: {task}\n\nResearch:\n{research}"
    if previous_draft and critique:
        prompt += f"\n\nPrevious draft:\n{previous_draft}\n\nFeedback:\n{critique}"

    response = writer_llm.invoke([
        SystemMessage(content="""You are a skilled writer. Your job is to:
1. Transform research into engaging, clear content
2. Ensure proper structure and flow
3. Address any feedback from previous iterations

Write professional, polished content."""),
        HumanMessage(content=prompt)
    ])

    return {
        "draft": response.content,
        "messages": [AIMessage(content=f"[WRITER] Draft created ({len(response.content)} chars)")]
    }

def critic_agent(state: MultiAgentState) -> dict:
    """Critic agent: Reviews and provides feedback."""
    task = state["task"]
    draft = state["draft"]
    research = state["research_notes"]

    response = critic_llm.invoke([
        SystemMessage(content="""You are a critical reviewer. Evaluate the draft for:
1. Accuracy (matches research?)
2. Completeness (addresses the task?)
3. Quality (well-written?)
4. Improvements (specific suggestions)

If the draft is excellent, respond with: APPROVED
Otherwise, provide constructive feedback."""),
        HumanMessage(content=f"Task: {task}\n\nResearch:\n{research}\n\nDraft:\n{draft}")
    ])

    critique = response.content
    approved = "APPROVED" in critique.upper()

    return {
        "critique": critique if not approved else "",
        "messages": [AIMessage(content=f"[CRITIC] {'APPROVED' if approved else 'Needs revision'}")],
        "next_agent": "output" if approved else "supervisor"
    }

def supervisor_agent(state: MultiAgentState) -> dict:
    """Supervisor: Decides which agent should work next."""
    task = state["task"]
    research = state.get("research_notes", "")
    draft = state.get("draft", "")
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)

    # Build context
    context = f"Task: {task}\n"
    context += f"Research completed: {'Yes' if research else 'No'}\n"
    context += f"Draft created: {'Yes' if draft else 'No'}\n"
    context += f"Latest critique: {critique[:200] if critique else 'None'}\n"
    context += f"Iteration: {iteration}"

    response = supervisor_llm.invoke([
        SystemMessage(content="""You are a project supervisor coordinating agents.

Available agents:
- RESEARCH: Gathers information (use when more research needed)
- WRITER: Creates content (use when research is ready)
- OUTPUT: Finalize (only when work is complete)

Based on the current state, which agent should work next?
Respond with just the agent name: RESEARCH, WRITER, or OUTPUT"""),
        HumanMessage(content=context)
    ])

    next_agent = response.content.strip().upper()
    if next_agent not in ["RESEARCH", "WRITER", "OUTPUT"]:
        # Default logic
        if not research:
            next_agent = "RESEARCH"
        elif not draft:
            next_agent = "WRITER"
        elif critique:
            next_agent = "WRITER"
        else:
            next_agent = "OUTPUT"

    return {
        "next_agent": next_agent.lower(),
        "iteration": iteration + 1
    }

def output_agent(state: MultiAgentState) -> dict:
    """Output agent: Formats final result."""
    draft = state["draft"]

    return {
        "final_output": draft,
        "messages": [AIMessage(content=f"[OUTPUT] Final content ready:\n\n{draft}")]
    }

# =============================================================================
# ROUTING
# =============================================================================

def route_to_agent(state: MultiAgentState) -> Literal["research", "writer", "critic", "output"]:
    """Route to the appropriate agent."""
    next_agent = state.get("next_agent", "research")

    # Safety: max iterations
    if state.get("iteration", 0) > 5:
        return "output"

    return next_agent

def after_critic(state: MultiAgentState) -> Literal["supervisor", "output"]:
    """Route after critic review."""
    return state.get("next_agent", "supervisor")

# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_multi_agent_system():
    builder = StateGraph(MultiAgentState)

    # Add agent nodes
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("research", research_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("critic", critic_agent)
    builder.add_node("output", output_agent)

    # Start with supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor routes to agents
    builder.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "research": "research",
            "writer": "writer",
            "output": "output"
        }
    )

    # After research, go to supervisor
    builder.add_edge("research", "supervisor")

    # After writing, go to critic
    builder.add_edge("writer", "critic")

    # After critic, check if approved
    builder.add_conditional_edges(
        "critic",
        after_critic,
        {
            "supervisor": "supervisor",
            "output": "output"
        }
    )

    # Output ends the flow
    builder.add_edge("output", END)

    return builder.compile()

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    system = build_multi_agent_system()

    result = system.invoke({
        "messages": [],
        "task": "Write a brief explanation of how photosynthesis works for a high school student",
        "research_notes": "",
        "draft": "",
        "critique": "",
        "final_output": "",
        "next_agent": "research",
        "iteration": 0
    })

    print("=" * 60)
    print("MULTI-AGENT OUTPUT")
    print("=" * 60)
    print(f"\nFinal Output:\n{result['final_output']}")
    print(f"\nIterations: {result['iteration']}")

    print("\n--- Agent Activity Log ---")
    for msg in result["messages"]:
        if hasattr(msg, "content"):
            print(msg.content[:100])
```

---

## Part 3: Error Handling and Recovery

### The Error Handling Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                   ERROR HANDLING HIERARCHY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: NODE-LEVEL (Try/Catch in node functions)              │
│           ↓ escalate if can't handle                            │
│                                                                  │
│  Level 2: GRAPH-LEVEL (Error edges, fallback nodes)             │
│           ↓ escalate if critical                                │
│                                                                  │
│  Level 3: WORKFLOW-LEVEL (Retry policies, circuit breakers)     │
│           ↓ escalate if persistent                              │
│                                                                  │
│  Level 4: SYSTEM-LEVEL (Graceful degradation, human escalation) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# code/13_error_handling.py

from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import time
import random

# =============================================================================
# STATE WITH ERROR TRACKING
# =============================================================================

class RobustState(TypedDict):
    messages: Annotated[list, add_messages]
    data: dict
    result: Optional[str]
    error: Optional[dict]
    retry_count: int
    max_retries: int
    fallback_used: bool

# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

class RetryableError(Exception):
    """Error that can be retried."""
    pass

class FatalError(Exception):
    """Error that cannot be recovered from."""
    pass

def with_retry(func, max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retrying functions with exponential backoff."""
    def wrapper(state: RobustState) -> dict:
        last_error = None

        for attempt in range(max_retries):
            try:
                return func(state)
            except RetryableError as e:
                last_error = e
                wait_time = backoff_factor ** attempt
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            except FatalError as e:
                return {
                    "error": {
                        "type": "fatal",
                        "message": str(e),
                        "recoverable": False
                    }
                }

        return {
            "error": {
                "type": "retryable",
                "message": str(last_error),
                "attempts": max_retries,
                "recoverable": True
            }
        }
    return wrapper

# =============================================================================
# NODES WITH ERROR HANDLING
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini")

def process_with_potential_failure(state: RobustState) -> dict:
    """Node that might fail (simulated)."""
    # Simulate random failures for demo
    if random.random() < 0.3:  # 30% chance of failure
        raise RetryableError("Temporary API failure")

    response = llm.invoke([
        HumanMessage(content="Say 'Processing successful!'")
    ])

    return {
        "result": response.content,
        "error": None
    }

# Apply retry decorator
robust_process = with_retry(process_with_potential_failure, max_retries=3)

def validate_result(state: RobustState) -> dict:
    """Validate the result."""
    if state.get("error"):
        return {}  # Pass through error

    result = state.get("result", "")
    if not result or len(result) < 5:
        return {
            "error": {
                "type": "validation",
                "message": "Result too short or empty",
                "recoverable": True
            }
        }

    return {"error": None}

def fallback_process(state: RobustState) -> dict:
    """Fallback when main processing fails."""
    return {
        "result": "Fallback response: Unable to complete full processing. Here's a basic response.",
        "fallback_used": True,
        "error": None
    }

def handle_fatal_error(state: RobustState) -> dict:
    """Handle unrecoverable errors."""
    error = state.get("error", {})
    return {
        "messages": [AIMessage(content=f"I apologize, but I encountered an error I cannot recover from: {error.get('message', 'Unknown error')}. Please try again or contact support.")]
    }

def format_success(state: RobustState) -> dict:
    """Format successful result."""
    result = state["result"]
    fallback = state.get("fallback_used", False)

    content = result
    if fallback:
        content = f"[FALLBACK MODE]\n{result}"

    return {
        "messages": [AIMessage(content=content)]
    }

# =============================================================================
# ROUTING WITH ERROR AWARENESS
# =============================================================================

def check_error(state: RobustState) -> Literal["success", "retry", "fallback", "fatal"]:
    """Route based on error state."""
    error = state.get("error")

    if not error:
        return "success"

    if error.get("type") == "fatal":
        return "fatal"

    if error.get("recoverable", False):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if retry_count < max_retries:
            return "retry"
        else:
            return "fallback"

    return "fallback"

def increment_retry(state: RobustState) -> dict:
    """Increment retry counter."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "error": None  # Clear error for retry
    }

# =============================================================================
# BUILD ROBUST GRAPH
# =============================================================================

def build_robust_pipeline():
    builder = StateGraph(RobustState)

    # Main flow
    builder.add_node("process", robust_process)
    builder.add_node("validate", validate_result)
    builder.add_node("success", format_success)

    # Error handling
    builder.add_node("retry", increment_retry)
    builder.add_node("fallback", fallback_process)
    builder.add_node("fatal", handle_fatal_error)

    # Edges
    builder.add_edge(START, "process")
    builder.add_edge("process", "validate")

    builder.add_conditional_edges(
        "validate",
        check_error,
        {
            "success": "success",
            "retry": "retry",
            "fallback": "fallback",
            "fatal": "fatal"
        }
    )

    # Retry goes back to process
    builder.add_edge("retry", "process")

    # Fallback goes to success formatting
    builder.add_edge("fallback", "success")

    # End nodes
    builder.add_edge("success", END)
    builder.add_edge("fatal", END)

    return builder.compile()

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    pipeline = build_robust_pipeline()

    for i in range(3):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}")
        print("="*60)

        result = pipeline.invoke({
            "messages": [],
            "data": {},
            "result": None,
            "error": None,
            "retry_count": 0,
            "max_retries": 3,
            "fallback_used": False
        })

        print(f"Final message: {result['messages'][-1].content}")
        print(f"Fallback used: {result.get('fallback_used', False)}")
```

---

## Part 4: Testing LangGraph Workflows

### Testing Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     TESTING PYRAMID                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌─────────────────┐                          │
│                    │   E2E Tests     │  (Few, expensive)        │
│                    │  Full pipeline  │                          │
│                    └────────┬────────┘                          │
│               ┌─────────────┴─────────────┐                     │
│               │    Integration Tests      │  (More)             │
│               │   Multi-node sequences    │                     │
│               └─────────────┬─────────────┘                     │
│          ┌──────────────────┴──────────────────┐                │
│          │          Unit Tests                  │  (Many, cheap)│
│          │   Individual nodes and routing      │                │
│          └──────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Test Implementation

```python
# code/test_workflows.py

import pytest
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from unittest.mock import Mock, patch

# =============================================================================
# FIXTURES
# =============================================================================

class TestState(TypedDict):
    input: str
    intermediate: str
    output: str
    error: str | None

def node_a(state: TestState) -> dict:
    return {"intermediate": state["input"].upper()}

def node_b(state: TestState) -> dict:
    return {"output": f"Processed: {state['intermediate']}"}

def router(state: TestState) -> str:
    if "error" in state["input"].lower():
        return "error_handler"
    return "process"

def error_handler(state: TestState) -> dict:
    return {"error": "Error detected", "output": "Error response"}

@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    builder = StateGraph(TestState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder.compile()

@pytest.fixture
def conditional_graph():
    """Create a graph with conditional routing."""
    builder = StateGraph(TestState)
    builder.add_node("process", node_b)
    builder.add_node("error_handler", error_handler)

    builder.add_conditional_edges(
        START,
        router,
        {
            "process": "process",
            "error_handler": "error_handler"
        }
    )
    builder.add_edge("process", END)
    builder.add_edge("error_handler", END)
    return builder.compile()

# =============================================================================
# UNIT TESTS
# =============================================================================

class TestNodes:
    """Test individual node functions."""

    def test_node_a_transforms_input(self):
        state = {"input": "hello", "intermediate": "", "output": "", "error": None}
        result = node_a(state)
        assert result["intermediate"] == "HELLO"

    def test_node_b_formats_output(self):
        state = {"input": "", "intermediate": "TEST", "output": "", "error": None}
        result = node_b(state)
        assert result["output"] == "Processed: TEST"

    def test_router_returns_process_for_normal_input(self):
        state = {"input": "normal", "intermediate": "", "output": "", "error": None}
        assert router(state) == "process"

    def test_router_returns_error_for_error_input(self):
        state = {"input": "has error here", "intermediate": "", "output": "", "error": None}
        assert router(state) == "error_handler"

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSimpleGraph:
    """Test simple graph execution."""

    def test_processes_input_correctly(self, simple_graph):
        result = simple_graph.invoke({
            "input": "test",
            "intermediate": "",
            "output": "",
            "error": None
        })
        assert result["output"] == "Processed: TEST"

    def test_handles_empty_input(self, simple_graph):
        result = simple_graph.invoke({
            "input": "",
            "intermediate": "",
            "output": "",
            "error": None
        })
        assert result["output"] == "Processed: "

class TestConditionalGraph:
    """Test conditional routing."""

    def test_routes_to_process_for_normal_input(self, conditional_graph):
        result = conditional_graph.invoke({
            "input": "normal input",
            "intermediate": "NORMAL INPUT",
            "output": "",
            "error": None
        })
        assert "Processed:" in result["output"]

    def test_routes_to_error_handler_for_error_input(self, conditional_graph):
        result = conditional_graph.invoke({
            "input": "input with error",
            "intermediate": "",
            "output": "",
            "error": None
        })
        assert result["error"] == "Error detected"

# =============================================================================
# MOCKING LLM CALLS
# =============================================================================

class TestWithMockedLLM:
    """Test workflows with mocked LLM calls."""

    def test_llm_node_with_mock(self):
        """Test a node that uses LLM with mocking."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import AIMessage

        # Create mock response
        mock_response = AIMessage(content="Mocked response")

        with patch.object(ChatOpenAI, 'invoke', return_value=mock_response):
            llm = ChatOpenAI(model="gpt-4o-mini")
            response = llm.invoke("Test prompt")
            assert response.content == "Mocked response"

# =============================================================================
# STATE VERIFICATION TESTS
# =============================================================================

class TestStateTransitions:
    """Test state transitions through the graph."""

    def test_state_accumulates_correctly(self, simple_graph):
        """Verify state accumulates through nodes."""
        # Use stream to check intermediate states
        states = []
        for step in simple_graph.stream({
            "input": "hello",
            "intermediate": "",
            "output": "",
            "error": None
        }):
            states.append(step)

        # First step should have intermediate set
        assert "a" in states[0]
        assert states[0]["a"]["intermediate"] == "HELLO"

        # Second step should have output set
        assert "b" in states[1]
        assert "Processed: HELLO" in states[1]["b"]["output"]

# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Part 5: Performance Optimization

### Optimization Strategies

```python
# code/14_optimization.py

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio
from functools import lru_cache
import hashlib

# =============================================================================
# OPTIMIZATION 1: CACHING
# =============================================================================

def get_cache_key(prompt: str) -> str:
    """Generate a cache key for a prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()

# Simple in-memory cache
_cache = {}

def cached_llm_call(llm, messages, cache_ttl: int = 3600):
    """LLM call with caching."""
    key = get_cache_key(str(messages))

    if key in _cache:
        return _cache[key]

    result = llm.invoke(messages)
    _cache[key] = result
    return result

# =============================================================================
# OPTIMIZATION 2: PARALLEL EXECUTION
# =============================================================================

async def parallel_llm_calls(prompts: list[str]) -> list[str]:
    """Execute multiple LLM calls in parallel."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    async def call(prompt):
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    tasks = [call(p) for p in prompts]
    return await asyncio.gather(*tasks)

# =============================================================================
# OPTIMIZATION 3: STREAMING
# =============================================================================

async def stream_response(state: dict, llm: ChatOpenAI):
    """Stream LLM response for better UX."""
    full_response = ""

    async for chunk in llm.astream([HumanMessage(content=state["input"])]):
        if chunk.content:
            full_response += chunk.content
            yield chunk.content  # Yield for real-time display

    return full_response

# =============================================================================
# OPTIMIZATION 4: BATCHING
# =============================================================================

class BatchProcessor:
    """Batch multiple requests for efficiency."""

    def __init__(self, batch_size: int = 5, wait_time: float = 0.1):
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.queue = []
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    async def add_to_batch(self, prompt: str):
        """Add a prompt to the batch."""
        self.queue.append(prompt)

        if len(self.queue) >= self.batch_size:
            return await self._process_batch()

        # Wait for more items
        await asyncio.sleep(self.wait_time)
        if self.queue:
            return await self._process_batch()

    async def _process_batch(self):
        """Process accumulated batch."""
        prompts = self.queue[:]
        self.queue.clear()

        # Use batch method for efficiency
        results = await self.llm.abatch([
            [HumanMessage(content=p)] for p in prompts
        ])

        return [r.content for r in results]

# =============================================================================
# OPTIMIZATION 5: EARLY EXIT
# =============================================================================

class OptimizedState(TypedDict):
    input: str
    output: str
    confidence: float
    should_continue: bool

def quick_check(state: OptimizedState) -> dict:
    """Quick check if we can answer without full processing."""
    # Simple questions don't need full pipeline
    simple_patterns = ["hello", "hi", "thanks", "bye"]
    if any(p in state["input"].lower() for p in simple_patterns):
        return {
            "output": "Hello! How can I help you?",
            "confidence": 1.0,
            "should_continue": False
        }
    return {"should_continue": True}

def route_by_confidence(state: OptimizedState) -> str:
    if state.get("should_continue", True):
        return "full_process"
    return "output"

# =============================================================================
# OPTIMIZATION 6: MODEL SELECTION
# =============================================================================

def select_model_by_complexity(task: str) -> ChatOpenAI:
    """Select appropriate model based on task complexity."""
    # Simple heuristic - in production, use a classifier
    complex_indicators = ["analyze", "compare", "explain in detail", "code", "mathematical"]

    is_complex = any(ind in task.lower() for ind in complex_indicators)

    if is_complex:
        return ChatOpenAI(model="gpt-4o", temperature=0)  # More capable
    else:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Faster, cheaper

# =============================================================================
# OPTIMIZED GRAPH
# =============================================================================

def build_optimized_graph():
    """Build a graph with optimization patterns."""
    builder = StateGraph(OptimizedState)

    builder.add_node("quick_check", quick_check)
    builder.add_node("full_process", lambda s: {"output": "Full processing result"})
    builder.add_node("output", lambda s: s)

    builder.add_edge(START, "quick_check")

    builder.add_conditional_edges(
        "quick_check",
        route_by_confidence,
        {
            "full_process": "full_process",
            "output": "output"
        }
    )

    builder.add_edge("full_process", "output")
    builder.add_edge("output", END)

    return builder.compile()
```

---

## Summary: Production Patterns

| Pattern | Purpose | Key Implementation |
|---------|---------|-------------------|
| Quality Gates | Ensure output quality | Validation nodes, quality metrics |
| Multi-Agent | Specialized processing | Supervisor coordination |
| Error Recovery | Handle failures | Retry, fallback, escalation |
| Testing | Verify correctness | Unit, integration, E2E tests |
| Optimization | Performance | Caching, batching, parallelism |

---

## Key Takeaways

1. **Production RAG** needs query understanding, validation, and quality checks
2. **Multi-agent systems** benefit from a supervisor pattern for coordination
3. **Error handling** should be hierarchical: node → graph → workflow → system
4. **Testing** follows the pyramid: many unit tests, fewer integration, few E2E
5. **Optimization** includes caching, parallelism, early exit, and model selection
6. **Emergence thinking**: Complex, reliable behavior emerges from well-designed simple components

---

## Final Project Challenge

Build a production-ready document processing pipeline that:

1. **Accepts** multiple document types (PDF, text, web pages)
2. **Routes** to specialized processors based on type
3. **Extracts** key information with quality validation
4. **Handles** failures gracefully with retries and fallbacks
5. **Coordinates** multiple agents for complex documents
6. **Streams** progress updates in real-time
7. **Persists** state for long-running operations
8. **Includes** human-in-the-loop for ambiguous cases
9. **Has** comprehensive test coverage

---

## What's Next

Congratulations on completing Week 3! You now understand:
- How to orchestrate complex agent workflows
- Event-driven patterns for responsive systems
- Production-grade pipeline design

In **Week 4: APIs & Real Data**, you'll learn to:
- Connect agents to real APIs and enterprise data
- Implement RAG with live data sources
- Build a live-data powered AI assistant

---

*"The mark of a mature system is not that it never fails, but that it fails gracefully and recovers quickly."*
