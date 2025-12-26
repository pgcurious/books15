# Week 5: Multi-Agent Systems

> "The whole is greater than the sum of its parts." — Aristotle

## Learning Objectives

By the end of this week, you will be able to:

- **Design agent architectures** that collaborate, delegate, and specialize
- **Implement memory systems** that enable short-term context and long-term learning
- **Build agent teams** that solve complex problems through emergent coordination
- **Apply orchestration patterns** for supervisor, peer-to-peer, and hierarchical systems
- **Debug and optimize** multi-agent interactions for production environments

---

## Contents

| Module | Topic | Duration |
|--------|-------|----------|
| 5.1 | [Designing Collaborative Agents](01_designing_collaborative_agents.md) | 90 min |
| 5.2 | [Agent Memory Systems](02_agent_memory_systems.md) | 90 min |
| 5.3 | [Building Agent Teams](03_building_agent_teams.md) | 120 min |

**Total Duration**: ~5 hours

---

## The Week 5 Philosophy: Thinking in Systems

This week, we apply our three thinking frameworks to understand how individual agents become powerful teams:

### First Principles Thinking: What Makes Collaboration Work?

Strip away the complexity and ask: **What are the fundamental requirements for agents to work together?**

```
Multi-Agent Collaboration = Communication + Coordination + Shared Goals

Where:
├── Communication = Message passing between agents
├── Coordination = Protocols for who does what, when
└── Shared Goals = Alignment on outcomes and success metrics
```

At the atomic level, multi-agent systems require:
1. **Identity**: Each agent knows itself and others
2. **Messages**: A way to send/receive information
3. **State**: Shared or distributed knowledge
4. **Protocols**: Rules for interaction

### Analogical Thinking: Agents as Organizations

| Human Organization | Multi-Agent System | Key Insight |
|-------------------|-------------------|-------------|
| Company with CEO | Supervisor Agent Pattern | Central authority delegates and reviews |
| Self-organizing team | Peer-to-Peer Agents | Emergent leadership, consensus decisions |
| Hospital ER | Specialist Agent Pool | Right expert for each problem |
| Assembly line | Pipeline Agents | Sequential processing, handoffs |
| Research lab | Collaborative Agents | Shared memory, iterative refinement |
| Bee colony | Swarm Intelligence | Simple rules, complex outcomes |

Just as organizations have evolved different structures for different purposes, multi-agent systems use different patterns based on their goals.

### Emergence Thinking: Complex Behavior from Simple Agents

The magic of multi-agent systems is **emergence**—complex, intelligent behavior arising from simple agent interactions:

```
Individual Agent Rules          →  Emergent System Behavior
─────────────────────────────────────────────────────────────
"Answer questions in my domain" →  Comprehensive coverage
"Ask for help when uncertain"   →  Graceful degradation
"Share useful findings"         →  Collective intelligence
"Validate others' work"         →  Error correction
"Remember past interactions"    →  Learning and adaptation
```

**The emergence principle**: You don't program the team's intelligence—you program how individuals interact, and intelligence emerges.

---

## Architecture Overview: The Multi-Agent Landscape

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT ARCHITECTURES                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │              ORCHESTRATION PATTERNS                        │     │
│   ├───────────────────────────────────────────────────────────┤     │
│   │                                                            │     │
│   │  ┌─────────┐    ┌─────────┐    ┌─────────┐               │     │
│   │  │SUPERVISOR│    │  PEER   │    │HIERARCHI│               │     │
│   │  │ PATTERN │    │ TO PEER │    │   CAL   │               │     │
│   │  └────┬────┘    └────┬────┘    └────┬────┘               │     │
│   │       │              │              │                     │     │
│   │       ▼              ▼              ▼                     │     │
│   │   ┌───────┐      ┌──┴──┐       ┌───────┐                 │     │
│   │   │  Boss │      │Agent│◄─────►│ Layer │                 │     │
│   │   └───┬───┘      └──┬──┘       │   1   │                 │     │
│   │       │             │          └───┬───┘                  │     │
│   │   ┌───┴───┐     ┌──┴──┐           │                      │     │
│   │   ▼   ▼   ▼     │Agent│◄─────►┌───┴───┐                  │     │
│   │  A1  A2  A3     └──┬──┘       │ Layer │                  │     │
│   │                    │          │   2   │                  │     │
│   │                ┌──┴──┐       └───────┘                   │     │
│   │                │Agent│                                    │     │
│   │                └─────┘                                    │     │
│   │                                                            │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │                 MEMORY ARCHITECTURE                        │     │
│   ├───────────────────────────────────────────────────────────┤     │
│   │                                                            │     │
│   │   SHORT-TERM                    LONG-TERM                  │     │
│   │   ┌──────────────┐             ┌──────────────┐           │     │
│   │   │ Conversation │             │ Vector Store │           │     │
│   │   │   Context    │             │  (Semantic)  │           │     │
│   │   └──────┬───────┘             └──────┬───────┘           │     │
│   │          │                            │                    │     │
│   │   ┌──────┴───────┐             ┌──────┴───────┐           │     │
│   │   │   Working    │             │  Episodic    │           │     │
│   │   │   Memory     │             │   Memory     │           │     │
│   │   └──────────────┘             └──────────────┘           │     │
│   │                                                            │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │              COMMUNICATION PATTERNS                        │     │
│   ├───────────────────────────────────────────────────────────┤     │
│   │                                                            │     │
│   │   DIRECT         BROADCAST       BLACKBOARD               │     │
│   │   A ──► B        A ──► ALL       ┌─────────┐              │     │
│   │                                  │ Shared  │              │     │
│   │   REQUEST/       PUB/SUB        │  State  │              │     │
│   │   RESPONSE       A ──► Topic    │◄──────► │ A,B,C        │     │
│   │   A ◄──► B       └──► B,C       └─────────┘              │     │
│   │                                                            │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Setup for Week 5

### Prerequisites

Ensure you have completed Weeks 1-4 and have your environment configured:

```bash
# Navigate to the course directory
cd /path/to/agentic-ai-course

# Install additional dependencies for Week 5
pip install langgraph>=0.0.40 langchain-community>=0.0.10 chromadb>=0.4.0

# Verify installation
python -c "import langgraph; print(f'LangGraph version: {langgraph.__version__}')"
```

### Environment Variables

Ensure your `.env` file contains:

```bash
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key  # Optional, for tracing
LANGCHAIN_TRACING_V2=true  # Optional
```

### Recommended: LangSmith Tracing

Multi-agent systems can be complex to debug. We strongly recommend enabling LangSmith for this week:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="week5-multi-agent"
```

---

## What You'll Build

By the end of this week, you'll have built:

### 1. Collaborative Research System
A team of specialized agents that work together:
- **Research Agent**: Gathers information from multiple sources
- **Analysis Agent**: Processes and synthesizes findings
- **Writing Agent**: Produces coherent reports
- **Reviewer Agent**: Validates and improves output

### 2. Memory-Enhanced Agent
An agent with both:
- **Short-term memory**: Conversation context and working memory
- **Long-term memory**: Persistent knowledge and episodic recall

### 3. Self-Organizing Agent Team
A dynamic system where:
- Agents discover and leverage each other's capabilities
- Work is automatically routed to the best agent
- The team adapts to new problems without reprogramming

---

## Module Overview

### Module 5.1: Designing Collaborative Agents

**Core Topics:**
- Agent specialization and the division of labor
- Communication protocols between agents
- Delegation patterns: when and how to hand off work
- Supervisor vs. peer-to-peer architectures

**You'll Learn:**
- How to design agents that know their boundaries
- When to use hierarchical vs. flat structures
- How to implement agent-to-agent communication

### Module 5.2: Agent Memory Systems

**Core Topics:**
- Short-term memory: conversation buffers and working memory
- Long-term memory: vector stores and episodic memory
- Memory retrieval strategies
- Shared memory for multi-agent coordination

**You'll Learn:**
- How to give agents persistent memory
- When to use different memory types
- How to build agents that learn from experience

### Module 5.3: Building Agent Teams

**Core Topics:**
- Team composition and role design
- Orchestration with LangGraph
- Error handling and graceful degradation
- Monitoring and debugging multi-agent systems

**You'll Learn:**
- How to assemble effective agent teams
- How to handle failures gracefully
- How to observe and optimize team performance

---

## Key Concepts Preview

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| **Agent Specialization** | Designing agents for specific tasks | Enables expertise and reduces complexity |
| **Supervisor Pattern** | Central agent coordinates others | Clear authority, easier to debug |
| **Peer-to-Peer Pattern** | Agents communicate directly | More flexible, can be more efficient |
| **Short-term Memory** | Context within a session | Maintains conversation coherence |
| **Long-term Memory** | Persistent knowledge | Enables learning and personalization |
| **Shared State** | Memory accessible by all agents | Enables coordination and handoffs |
| **Emergent Behavior** | Complex outcomes from simple rules | The goal of well-designed systems |
| **Graceful Degradation** | System works even when parts fail | Essential for production reliability |

---

## The Journey So Far

```
Week 1: Foundations          Week 2: First Agent       Week 3: LangGraph
┌─────────────────┐         ┌─────────────────┐       ┌─────────────────┐
│ What is an      │         │ Built a working │       │ Orchestrated    │
│ agent? LLMs,    │   ──►   │ agent with      │  ──►  │ multi-step      │
│ tools, memory   │         │ LangChain       │       │ workflows       │
└─────────────────┘         └─────────────────┘       └─────────────────┘
                                                              │
                                                              ▼
Week 4: Real Data           Week 5: Multi-Agent       Week 6: Guardrails
┌─────────────────┐         ┌─────────────────┐       ┌─────────────────┐
│ APIs, RAG,      │         │ Teams of agents │       │ Safety,         │
│ production      │   ──►   │ with memory,    │  ──►  │ compliance,     │
│ pipelines       │         │ collaboration   │       │ monitoring      │
└─────────────────┘         └─────────────────┘       └─────────────────┘
                                   ▲
                                   │
                              YOU ARE HERE
```

---

## Troubleshooting

### Common Issues

**"Agents not communicating"**
- Verify shared state is properly configured
- Check that message formats match expectations
- Enable LangSmith tracing to see message flow

**"Memory not persisting"**
- Confirm vector store is initialized correctly
- Check that embeddings are being generated
- Verify storage backend is accessible

**"Agent loops infinitely"**
- Add maximum iteration limits
- Implement termination conditions
- Check for circular dependencies in agent routing

**"Performance is slow"**
- Consider parallelizing independent agent calls
- Use caching for repeated operations
- Optimize prompts for efficiency

### Getting Help

1. Check the code examples in `code/`
2. Review the LangGraph documentation
3. Enable tracing to debug agent interactions
4. Refer to the troubleshooting guides in earlier weeks

---

## Ready to Build Multi-Agent Systems?

Let's start with **Module 5.1: Designing Collaborative Agents** to understand how to create agents that work together effectively.

The future of AI isn't a single, omniscient agent—it's teams of specialized agents collaborating to solve problems no single agent could handle alone.

**Let's build that future!**

[Start Module 5.1 →](01_designing_collaborative_agents.md)
