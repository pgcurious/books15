# Week 2: Build Your First Agent

> "I hear and I forget. I see and I remember. I do and I understand." — Confucius

Welcome to Week 2! This week we move from theory to practice. You'll build, deploy, and debug real AI agents using LangChain.

---

## Learning Objectives

By the end of this week, you will:
- Master LangChain's core abstractions for building agents
- Build a fully functional agent from scratch
- Deploy your agent as a web service
- Debug and optimize agent performance
- Understand common pitfalls and how to avoid them

---

## Contents

| Module | Topic | Duration |
|--------|-------|----------|
| 2.1 | [Hands-on Coding with LangChain](01_langchain_fundamentals.md) | 90 min |
| 2.2 | [Deploy Your First Working Agent](02_deploy_your_agent.md) | 75 min |
| 2.3 | [Debugging & Optimization Basics](03_debugging_and_optimization.md) | 60 min |

---

## Prerequisites

Before starting Week 2, ensure you have:

1. ✅ Completed Week 1 (conceptual foundation)
2. ✅ Python 3.9+ installed
3. ✅ OpenAI API key (or compatible LLM provider)
4. ✅ Virtual environment set up

```bash
# Quick setup check
python --version  # Should be 3.9+
pip list | grep langchain  # Should show langchain packages
echo $OPENAI_API_KEY  # Should show your key (or check .env)
```

---

## The Week 2 Philosophy

### First Principles: Why LangChain?

Building agents from scratch requires handling:
- Message formatting for different LLMs
- Tool definition and execution
- Memory management
- Error handling and retries
- Streaming and async operations

**First Principle:** Don't reinvent the wheel. LangChain abstracts these complexities so you can focus on agent behavior.

### Analogical Thinking: LangChain as a Framework

Think of LangChain like a web framework (Django, Rails, Express):

| Web Framework | LangChain |
|--------------|-----------|
| HTTP handling | LLM API calls |
| Routing | Chains and agents |
| Middleware | Callbacks and hooks |
| ORM | Memory and retrievers |
| Templates | Prompt templates |
| Deployment | LangServe |

You *could* build everything from scratch. But why would you?

### Emergence Thinking: Composability Creates Power

LangChain's power comes from composability:

```
Simple Components:
  Prompt + LLM + Tool + Memory

Combined via Chains:
  PromptTemplate | LLM | OutputParser

Creates Emergent Capabilities:
  Sophisticated, reliable, observable agent behavior
```

---

## What You'll Build

By the end of Week 2, you'll have built:

### 1. A Research Assistant Agent
An agent that can:
- Search the web for information
- Summarize documents
- Answer follow-up questions with context
- Remember conversation history

### 2. A Deployed API Service
Your agent exposed as:
- REST API endpoint
- Interactive playground UI
- Streaming responses
- Proper error handling

### 3. An Observability Dashboard
Tools to understand:
- What your agent is doing
- Where it's spending time
- Why it might be failing
- How to make it better

---

## Quick Architecture Overview

```
Week 2 Architecture
===================

┌──────────────────────────────────────────────────────────────┐
│                        YOUR AGENT                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                     LangChain                          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │ │
│  │  │  Prompt  │──│   LLM    │──│  Tools   │            │ │
│  │  │ Template │  │ (GPT-4)  │  │(Search,  │            │ │
│  │  └──────────┘  └──────────┘  │ Calculate)│            │ │
│  │       │                      └──────────┘            │ │
│  │       │        ┌──────────┐                          │ │
│  │       └────────│  Memory  │                          │ │
│  │                └──────────┘                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    LangServe                           │ │
│  │            (FastAPI-based deployment)                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    LangSmith                           │ │
│  │         (Observability & debugging)                    │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Setup for Week 2

Create a new directory for your Week 2 work:

```bash
# From repository root
cd week2/code

# Install additional dependencies
pip install langchain langchain-openai langchain-community
pip install fastapi uvicorn  # For deployment
pip install duckduckgo-search  # For web search tool

# Verify installation
python -c "import langchain; print(f'LangChain version: {langchain.__version__}')"
```

Create a `.env` file if you haven't:

```bash
# .env file in repository root
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional, for LangSmith
LANGCHAIN_TRACING_V2=true  # Optional, enables tracing
```

---

## Module Overview

### Module 2.1: Hands-on Coding with LangChain
- LangChain Expression Language (LCEL)
- Building chains step by step
- Creating and binding tools
- Implementing memory
- Building your first complete agent

### Module 2.2: Deploy Your First Working Agent
- FastAPI basics for AI services
- LangServe for quick deployment
- The playground interface
- Handling streaming responses
- Production considerations

### Module 2.3: Debugging & Optimization Basics
- Common failure modes
- LangSmith for observability
- Prompt engineering for reliability
- Performance optimization
- Testing strategies

---

## Key Concepts Preview

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| LCEL | LangChain Expression Language | Composable, streamable chains |
| Runnable | Base interface for chain components | Everything is a Runnable |
| Tool | Function the LLM can call | Extends agent capabilities |
| Memory | Conversation persistence | Context across interactions |
| Callbacks | Hooks into execution | Logging, streaming, debugging |
| LangServe | Deployment framework | Production-ready APIs |

---

## Let's Build!

Start with [Module 2.1: Hands-on Coding with LangChain](01_langchain_fundamentals.md)

---

## Troubleshooting

### Common Setup Issues

**Import errors?**
```bash
pip install --upgrade langchain langchain-openai langchain-community
```

**API key issues?**
```bash
# Check your key is set
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT SET')[:10])"
```

**Rate limiting?**
- Use `gpt-4o-mini` instead of `gpt-4` for development
- Add delays between calls if needed

---

*"The best way to learn is by doing. Let's build something."*
