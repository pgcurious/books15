# Week 1: Foundations of Agentic AI

> "The best way to predict the future is to invent it." — Alan Kay

Welcome to Week 1! This week we build the mental models you need to truly understand Agentic AI—not just how to use it, but *why* it works.

---

## Learning Objectives

By the end of this week, you will:
- Understand what makes AI "agentic" at a fundamental level
- Master the three core building blocks: LLMs, Tools, and Memory
- Recognize patterns of agentic AI across industries
- Develop intuition for when and how to apply agentic approaches

---

## Contents

| Module | Topic | Duration |
|--------|-------|----------|
| 1.1 | [Introduction to Agentic AI](01_introduction_to_agentic_ai.md) | 45 min |
| 1.2 | [Core Building Blocks](02_core_building_blocks.md) | 60 min |
| 1.3 | [Industry Use Cases](03_industry_use_cases.md) | 45 min |

---

## The Three Thinking Frameworks

Throughout this week, we'll apply three powerful thinking frameworks:

### 🔬 First Principles Thinking

**What it is:** Breaking down complex problems into their most basic elements and reasoning up from there.

**Example in Agentic AI:**
- First principle: Language models predict the next token
- Built up: If we can predict tokens, we can complete thoughts
- Further: If we can complete thoughts, we can reason
- Finally: If we can reason AND take actions, we have agency

### 🔄 Analogical Thinking

**What it is:** Understanding new concepts by mapping them to familiar ones.

**The Master Analogy for This Course:**

```
An AI Agent is like a new employee at your company:

- The LLM is their brain (knowledge, reasoning)
- Tools are their access to systems (email, databases, APIs)
- Memory is their notebook and experience
- The prompt is their job description and training
- The task is what you've asked them to accomplish
```

### 🌊 Emergence Thinking

**What it is:** Understanding how complex behaviors arise from simple components interacting.

**Key Insight:**
An LLM alone can only generate text. Tools alone are just functions. Memory alone is just storage. But combine them with the right orchestration, and **agency emerges**—behavior that appears goal-directed, adaptive, and intelligent.

---

## Quick Start

Before diving into the modules, ensure your environment is set up:

```bash
# From the repository root
pip install -r requirements.txt

# Create your .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

Test your setup:

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Say 'Hello, Agentic AI!'")
print(response.content)
```

---

## Key Takeaways Preview

After completing Week 1, you'll understand:

1. **Agency is not magic**—it's a well-designed system of components
2. **The power is in the composition**—LLM + Tools + Memory = Agent
3. **Patterns repeat across industries**—master the pattern, apply anywhere
4. **Thinking frameworks accelerate learning**—first principles, analogies, and emergence

---

## Next Steps

Start with [Module 1.1: Introduction to Agentic AI](01_introduction_to_agentic_ai.md)

---

*"Understanding the fundamentals deeply is worth more than memorizing a thousand tutorials."*
