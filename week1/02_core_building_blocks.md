# Module 1.2: Core Building Blocks

## What You'll Learn
- Deep understanding of LLMs as reasoning engines
- How tools extend agent capabilities
- Memory systems: short-term, long-term, and semantic
- How these components interact to create agency

---

## The Trinity of Agentic AI

```
                    ┌─────────────┐
                    │    AGENT    │
                    │   BEHAVIOR  │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │    LLM      │ │   TOOLS     │ │   MEMORY    │
    │  (Reasoning)│ │  (Actions)  │ │ (Persistence)│
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
           └───────────────┴───────────────┘
                           │
                    Emergence of Agency
```

Let's examine each building block through our three thinking frameworks.

---

## Part 1: Large Language Models (LLMs)

### First Principles: What Is an LLM, Really?

Strip away the hype. At its core, an LLM is:

**First Principle #1: A function that predicts the next token given previous tokens.**

```
P(next_token | previous_tokens)
```

That's it. Everything else—reasoning, creativity, understanding—emerges from this simple function applied at massive scale.

**First Principle #2: Scale transforms quantity into quality.**

- 1 million parameters → autocomplete
- 1 billion parameters → grammar, coherence
- 100 billion parameters → reasoning, creativity
- 1 trillion+ parameters → complex multi-step reasoning

This is **emergence** at work. No one programmed "reasoning" into GPT-4. It emerged from predicting tokens at scale.

### Analogical Thinking: LLM as Brain

Think of the LLM as the agent's **brain**:

| Human Brain | LLM |
|-------------|-----|
| Neurons & synapses | Parameters & weights |
| Training & education | Pre-training on text |
| Specialized regions | Attention heads |
| Working memory | Context window |
| Intuition | Pattern matching |
| Reasoning | Chain of thought |

**Key Insight:** Just like a brain without senses or limbs is limited, an LLM without tools is limited to what it already knows.

### What LLMs Do Well

1. **Pattern Recognition**: Identifying structures in text and data
2. **Reasoning**: Breaking down complex problems (with prompting)
3. **Generation**: Creating coherent, contextual text
4. **Translation**: Between languages, formats, styles
5. **Summarization**: Condensing information
6. **Planning**: Outlining multi-step approaches

### What LLMs Cannot Do (Alone)

1. **Access Current Information**: Knowledge cutoff
2. **Perform Actions**: Can't send emails, run code
3. **Remember Past Sessions**: No persistence
4. **Verify Facts**: Can't check against reality
5. **Learn After Deployment**: Weights are frozen

**This is why we need tools and memory.**

### Practical: Understanding Your LLM

```python
# code/01_llm_basics.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,  # Controls randomness
    max_tokens=1000   # Controls response length
)

# Simple invocation
response = llm.invoke("What are the three laws of thermodynamics?")
print("Simple Response:")
print(response.content)

# With system context (this shapes the agent's "personality")
messages = [
    SystemMessage(content="You are a physics teacher who explains concepts simply."),
    HumanMessage(content="What are the three laws of thermodynamics?")
]
response = llm.invoke(messages)
print("\nWith System Context:")
print(response.content)

# Chain of thought (emergence of reasoning)
cot_prompt = """
Solve this step by step:
If a train travels at 60 mph for 2.5 hours, then speeds up to 80 mph for 1.5 hours,
what is the total distance traveled?

Think through each step before giving the final answer.
"""
response = llm.invoke(cot_prompt)
print("\nChain of Thought:")
print(response.content)
```

---

## Part 2: Tools

### First Principles: What Is a Tool?

**First Principle #1: A tool is a function the LLM can choose to invoke.**

```python
def tool(inputs) -> outputs:
    """Perform some action in the world."""
    pass
```

**First Principle #2: Tools extend the agent's capability boundary.**

Without tools, an agent can only:
- Transform input text to output text
- Use knowledge from training data

With tools, an agent can:
- Access current information (search, APIs)
- Perform actions (send emails, create files)
- Interact with systems (databases, services)
- Verify and validate (calculators, code execution)

### Analogical Thinking: Tools as Senses and Limbs

If the LLM is the brain, tools are the **senses and limbs**:

| Human Capability | Agent Tool |
|-----------------|------------|
| Eyes | Web scraper, image analysis |
| Ears | Speech recognition, audio processing |
| Hands | API calls, file operations |
| Calculator | Math computation tool |
| Encyclopedia | Search engine, knowledge base |
| Phone | Email, messaging APIs |

**Key Insight:** A brain without a body can think but cannot act. Tools give the agent a body.

### Types of Tools

```
┌─────────────────────────────────────────────────────────────────┐
│                        TOOL TAXONOMY                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INFORMATION RETRIEVAL          ACTION EXECUTION                │
│  ├── Search (web, semantic)     ├── API calls                  │
│  ├── Database queries           ├── File operations            │
│  ├── Document retrieval         ├── Code execution             │
│  └── API data fetching          └── System commands            │
│                                                                  │
│  COMPUTATION                    COMMUNICATION                   │
│  ├── Calculator                 ├── Email sending              │
│  ├── Code interpreter           ├── Message posting            │
│  ├── Data analysis              ├── Notification dispatch      │
│  └── Format conversion          └── Human escalation           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Emergence: Tool Selection

Here's where emergence becomes beautiful. When you give an LLM access to tools, it learns to:

1. **Recognize when a tool is needed** ("I need current information...")
2. **Select the appropriate tool** ("...so I'll use search")
3. **Format the input correctly** (structured query)
4. **Interpret the results** (extract relevant information)
5. **Decide if more tools are needed** (iterative refinement)

No one explicitly programs this decision tree. It **emerges** from the interaction of:
- Tool descriptions in the prompt
- LLM's reasoning capabilities
- The structure of the task

### Practical: Building Tools

```python
# code/02_tools_basics.py

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()

# Define tools using the @tool decorator
@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use this for any mathematical calculations.

    Args:
        expression: A mathematical expression like "2 + 2" or "sqrt(16)"
    """
    try:
        # Safe evaluation (in production, use a proper math parser)
        allowed_names = {"sqrt": __import__("math").sqrt, "pi": 3.14159}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_weather(location: str) -> str:
    """
    Get the current weather for a location.
    Use this when the user asks about weather.

    Args:
        location: The city and state, e.g., "San Francisco, CA"
    """
    # Simulated weather data (in production, call a weather API)
    weather_data = {
        "San Francisco, CA": {"temp": 65, "condition": "Foggy"},
        "New York, NY": {"temp": 45, "condition": "Cloudy"},
        "Miami, FL": {"temp": 80, "condition": "Sunny"},
    }
    data = weather_data.get(location, {"temp": 70, "condition": "Unknown"})
    return f"Weather in {location}: {data['temp']}°F, {data['condition']}"

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base for information.
    Use this for company-specific or domain-specific questions.

    Args:
        query: The search query
    """
    # Simulated knowledge base (in production, use vector search)
    knowledge = {
        "vacation policy": "Employees get 20 days PTO per year.",
        "expense reporting": "Submit expenses within 30 days via the ExpenseBot.",
        "meeting rooms": "Book via the calendar app. Max 2 hours per booking.",
    }

    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "No relevant information found. Please contact HR."

# Create the LLM with tools bound
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([calculate, get_current_weather, search_knowledge_base])

# Test tool selection (the LLM chooses which tool to use)
test_queries = [
    "What's 15% of 230?",
    "What's the weather in Miami, FL?",
    "What's our company vacation policy?",
    "Tell me a joke"  # No tool needed
]

for query in test_queries:
    print(f"\nQuery: {query}")
    response = llm_with_tools.invoke(query)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool Called: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
    else:
        print(f"Response: {response.content}")
```

---

## Part 3: Memory

### First Principles: What Is Memory?

**First Principle #1: Memory is persistence of information across time.**

Without memory, every interaction is isolated. The agent has no context, no history, no learning.

**First Principle #2: Different types of information require different memory systems.**

Just as humans have working memory, short-term memory, and long-term memory, agents need different storage mechanisms.

### Analogical Thinking: Memory Types

| Human Memory | Agent Memory | Purpose |
|-------------|--------------|---------|
| Working Memory | Context Window | Current task processing |
| Short-term Memory | Conversation History | Recent interactions |
| Long-term Memory | Vector Database | Persistent knowledge |
| Episodic Memory | Interaction Logs | Specific experiences |
| Semantic Memory | Knowledge Graphs | Facts and relationships |
| Procedural Memory | Saved Prompts/Tools | How to do things |

### Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT MEMORY SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              WORKING MEMORY (Context Window)             │   │
│  │  • Current conversation                                  │   │
│  │  • Active task context                                   │   │
│  │  • Recently retrieved information                        │   │
│  │  • Limited size (4K - 200K tokens)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │             SHORT-TERM MEMORY (Session Buffer)           │   │
│  │  • Full conversation history                             │   │
│  │  • Summarized when too long                              │   │
│  │  • Per-session persistence                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LONG-TERM MEMORY (Vector Store)             │   │
│  │  • Past conversations (summarized)                       │   │
│  │  • Domain knowledge                                      │   │
│  │  • User preferences                                      │   │
│  │  • Learned patterns                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Emergence: Memory Enables Learning

Here's the profound insight: **memory enables learning without retraining**.

An agent with memory can:
1. Remember user preferences
2. Recall past decisions and outcomes
3. Build up domain expertise through interactions
4. Avoid repeating mistakes
5. Personalize responses over time

This creates a form of **emergent learning**—the agent appears to learn and improve without any change to its underlying model.

### Practical: Implementing Memory

```python
# code/03_memory_basics.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

# ============================================
# Part 1: Simple Conversation Memory
# ============================================

class ConversationMemory:
    """Simple in-memory conversation storage."""

    def __init__(self):
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []

# Usage demonstration
memory = ConversationMemory()
llm = ChatOpenAI(model="gpt-4o-mini")

def chat_with_memory(user_input: str) -> str:
    """Chat function that maintains conversation context."""
    memory.add_user_message(user_input)

    # Include system message + conversation history
    messages = [
        SystemMessage(content="You are a helpful assistant. Be concise."),
        *memory.get_messages()
    ]

    response = llm.invoke(messages)
    memory.add_ai_message(response.content)

    return response.content

# Test memory persistence
print("Testing Conversation Memory:")
print("-" * 40)

print("User: My name is Alice")
print(f"AI: {chat_with_memory('My name is Alice')}")

print("\nUser: What's 2+2?")
print(f"AI: {chat_with_memory('What is 2+2?')}")

print("\nUser: What's my name?")
print(f"AI: {chat_with_memory('What is my name?')}")  # Should remember Alice!

# ============================================
# Part 2: LangChain's Built-in Memory
# ============================================

# Store for multiple sessions
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Create a runnable with message history
llm_with_memory = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

# ============================================
# Part 3: Summarization for Long Conversations
# ============================================

def summarize_conversation(messages: list, llm) -> str:
    """Summarize a conversation to save context space."""
    conversation_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages
    ])

    summary_prompt = f"""Summarize this conversation in 2-3 sentences,
    capturing the key points and any important information about the user:

    {conversation_text}
    """

    response = llm.invoke(summary_prompt)
    return response.content

# ============================================
# Part 4: Semantic Memory with Vector Store (Conceptual)
# ============================================

"""
For long-term semantic memory, you would use a vector database:

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings,
    persist_directory="./memory_db"
)

# Store a memory
vectorstore.add_texts(
    texts=["User prefers concise responses"],
    metadatas=[{"type": "preference", "user_id": "alice"}]
)

# Retrieve relevant memories
results = vectorstore.similarity_search(
    "How should I format my response?",
    k=3
)
"""

print("\n" + "=" * 40)
print("Memory Concepts Demonstrated!")
print("=" * 40)
```

---

## How The Trinity Works Together

Now let's see how LLM, Tools, and Memory combine to create agency:

```
User Query: "What was the weather when I last asked about San Francisco?"

┌─────────────────────────────────────────────────────────────────┐
│  1. PERCEIVE (Memory Read)                                       │
│     └── Query memory: "When did user ask about SF weather?"     │
│     └── Found: "3 days ago, user asked about SF weather (65°F)" │
├─────────────────────────────────────────────────────────────────┤
│  2. REASON (LLM)                                                 │
│     └── "User wants historical comparison"                       │
│     └── "I should also get current weather for comparison"       │
├─────────────────────────────────────────────────────────────────┤
│  3. ACT (Tool Use)                                               │
│     └── Call: get_current_weather("San Francisco, CA")          │
│     └── Result: "68°F, Partly Cloudy"                           │
├─────────────────────────────────────────────────────────────────┤
│  4. SYNTHESIZE (LLM)                                             │
│     └── Combine memory + tool result                             │
│     └── "3 days ago it was 65°F and foggy, now it's 68°F..."   │
├─────────────────────────────────────────────────────────────────┤
│  5. PERSIST (Memory Write)                                       │
│     └── Store: "User asked about SF weather comparison"          │
│     └── Update: "Current SF weather: 68°F"                       │
└─────────────────────────────────────────────────────────────────┘
```

### Practical: Complete Agent Loop

```python
# code/04_agent_loop.py

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Simple memory store
class AgentMemory:
    def __init__(self):
        self.conversation_history = []
        self.facts = {}  # Long-term storage of facts

    def add_interaction(self, user_input: str, agent_response: str):
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "agent": agent_response
        })

    def store_fact(self, key: str, value: str):
        self.facts[key] = value

    def get_fact(self, key: str) -> str:
        return self.facts.get(key, "Unknown")

    def get_recent_context(self, n: int = 5) -> str:
        recent = self.conversation_history[-n:]
        return "\n".join([
            f"User: {i['user']}\nAgent: {i['agent']}"
            for i in recent
        ])

# Tools
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except:
        return "Error evaluating expression"

@tool
def remember_fact(fact_key: str, fact_value: str) -> str:
    """
    Remember a fact for later. Use this to store important information.

    Args:
        fact_key: A short key to identify this fact (e.g., "user_name")
        fact_value: The value to remember (e.g., "Alice")
    """
    # This would connect to memory.store_fact() in a real implementation
    return f"Remembered: {fact_key} = {fact_value}"

@tool
def recall_fact(fact_key: str) -> str:
    """
    Recall a previously stored fact.

    Args:
        fact_key: The key of the fact to recall
    """
    # This would connect to memory.get_fact() in a real implementation
    return f"Recalled fact for '{fact_key}'"

# Create agent
class SimpleAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.tools = [get_current_time, calculate, remember_fact, recall_fact]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = AgentMemory()

        self.system_prompt = """You are a helpful AI assistant with memory and tools.

You have access to these tools:
- get_current_time: Get the current date and time
- calculate: Perform mathematical calculations
- remember_fact: Store important information for later
- recall_fact: Retrieve previously stored information

When appropriate, use tools to help answer questions. Always be helpful and concise.

Recent conversation context:
{context}
"""

    def run(self, user_input: str) -> str:
        # Build context from memory
        context = self.memory.get_recent_context()

        # Create messages
        messages = [
            SystemMessage(content=self.system_prompt.format(context=context)),
            HumanMessage(content=user_input)
        ]

        # Get LLM response (may include tool calls)
        response = self.llm_with_tools.invoke(messages)

        # Handle tool calls if any
        if response.tool_calls:
            # Execute tools and gather results
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                # Find and execute the tool
                for t in self.tools:
                    if t.name == tool_name:
                        result = t.invoke(tool_args)
                        tool_results.append(f"{tool_name}: {result}")
                        break

            # Get final response with tool results
            messages.append(response)
            messages.append(HumanMessage(
                content=f"Tool results:\n" + "\n".join(tool_results)
            ))
            final_response = self.llm.invoke(messages)
            agent_response = final_response.content
        else:
            agent_response = response.content

        # Store in memory
        self.memory.add_interaction(user_input, agent_response)

        return agent_response

# Demo
if __name__ == "__main__":
    agent = SimpleAgent()

    print("Simple Agent Demo")
    print("=" * 40)

    queries = [
        "What time is it right now?",
        "Calculate 15% tip on a $67.50 bill",
        "My favorite color is blue, please remember that",
        "What's my favorite color?"
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = agent.run(query)
        print(f"Agent: {response}")
```

---

## Summary: The Building Blocks

| Component | First Principle | Analogy | Enables |
|-----------|-----------------|---------|---------|
| **LLM** | Token prediction at scale | Brain | Reasoning, generation |
| **Tools** | Functions LLM can invoke | Hands & senses | Actions, real-world access |
| **Memory** | Persistence across time | Notebook & experience | Context, learning |

### The Emergence Formula

```
LLM (reasoning) + Tools (action) + Memory (persistence) = Agency (emergent behavior)
```

**Key Insight:** None of these components alone is "agentic." Agency **emerges** from their interaction, just as consciousness emerges from neurons, or life emerges from chemistry.

---

## Exercises

1. **Modify the tools**: Add a new tool to the agent (e.g., `search_wikipedia`)
2. **Enhance memory**: Implement a summarization function when history gets too long
3. **Chain tools**: Create a task that requires multiple tools in sequence
4. **Test emergence**: Give the agent a complex task and observe how it chains reasoning, tool use, and memory

---

## Next Steps

In [Module 1.3: Industry Use Cases](03_industry_use_cases.md), we'll see how these building blocks combine to solve real-world problems across industries.
