# Module 2.3: Debugging & Optimization Basics

## What You'll Learn
- Common failure modes in AI agents
- Debugging techniques and tools
- Using LangSmith for observability
- Performance optimization strategies
- Testing approaches for agents

---

## First Principles: Why Agents Fail

**First Principle #1:** Agents fail because of uncertainty at every layer.

```
Uncertainty Stack:
├── User Intent: What did they really mean?
├── LLM Reasoning: Did it understand correctly?
├── Tool Selection: Did it choose the right tool?
├── Tool Execution: Did the tool work correctly?
├── Response Synthesis: Is the answer accurate?
└── Memory: Is the context correct?
```

**First Principle #2:** Debugging = Reducing uncertainty layer by layer.

---

## The Analogy: Medical Diagnosis

Debugging an agent is like diagnosing a patient:

| Medical Diagnosis | Agent Debugging |
|------------------|-----------------|
| Patient symptoms | Error messages/bad output |
| Medical history | Conversation history |
| Diagnostic tests | Logging and tracing |
| Lab results | Tool outputs |
| Vital signs | Latency and token usage |
| Differential diagnosis | Hypothesis testing |

You systematically narrow down the cause by gathering evidence.

---

## Part 1: Common Failure Modes

### 1. Prompt Failures

**Symptoms:**
- Agent doesn't follow instructions
- Inconsistent behavior
- Wrong format or tone

**Causes:**
```python
# BAD: Vague prompt
system_prompt = "Be helpful"

# GOOD: Specific prompt
system_prompt = """You are a customer service agent for TechCo.

RULES:
1. Always greet the customer by name if known
2. Never make up information - say "I don't know" if unsure
3. For refund requests, gather order number before proceeding
4. Keep responses under 3 sentences unless detail is needed

TONE: Professional but friendly
"""
```

**Fix:** Be explicit, provide examples, set guardrails.

### 2. Tool Failures

**Symptoms:**
- Wrong tool selected
- Tool called with wrong arguments
- Tool errors not handled

**Example:**
```python
# BAD: Vague tool description
@tool
def search(q: str) -> str:
    """Search for things."""
    pass

# GOOD: Precise tool description
@tool
def search_product_catalog(
    query: str,
    category: str = None,
    max_results: int = 5
) -> str:
    """
    Search the product catalog for items.

    Use this when the user asks about products, prices, or availability.
    Do NOT use for general knowledge questions.

    Args:
        query: Search terms (e.g., "blue wireless headphones")
        category: Optional category filter ("electronics", "clothing", etc.)
        max_results: Maximum results to return (default 5)

    Returns:
        JSON list of matching products with name, price, availability
    """
    pass
```

### 3. Memory Failures

**Symptoms:**
- Agent forgets context
- Contradicts itself
- References wrong conversation

**Causes:**
- Context window overflow
- Session ID mismatch
- Memory not persisted

**Fix:**
```python
class RobustMemory:
    def __init__(self, max_messages=50, max_tokens=4000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._enforce_limits()

    def _enforce_limits(self):
        # Trim old messages if needed
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)

        # Summarize if tokens exceed limit
        if self._count_tokens() > self.max_tokens:
            self._summarize_old_messages()
```

### 4. Hallucination

**Symptoms:**
- Agent invents facts
- Provides confident but wrong answers
- Makes up tool results

**Mitigation:**
```python
system_prompt = """
CRITICAL RULES:
1. If you don't know something, say "I don't know"
2. Never invent statistics or facts
3. When using search results, cite the source
4. If a tool fails, tell the user rather than guessing
5. Distinguish between facts and opinions
"""
```

### 5. Infinite Loops

**Symptoms:**
- Agent keeps calling tools repeatedly
- Never produces final answer
- Token usage explodes

**Fix:**
```python
class SafeAgent:
    MAX_TOOL_CALLS = 10

    def run(self, message):
        tool_calls = 0

        while tool_calls < self.MAX_TOOL_CALLS:
            response = self.llm.invoke(messages)

            if not response.tool_calls:
                return response.content  # Done!

            tool_calls += len(response.tool_calls)
            # Execute tools...

        raise TooManyToolCallsError("Agent exceeded maximum tool calls")
```

---

## Part 2: Debugging Techniques

### Technique 1: Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebuggableAgent:
    def run(self, message):
        logger.info(f"INPUT: {message}")
        logger.debug(f"SESSION: {self.session_id}")
        logger.debug(f"MEMORY: {len(self.memory.messages)} messages")

        response = self.llm.invoke(messages)

        logger.info(f"LLM RESPONSE TYPE: {'tool_call' if response.tool_calls else 'direct'}")

        if response.tool_calls:
            for call in response.tool_calls:
                logger.info(f"TOOL CALL: {call['name']}({call['args']})")

        logger.info(f"OUTPUT: {response.content[:100]}...")

        return response.content
```

### Technique 2: Conversation Replay

Save and replay problematic conversations:

```python
import json
from datetime import datetime

class ReplayableAgent:
    def __init__(self):
        self.conversation_log = []

    def run(self, message):
        # Log input
        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "input",
            "content": message
        })

        response = self._process(message)

        # Log output
        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "output",
            "content": response
        })

        return response

    def save_conversation(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.conversation_log, f, indent=2)

    def replay_conversation(self, filename):
        """Replay a saved conversation for debugging."""
        with open(filename, 'r') as f:
            log = json.load(f)

        for entry in log:
            if entry["type"] == "input":
                print(f"USER: {entry['content']}")
                response = self._process(entry['content'])
                print(f"AGENT: {response}")
                print("---")
```

### Technique 3: Step-by-Step Inspection

```python
class InspectableAgent:
    def run_with_inspection(self, message):
        """Run with detailed step inspection."""
        print("=" * 50)
        print(f"STEP 1: Received message")
        print(f"  Message: {message}")
        print()

        print(f"STEP 2: Building context")
        print(f"  Memory messages: {len(self.memory.messages)}")
        print(f"  System prompt length: {len(self.system_prompt)}")
        print()

        print(f"STEP 3: Invoking LLM")
        response = self.llm.invoke(messages)
        print(f"  Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")
        print()

        if response.tool_calls:
            print(f"STEP 4: Executing tools")
            for i, call in enumerate(response.tool_calls):
                print(f"  Tool {i+1}: {call['name']}")
                print(f"  Args: {call['args']}")
                result = self._execute_tool(call)
                print(f"  Result: {result[:100]}...")
                print()

        print(f"STEP 5: Final response")
        print(f"  Length: {len(response.content)} chars")
        print(f"  Preview: {response.content[:200]}...")
        print("=" * 50)

        return response.content
```

---

## Part 3: LangSmith for Observability

LangSmith provides production-grade observability for LangChain applications.

### Setup

```bash
# Install
pip install langsmith

# Set environment variables
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_api_key
export LANGCHAIN_PROJECT="my-agent-project"
```

### Automatic Tracing

Once configured, all LangChain operations are automatically traced:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# This is automatically traced!
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = prompt | llm

# Each invocation creates a trace
response = chain.invoke({"topic": "AI agents"})
```

### What LangSmith Shows

```
Trace: "Tell me about AI agents"
├── ChatPromptTemplate (0.001s)
│   └── Input: {"topic": "AI agents"}
│   └── Output: ChatPromptValue(...)
├── ChatOpenAI (1.234s)
│   └── Input: [SystemMessage, HumanMessage]
│   └── Output: AIMessage(content="...")
│   └── Tokens: 150 input, 300 output
│   └── Cost: $0.0012
└── Total: 1.235s
```

### Custom Tracing

```python
from langsmith import traceable

@traceable(name="custom_process")
def process_with_tracing(data):
    """This function will be traced in LangSmith."""
    result = complex_operation(data)
    return result

@traceable(run_type="tool")
def my_custom_tool(query: str):
    """Traced as a tool invocation."""
    return search(query)
```

### Adding Metadata

```python
from langchain_core.runnables import RunnableConfig

# Add metadata to traces
config = RunnableConfig(
    tags=["production", "user-facing"],
    metadata={
        "user_id": "user_123",
        "session_id": "session_456",
        "feature_flag": "new_prompt_v2"
    }
)

response = chain.invoke({"topic": "AI"}, config=config)
```

---

## Part 4: Performance Optimization

### 1. Model Selection

| Model | Speed | Cost | Quality |
|-------|-------|------|---------|
| gpt-4o-mini | Fast | Low | Good |
| gpt-4o | Medium | High | Best |
| gpt-4-turbo | Medium | Medium | Very Good |

**Strategy:** Use `gpt-4o-mini` for most tasks, escalate to `gpt-4o` for complex reasoning.

```python
class TieredAgent:
    def __init__(self):
        self.fast_llm = ChatOpenAI(model="gpt-4o-mini")
        self.smart_llm = ChatOpenAI(model="gpt-4o")

    def route_to_model(self, message):
        """Route to appropriate model based on complexity."""
        complexity_indicators = [
            "analyze", "compare", "complex", "detailed",
            "step by step", "multiple", "comprehensive"
        ]

        is_complex = any(ind in message.lower() for ind in complexity_indicators)

        return self.smart_llm if is_complex else self.fast_llm
```

### 2. Caching

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(InMemoryCache())

# Now identical prompts return cached results
llm = ChatOpenAI(model="gpt-4o-mini")
response1 = llm.invoke("What is 2+2?")  # API call
response2 = llm.invoke("What is 2+2?")  # Cached!
```

### 3. Batching

```python
# Sequential (slow)
results = []
for item in items:
    result = chain.invoke(item)
    results.append(result)

# Batched (fast - parallel API calls)
results = chain.batch(items, config={"max_concurrency": 5})
```

### 4. Streaming for Perceived Performance

Even if total time is the same, streaming feels faster:

```python
# Without streaming: User waits 3 seconds, then sees full response
response = llm.invoke(prompt)

# With streaming: User sees tokens immediately
for chunk in llm.stream(prompt):
    print(chunk.content, end="", flush=True)
```

### 5. Prompt Optimization

```python
# BAD: Verbose prompt (more tokens = slower + costlier)
prompt = """
I would like you to please help me understand something.
Could you possibly explain to me, if you don't mind,
what the concept of machine learning is? I would really
appreciate a detailed explanation if that's okay with you.
"""

# GOOD: Concise prompt
prompt = "Explain machine learning in 2-3 sentences."
```

### 6. Early Exit Strategies

```python
class EfficientAgent:
    def run(self, message):
        # Check if we can answer without tools
        if self._can_answer_directly(message):
            return self.llm.invoke(message)

        # Check cache
        cached = self.cache.get(message)
        if cached:
            return cached

        # Full agent flow only when necessary
        return self._full_agent_flow(message)
```

---

## Part 5: Testing Strategies

### Unit Testing Tools

```python
import pytest
from your_agent import search_web, calculate

def test_calculate_basic():
    """Test basic calculation."""
    result = calculate.invoke({"expression": "2 + 2"})
    assert "4" in result

def test_calculate_complex():
    """Test complex calculation."""
    result = calculate.invoke({"expression": "sqrt(16) * 5"})
    assert "20" in result

def test_calculate_error():
    """Test error handling."""
    result = calculate.invoke({"expression": "invalid"})
    assert "Error" in result
```

### Integration Testing

```python
import pytest
from your_agent import ResearchAssistant

@pytest.fixture
def agent():
    return ResearchAssistant()

def test_basic_conversation(agent):
    """Test basic Q&A."""
    response = agent.chat("What is 2+2?")
    assert "4" in response

def test_memory_persistence(agent):
    """Test that memory works."""
    agent.chat("My name is TestUser")
    response = agent.chat("What's my name?")
    assert "TestUser" in response

def test_tool_usage(agent):
    """Test that tools are called appropriately."""
    response = agent.chat("What time is it?")
    # Should have used the time tool
    assert ":" in response or "AM" in response or "PM" in response
```

### Evaluation Testing

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Define test cases
test_cases = [
    {
        "input": "What is the capital of France?",
        "expected": "Paris"
    },
    {
        "input": "Calculate 15% of 200",
        "expected": "30"
    }
]

# Create dataset
dataset = client.create_dataset("agent-eval")
for case in test_cases:
    client.create_example(
        inputs={"message": case["input"]},
        outputs={"expected": case["expected"]},
        dataset_id=dataset.id
    )

# Define evaluator
def correctness_evaluator(run, example):
    """Check if the response contains the expected answer."""
    response = run.outputs["response"]
    expected = example.outputs["expected"]
    return {"score": 1 if expected.lower() in response.lower() else 0}

# Run evaluation
results = evaluate(
    agent.chat,
    data=dataset,
    evaluators=[correctness_evaluator]
)
```

---

## Debugging Checklist

When your agent misbehaves, work through this checklist:

```
□ 1. CHECK THE INPUT
  - Is the user message what you expect?
  - Any encoding or formatting issues?

□ 2. CHECK THE PROMPT
  - Is the system prompt loaded correctly?
  - Are variables substituted properly?

□ 3. CHECK THE MEMORY
  - Is conversation history correct?
  - Is the right session being used?

□ 4. CHECK TOOL SELECTION
  - Did the LLM choose the right tool?
  - Are tool descriptions clear enough?

□ 5. CHECK TOOL EXECUTION
  - Did the tool run successfully?
  - Is the output format correct?

□ 6. CHECK THE RESPONSE
  - Is the final response accurate?
  - Did the LLM use tool results correctly?

□ 7. CHECK INFRASTRUCTURE
  - API keys valid?
  - Rate limits hit?
  - Network issues?
```

---

## Key Takeaways

1. **Agents fail predictably** - Learn the common patterns
2. **Logging is essential** - You can't fix what you can't see
3. **LangSmith transforms debugging** - From guessing to knowing
4. **Optimization is about trade-offs** - Speed vs quality vs cost
5. **Testing prevents regressions** - Catch issues before users do

---

## What's Next

Congratulations! You've completed Week 2. You can now:
- ✅ Build agents with LangChain
- ✅ Deploy them as production APIs
- ✅ Debug and optimize their behavior

In **Week 3**, we'll explore LangGraph for complex, multi-step workflows.

In **Week 4**, we'll connect agents to real data with APIs and RAG.

---

## Further Reading

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Debugging Guide](https://python.langchain.com/docs/guides/debugging)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
