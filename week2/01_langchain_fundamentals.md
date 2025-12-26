# Module 2.1: Hands-on Coding with LangChain

## What You'll Learn
- LangChain's core abstractions and philosophy
- LangChain Expression Language (LCEL) for composable chains
- Building tools and binding them to LLMs
- Implementing conversation memory
- Creating your first production-ready agent

---

## First Principles: What Is LangChain?

At its core, LangChain is a **composition framework for LLM applications**.

**First Principle #1:** Complex behavior = Simple components + Smart composition

```python
# This simple composition...
chain = prompt | llm | parser

# ...handles all of this automatically:
# - Format the prompt with variables
# - Send to LLM and handle API specifics
# - Parse the response into structured data
# - Handle errors, retries, streaming
```

**First Principle #2:** Everything is a Runnable

A `Runnable` is anything that can be:
- Invoked with input
- Batched over multiple inputs
- Streamed for real-time output
- Composed with other Runnables

---

## The Analogy: LangChain as LEGO

Think of LangChain components as LEGO blocks:

| LEGO | LangChain |
|------|-----------|
| Individual bricks | Components (prompts, LLMs, tools) |
| Connecting studs | The `|` pipe operator |
| Building instructions | Your chain definition |
| Final creation | Your working agent |

Each piece snaps together in standardized ways, and you can build anything by combining simple pieces.

---

## Part 1: LangChain Expression Language (LCEL)

### The Pipe Operator: `|`

The pipe operator chains components together. Data flows left to right.

```python
# code/01_lcel_basics.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create individual components
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Compose them with the pipe operator
chain = prompt | llm | parser

# Use the chain
result = chain.invoke({"topic": "programming"})
print(result)
```

### What Happens Under the Hood

```
Input: {"topic": "programming"}
       │
       ▼
┌──────────────────────┐
│   ChatPromptTemplate │
│   "Tell me a joke    │
│    about {topic}"    │
└──────────────────────┘
       │
       ▼ ChatPromptValue("Tell me a joke about programming")
       │
┌──────────────────────┐
│      ChatOpenAI      │
│   (API call to GPT)  │
└──────────────────────┘
       │
       ▼ AIMessage(content="Why do programmers...")
       │
┌──────────────────────┐
│   StrOutputParser    │
│ (Extract text only)  │
└──────────────────────┘
       │
       ▼
Output: "Why do programmers prefer dark mode?..."
```

### Core Runnable Methods

Every Runnable supports these methods:

```python
# Synchronous invocation
result = chain.invoke({"topic": "AI"})

# Batch processing (parallel by default)
results = chain.batch([
    {"topic": "AI"},
    {"topic": "Python"},
    {"topic": "Databases"}
])

# Streaming (real-time output)
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Async versions
result = await chain.ainvoke({"topic": "AI"})
results = await chain.abatch([...])
async for chunk in chain.astream(...):
    print(chunk)
```

---

## Part 2: Building Blocks Deep Dive

### Prompt Templates

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Simple template
simple = ChatPromptTemplate.from_template("Translate to French: {text}")

# Multi-message template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    ("human", "{question}")
])

# With message history placeholder
with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Usage
result = chat_prompt.invoke({
    "domain": "Python programming",
    "question": "How do I read a file?"
})
```

### Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from pydantic import BaseModel, Field

# String parser (most common)
str_parser = StrOutputParser()

# JSON parser
json_parser = JsonOutputParser()

# Structured output with Pydantic
class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief review summary")

pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)

# The parser provides format instructions for the prompt
format_instructions = pydantic_parser.get_format_instructions()
```

### RunnableLambda: Custom Logic

```python
from langchain_core.runnables import RunnableLambda

# Wrap any function as a Runnable
def uppercase(text: str) -> str:
    return text.upper()

uppercase_runnable = RunnableLambda(uppercase)

# Use in a chain
chain = prompt | llm | StrOutputParser() | uppercase_runnable
```

### RunnablePassthrough: Forwarding Data

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Pass data through unchanged
chain = RunnablePassthrough() | llm

# Parallel execution with passthrough
chain = RunnableParallel(
    response=prompt | llm | StrOutputParser(),
    original_input=RunnablePassthrough()
)

result = chain.invoke({"topic": "AI"})
# result = {"response": "...", "original_input": {"topic": "AI"}}
```

---

## Part 3: Tools

### Defining Tools

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information.

    Args:
        query: The search term to look up on Wikipedia

    Returns:
        A summary of the Wikipedia article
    """
    # In production, use the actual Wikipedia API
    return f"Wikipedia result for '{query}': [Summary would go here]"

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A math expression like '2 + 2' or 'sqrt(16)'

    Returns:
        The result of the calculation
    """
    import math
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(
    city: str,
    unit: Optional[str] = "celsius"
) -> str:
    """
    Get current weather for a city.

    Args:
        city: The city name
        unit: Temperature unit ('celsius' or 'fahrenheit')

    Returns:
        Current weather conditions
    """
    # Mock response
    return f"Weather in {city}: 22°{'C' if unit == 'celsius' else 'F'}, Partly cloudy"
```

### Tool Structure (Emergence Insight)

When you define a tool with `@tool`, LangChain automatically extracts:

```python
# What the LLM sees:
print(search_wikipedia.name)        # "search_wikipedia"
print(search_wikipedia.description) # "Search Wikipedia for information..."
print(search_wikipedia.args_schema.schema())
# {
#   "properties": {
#     "query": {"type": "string", "description": "The search term..."}
#   },
#   "required": ["query"]
# }
```

The LLM uses this schema to:
1. Understand when the tool is relevant
2. Know what arguments to provide
3. Format the function call correctly

**Emergence:** Good tool descriptions lead to better tool selection!

### Binding Tools to LLMs

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_wikipedia, calculate, get_weather]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Now the LLM can choose to call tools
response = llm_with_tools.invoke("What's the weather in Paris?")

# Check if tools were called
if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call['name']}, Args: {call['args']}")
```

---

## Part 4: Memory

### Conversation Memory with LCEL

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Store for session histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create prompt with history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the base chain
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm

# Wrap with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Use with session management
config = {"configurable": {"session_id": "user-123"}}

response1 = chain_with_history.invoke(
    {"input": "My name is Alice"},
    config=config
)

response2 = chain_with_history.invoke(
    {"input": "What's my name?"},
    config=config
)
print(response2.content)  # Should reference Alice!
```

---

## Part 5: Building Your First Complete Agent

Now let's put it all together into a complete, production-ready agent.

```python
# code/05_complete_agent.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Dict, List, Any
import json

# =============================================================================
# TOOLS
# =============================================================================

@tool
def search_web(query: str) -> str:
    """
    Search the web for current information.
    Use this when you need up-to-date information or facts.

    Args:
        query: The search query
    """
    # Mock - in production, use DuckDuckGo, Tavily, or similar
    return f"Search results for '{query}': Found relevant information about {query}."

@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    Use this for any math calculations.

    Args:
        expression: Math expression (e.g., '15 * 0.20' for tip calculation)
    """
    try:
        import math
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_date_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

# =============================================================================
# AGENT CLASS
# =============================================================================

class ResearchAssistant:
    """
    A research assistant agent built with LangChain.

    Architecture:
    - LLM: GPT-4o-mini for reasoning
    - Tools: Search, calculate, datetime
    - Memory: Conversation history per session
    - Pattern: ReAct (Reason + Act)
    """

    def __init__(self):
        # Initialize LLM with tools
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.tools = [search_web, calculate, get_date_time]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Session store for memory
        self.sessions: Dict[str, ChatMessageHistory] = {}

        # System prompt
        self.system_prompt = """You are a helpful research assistant with access to tools.

Available tools:
- search_web: Search for current information
- calculate: Perform mathematical calculations
- get_date_time: Get current date and time

Guidelines:
1. Use tools when they would help answer the question
2. Think step by step for complex questions
3. Be concise but thorough
4. If you're unsure, say so
5. Always cite when using search results

Remember: You're having a conversation. Reference past context when relevant."""

    def _get_session(self, session_id: str) -> ChatMessageHistory:
        """Get or create session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    def _execute_tools(self, tool_calls: List[Dict]) -> List[ToolMessage]:
        """Execute tool calls and return results."""
        results = []
        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]

            if tool_name in self.tool_map:
                result = self.tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            results.append(ToolMessage(
                content=str(result),
                tool_call_id=call["id"]
            ))
        return results

    def chat(self, message: str, session_id: str = "default") -> str:
        """
        Process a user message and return a response.

        This implements the agent loop:
        1. Get context (memory)
        2. Reason (LLM)
        3. Act (tools)
        4. Synthesize response
        5. Update memory
        """
        session = self._get_session(session_id)

        # Build messages with history
        messages = [
            {"role": "system", "content": self.system_prompt},
            *[{"role": m.type, "content": m.content}
              for m in session.messages],
            {"role": "user", "content": message}
        ]

        # Get LLM response (may include tool calls)
        response = self.llm_with_tools.invoke(messages)

        # Handle tool calls if present
        if response.tool_calls:
            # Add assistant message with tool calls
            messages.append(response)

            # Execute tools
            tool_results = self._execute_tools(response.tool_calls)

            # Add tool results
            for result in tool_results:
                messages.append(result)

            # Get final response
            final_response = self.llm.invoke(messages)
            response_text = final_response.content
        else:
            response_text = response.content

        # Update memory
        session.add_user_message(message)
        session.add_ai_message(response_text)

        return response_text

    def clear_session(self, session_id: str = "default"):
        """Clear a session's history."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESEARCH ASSISTANT DEMO")
    print("=" * 60)

    assistant = ResearchAssistant()

    # Demo conversation
    demo_messages = [
        "Hi! What time is it?",
        "Can you calculate a 20% tip on $85.50?",
        "Search for recent news about AI agents",
        "Based on what you found, what are the main trends?",
    ]

    for message in demo_messages:
        print(f"\nUser: {message}")
        response = assistant.chat(message, session_id="demo")
        print(f"Assistant: {response}")
        print("-" * 40)
```

---

## Exercise: Extend the Agent

Try these enhancements:

1. **Add a new tool**: Create a `summarize_url` tool that fetches and summarizes web content

2. **Add streaming**: Modify the agent to stream responses token by token

3. **Add tool confirmation**: Before executing dangerous tools, ask for user confirmation

4. **Add error recovery**: If a tool fails, have the agent try an alternative approach

---

## Key Takeaways

1. **LCEL is the foundation**: The pipe operator creates composable, streamable chains
2. **Everything is a Runnable**: Prompts, LLMs, parsers, tools—all composable
3. **Tools extend capabilities**: Well-documented tools enable emergent selection
4. **Memory enables continuity**: Session-based history creates conversational agents
5. **The agent pattern**: Context → Reason → Act → Synthesize → Remember

---

## Next Steps

In [Module 2.2: Deploy Your First Working Agent](02_deploy_your_agent.md), we'll:
- Deploy this agent as a web service
- Create an interactive playground
- Handle production concerns
