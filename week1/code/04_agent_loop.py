"""
Module 1.2: The Complete Agent Loop
===================================
Bringing together LLM + Tools + Memory into a cohesive agent.
This is where emergence happens - simple components creating sophisticated behavior.
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Optional
import json
import math

load_dotenv()


# =============================================================================
# MEMORY SYSTEM
# =============================================================================

class AgentMemory:
    """
    Combined memory system for the agent.

    Emergence Insight: Memory doesn't just store data - it enables the agent
    to appear to "learn" and "grow" over time without any model changes.
    """

    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.facts: Dict[str, str] = {}
        self.interaction_count: int = 0

    def add_interaction(self, user_input: str, agent_response: str, tools_used: List[str] = None):
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "agent": agent_response,
            "tools": tools_used or []
        })
        self.interaction_count += 1

    def store_fact(self, key: str, value: str):
        """Store a fact for later retrieval."""
        self.facts[key] = value
        print(f"  [Memory] Stored: {key} = {value}")

    def recall_fact(self, key: str) -> Optional[str]:
        """Recall a stored fact."""
        return self.facts.get(key)

    def get_recent_context(self, n: int = 5) -> str:
        """Get recent conversation as context string."""
        recent = self.conversation_history[-n:] if self.conversation_history else []
        if not recent:
            return "No previous conversation."

        context_parts = []
        for interaction in recent:
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"Assistant: {interaction['agent']}")
        return "\n".join(context_parts)

    def get_facts_context(self) -> str:
        """Get all facts as context string."""
        if not self.facts:
            return "No stored facts."
        return "\n".join([f"- {k}: {v}" for k, v in self.facts.items()])


# =============================================================================
# TOOLS
# =============================================================================

@tool
def get_current_time() -> str:
    """
    Get the current date and time.
    Use when the user asks about the current time or date.
    """
    now = datetime.now()
    return f"Current time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for any calculations, math problems, or number computations.

    Args:
        expression: Math expression like "2 + 2" or "sqrt(16) * 5"
    """
    try:
        allowed = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "pi": math.pi, "e": math.e,
            "log": math.log, "pow": pow, "abs": abs, "round": round
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information.
    Use when you need up-to-date information or facts you're unsure about.

    Args:
        query: The search query
    """
    # Simulated search results (in production, use an actual search API)
    mock_results = {
        "weather": "Current weather varies by location. Check weather.com for specifics.",
        "news": "Latest headlines: Tech stocks rise, Climate summit begins, Sports finals tonight.",
        "python": "Python 3.12 is the latest stable release. Popular for AI/ML development.",
        "ai agents": "AI agents are autonomous systems using LLMs with tools and memory.",
    }

    for key, result in mock_results.items():
        if key in query.lower():
            return f"Search results for '{query}': {result}"

    return f"Search results for '{query}': Multiple relevant results found. Refine your query for specifics."


# Memory-connected tools (these interact with the agent's memory)
def create_memory_tools(memory: AgentMemory):
    """Create tools that have access to agent memory."""

    @tool
    def remember(key: str, value: str) -> str:
        """
        Remember a piece of information for later.
        Use when the user tells you something important to remember.

        Args:
            key: A short identifier (e.g., "user_name", "favorite_color")
            value: The information to remember
        """
        memory.store_fact(key, value)
        return f"I'll remember that {key} is {value}."

    @tool
    def recall(key: str) -> str:
        """
        Recall a previously remembered piece of information.

        Args:
            key: The identifier of the information to recall
        """
        value = memory.recall_fact(key)
        if value:
            return f"I remember: {key} = {value}"
        return f"I don't have any information stored for '{key}'."

    return [remember, recall]


# =============================================================================
# THE AGENT
# =============================================================================

class SimpleAgent:
    """
    A complete agent with LLM, Tools, and Memory.

    First Principles:
    - LLM provides reasoning
    - Tools provide actions
    - Memory provides context

    Emergence:
    - Goal-directed behavior emerges from the loop
    - Learning emerges from memory accumulation
    - Personality emerges from the system prompt
    """

    def __init__(self, name: str = "Atlas"):
        self.name = name
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.memory = AgentMemory()

        # Create tools (including memory-connected ones)
        memory_tools = create_memory_tools(self.memory)
        self.tools = [get_current_time, calculate, search_web] + memory_tools
        self.tool_map = {t.name: t for t in self.tools}

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # System prompt (this shapes the agent's "personality")
        self.system_prompt = """You are {name}, a helpful AI assistant with access to tools and memory.

Your capabilities:
- get_current_time: Check the current date/time
- calculate: Perform mathematical calculations
- search_web: Search for current information
- remember: Store important information about the user
- recall: Retrieve stored information

Guidelines:
1. Be helpful, concise, and friendly
2. Use tools when they would help answer the question
3. Remember important facts the user tells you (name, preferences, etc.)
4. Reference stored facts to personalize responses
5. If unsure, search for information rather than guessing

Known facts about the user:
{facts}

Recent conversation context:
{context}
"""

    def _build_system_message(self) -> str:
        """Build the system prompt with current context."""
        return self.system_prompt.format(
            name=self.name,
            facts=self.memory.get_facts_context(),
            context=self.memory.get_recent_context()
        )

    def _execute_tool(self, tool_call: dict) -> str:
        """Execute a tool and return its result."""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"  [Tool] Calling: {tool_name}({tool_args})")

        if tool_name in self.tool_map:
            result = self.tool_map[tool_name].invoke(tool_args)
            print(f"  [Tool] Result: {result}")
            return result
        else:
            return f"Error: Unknown tool {tool_name}"

    def run(self, user_input: str) -> str:
        """
        The Agent Loop: Perceive → Reason → Act → Learn

        This is where emergence happens. Watch how simple components
        combine to create sophisticated, goal-directed behavior.
        """
        print(f"\n{'='*50}")
        print(f"User: {user_input}")
        print(f"{'='*50}")

        # Step 1: PERCEIVE - Gather context
        print("\n[1. PERCEIVE] Building context from memory...")
        messages = [
            SystemMessage(content=self._build_system_message()),
            HumanMessage(content=user_input)
        ]

        # Step 2: REASON - LLM decides what to do
        print("[2. REASON] LLM processing...")
        response = self.llm_with_tools.invoke(messages)

        tools_used = []

        # Step 3: ACT - Execute any tools the LLM requested
        if response.tool_calls:
            print(f"[3. ACT] Executing {len(response.tool_calls)} tool(s)...")

            # Execute each tool and collect results
            messages.append(response)

            for tool_call in response.tool_calls:
                tool_result = self._execute_tool(tool_call)
                tools_used.append(tool_call["name"])

                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))

            # Get final response after tool execution
            print("[3. ACT] Getting final response...")
            final_response = self.llm.invoke(messages)
            agent_response = final_response.content
        else:
            print("[3. ACT] No tools needed, using direct response")
            agent_response = response.content

        # Step 4: LEARN - Update memory
        print("[4. LEARN] Updating memory...")
        self.memory.add_interaction(user_input, agent_response, tools_used)

        print(f"\n{self.name}: {agent_response}")
        return agent_response


# =============================================================================
# DEMO
# =============================================================================

def run_demo():
    """Demonstrate the complete agent loop."""
    print("\n" + "=" * 60)
    print("THE COMPLETE AGENT LOOP")
    print("Demonstrating Emergence from LLM + Tools + Memory")
    print("=" * 60)

    agent = SimpleAgent(name="Atlas")

    # Demo conversation showing different capabilities
    demo_queries = [
        # Memory
        "Hi! My name is Sam and I'm a data scientist.",

        # Tool use (calculate)
        "What's 15% tip on a $84.50 dinner bill?",

        # Tool use (time)
        "What's the current time?",

        # Memory recall (should remember name)
        "What do you remember about me?",

        # Tool use (search)
        "Search for information about AI agents.",

        # Combining context
        "Based on what you know about me, what AI topics might interest me?",
    ]

    for query in demo_queries:
        agent.run(query)
        print("\n" + "-" * 60)

    # Final summary
    print("\n" + "=" * 60)
    print("AGENT SESSION SUMMARY")
    print("=" * 60)
    print(f"Total interactions: {agent.memory.interaction_count}")
    print(f"\nStored facts:")
    print(agent.memory.get_facts_context())
    print(f"\nTools used across session:")
    all_tools = []
    for interaction in agent.memory.conversation_history:
        all_tools.extend(interaction.get("tools", []))
    print(f"  {set(all_tools) if all_tools else 'None'}")


if __name__ == "__main__":
    run_demo()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM THE AGENT LOOP")
    print("=" * 60)
    print("""
The Emergence Pattern:
----------------------
1. PERCEIVE: Agent gathers context from memory and input
2. REASON: LLM decides what to do (use tools? respond directly?)
3. ACT: Execute tools and synthesize response
4. LEARN: Update memory for future interactions

What Emerges:
-------------
- Goal-directed behavior (answering questions effectively)
- Personalization (using stored facts about user)
- Tool selection intelligence (knowing when to use which tool)
- Conversational coherence (maintaining context)
- Apparent learning (improving responses with more context)

None of these behaviors are explicitly programmed.
They EMERGE from the interaction of simple components.

This is the power of agentic AI.
    """)
