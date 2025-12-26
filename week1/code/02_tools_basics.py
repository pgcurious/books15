"""
Module 1.2: Tools Basics
========================
Understanding how tools extend agent capabilities.
Tools are what transform an LLM from a "brain" to a "person with hands."
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
import json
import math
from datetime import datetime

load_dotenv()


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Use this for any mathematical calculations.

    Args:
        expression: A mathematical expression (e.g., "2 + 2", "sqrt(16)", "15 * 0.18")

    Returns:
        The result of the calculation as a string
    """
    try:
        # Safe evaluation with limited builtins
        allowed_names = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "pow": pow,
            "abs": abs,
            "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_datetime() -> str:
    """
    Get the current date and time.
    Use this when the user asks about the current date or time.

    Returns:
        Current date and time in a readable format
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a location.
    Use this when the user asks about weather conditions.

    Args:
        location: The city and optionally state/country (e.g., "San Francisco, CA")

    Returns:
        Weather information for the location
    """
    # Simulated weather data (in production, this would call a weather API)
    weather_data = {
        "san francisco": {"temp": 62, "condition": "Foggy", "humidity": 78},
        "new york": {"temp": 45, "condition": "Partly Cloudy", "humidity": 55},
        "miami": {"temp": 82, "condition": "Sunny", "humidity": 70},
        "london": {"temp": 48, "condition": "Rainy", "humidity": 85},
        "tokyo": {"temp": 58, "condition": "Clear", "humidity": 60},
    }

    location_key = location.lower().split(",")[0].strip()
    data = weather_data.get(location_key, {"temp": 70, "condition": "Unknown", "humidity": 50})

    return f"Weather in {location}: {data['temp']}°F, {data['condition']}, Humidity: {data['humidity']}%"


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the company knowledge base for information.
    Use this for policy questions, procedures, or company-specific information.

    Args:
        query: The search query describing what information is needed

    Returns:
        Relevant information from the knowledge base
    """
    # Simulated knowledge base
    kb = {
        "vacation": "Vacation Policy: Full-time employees receive 20 days PTO annually. PTO accrues monthly and unused days roll over up to 5 days.",
        "expense": "Expense Policy: Submit expenses within 30 days via ExpenseBot. Receipts required for amounts over $25. Manager approval needed for expenses over $500.",
        "remote": "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Core hours are 10am-3pm local time.",
        "benefits": "Benefits: Health insurance (medical, dental, vision), 401k with 4% match, life insurance, and professional development budget of $1500/year.",
        "meeting": "Meeting Room Policy: Book via Google Calendar. Maximum 2-hour reservations. Cancel bookings you won't use at least 1 hour in advance.",
    }

    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return value

    return "No specific information found. Please contact HR at hr@company.com for assistance."


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_tool_binding():
    """Demonstrate binding tools to an LLM."""
    print("=" * 60)
    print("DEMO 1: Tool Binding")
    print("=" * 60)

    # Create LLM and bind tools
    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [calculate, get_current_datetime, get_weather, search_knowledge_base]

    # Bind tools to LLM - this enables the LLM to call these tools
    llm_with_tools = llm.bind_tools(tools)

    print(f"LLM now has access to {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}: {t.description[:50]}...")

    print()
    return llm_with_tools, tools


def demo_tool_selection(llm_with_tools, tools):
    """Show how LLM selects appropriate tools based on query."""
    print("=" * 60)
    print("DEMO 2: Tool Selection (Emergence in Action)")
    print("=" * 60)

    test_queries = [
        "What's 18% tip on a $67.50 bill?",
        "What's the weather like in Tokyo?",
        "What's our company vacation policy?",
        "What time is it?",
        "Tell me a joke about programming",  # No tool needed
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        response = llm_with_tools.invoke(query)

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  → Tool Selected: {tc['name']}")
                print(f"    Arguments: {tc['args']}")
        else:
            print(f"  → No tool needed. Direct response.")
            print(f"    Response: {response.content[:100]}...")

    print()


def demo_tool_execution(llm_with_tools, tools):
    """Show complete tool execution cycle."""
    print("=" * 60)
    print("DEMO 3: Complete Tool Execution Cycle")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")
    query = "Calculate the compound interest on $10,000 at 5% for 3 years"

    print(f"Query: {query}\n")

    # Step 1: LLM decides to use a tool
    print("Step 1: LLM processes query and selects tool")
    response = llm_with_tools.invoke(query)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"  Tool: {tool_call['name']}")
        print(f"  Args: {tool_call['args']}")

        # Step 2: Execute the tool
        print("\nStep 2: Execute the tool")
        tool_map = {t.name: t for t in tools}
        tool_result = tool_map[tool_call['name']].invoke(tool_call['args'])
        print(f"  Result: {tool_result}")

        # Step 3: Send result back to LLM for final response
        print("\nStep 3: LLM synthesizes final response")
        messages = [
            HumanMessage(content=query),
            response,
            ToolMessage(content=tool_result, tool_call_id=tool_call['id'])
        ]
        final_response = llm.invoke(messages)
        print(f"  Final Response: {final_response.content}")

    print()


def demo_multi_tool():
    """Show how a query might require multiple tools."""
    print("=" * 60)
    print("DEMO 4: Multi-Tool Scenarios")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [calculate, get_current_datetime, get_weather]
    llm_with_tools = llm.bind_tools(tools)

    query = "What's the weather in Miami, and what's 20% of the temperature as a tip calculation example?"

    print(f"Query: {query}\n")
    print("(Note: LLM may request multiple tools or chain them)")

    response = llm_with_tools.invoke(query)

    if response.tool_calls:
        print(f"Tools requested: {len(response.tool_calls)}")
        for i, tc in enumerate(response.tool_calls, 1):
            print(f"  {i}. {tc['name']}: {tc['args']}")
    else:
        print(f"Response: {response.content}")

    print()


def demo_tool_schema():
    """Show the schema that LLM uses to understand tools."""
    print("=" * 60)
    print("DEMO 5: Tool Schema (What LLM Sees)")
    print("=" * 60)

    # Show the schema for the calculate tool
    schema = calculate.args_schema.model_json_schema()
    print("Schema for 'calculate' tool:")
    print(json.dumps(schema, indent=2))

    print("\nThis schema tells the LLM:")
    print("  - What arguments the tool accepts")
    print("  - What types those arguments should be")
    print("  - Description of what the tool does")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TOOLS BASICS - Extending Agent Capabilities")
    print("=" * 60 + "\n")

    llm_with_tools, tools = demo_tool_binding()
    demo_tool_selection(llm_with_tools, tools)
    demo_tool_execution(llm_with_tools, tools)
    demo_multi_tool()
    demo_tool_schema()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. Tools extend LLM capabilities beyond text generation
2. LLMs SELECT tools based on query context (emergence!)
3. The tool execution cycle: Query → Select Tool → Execute → Synthesize
4. Tool descriptions are critical - they guide LLM selection
5. Multiple tools can be chained for complex queries
    """)
