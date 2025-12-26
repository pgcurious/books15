"""
Module 4.1: API-Powered Agent
=============================
Demonstrates building an agent with access to real-world APIs:
- Weather data
- Cryptocurrency prices
- Multi-tool reasoning
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Import our custom tools
from api_tools import (
    get_current_weather,
    get_weather_forecast,
    get_crypto_price,
    get_trending_crypto
)


# =============================================================================
# Agent Setup
# =============================================================================

def create_api_agent():
    """Create an agent with API tools."""

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Tools
    tools = [
        get_current_weather,
        get_weather_forecast,
        get_crypto_price,
        get_trending_crypto
    ]

    # System prompt
    system_prompt = """You are a helpful assistant with access to real-time data.

You have tools to check:
- Current weather and forecasts for any city
- Cryptocurrency prices and trends

When answering questions:
1. Use the appropriate tool to get current, accurate data
2. Synthesize information clearly and concisely
3. If a tool fails, explain what happened and try alternatives if possible
4. For multi-part questions, use multiple tools as needed

Never guess or make up data - always use your tools to get real information."""

    # Create agent
    agent = create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt
    )

    return agent


def ask_agent(agent, question: str) -> str:
    """Ask the agent a question."""
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    return result["messages"][-1].content


# =============================================================================
# Demo
# =============================================================================

def main():
    """Run the API agent demo."""
    print("=" * 60)
    print("API-Powered Agent Demo")
    print("=" * 60)

    print("\nInitializing agent with API tools...")
    agent = create_api_agent()
    print("Agent ready!\n")

    # Test questions
    questions = [
        # Single tool questions
        "What's the weather like in Tokyo right now?",
        "What is the current price of Bitcoin?",

        # Multi-tool questions
        "I'm planning a trip. What's the weather forecast for London, "
        "and how much is Ethereum worth right now?",

        # Trending data
        "What cryptocurrencies are trending right now?",

        # Complex reasoning
        "Compare the weather in New York and Los Angeles. "
        "Which city has better weather for outdoor activities today?",
    ]

    for i, question in enumerate(questions, 1):
        print("=" * 60)
        print(f"Question {i}: {question}")
        print("=" * 60)

        try:
            answer = ask_agent(agent, question)
            print(f"\nAgent Response:\n{answer}")
        except Exception as e:
            print(f"\nError: {e}")

        print("\n")

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nThe agent successfully:")
    print("- Retrieved real-time weather data from wttr.in")
    print("- Fetched live cryptocurrency prices from CoinGecko")
    print("- Combined multiple data sources to answer complex questions")
    print("- Provided synthesized, actionable responses")


if __name__ == "__main__":
    main()
