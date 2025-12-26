"""
Module 1.2: LLM Basics
======================
Understanding LLMs as the reasoning engine of agents.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY)
load_dotenv()


def demo_simple_invocation():
    """Basic LLM invocation - the foundation of everything."""
    print("=" * 60)
    print("DEMO 1: Simple LLM Invocation")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("What are the three laws of thermodynamics?")

    print(f"Response:\n{response.content}")
    print()


def demo_system_context():
    """Using system messages to shape LLM behavior."""
    print("=" * 60)
    print("DEMO 2: System Context (Shaping Behavior)")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Without context
    response_default = llm.invoke("Explain quantum entanglement")
    print("Default Response (first 200 chars):")
    print(response_default.content[:200] + "...")
    print()

    # With context - simplified explanation
    messages = [
        SystemMessage(content="You are a teacher explaining to a 10-year-old. Use simple words and fun analogies."),
        HumanMessage(content="Explain quantum entanglement")
    ]
    response_simple = llm.invoke(messages)
    print("With 'Explain to 10-year-old' context (first 300 chars):")
    print(response_simple.content[:300] + "...")
    print()


def demo_chain_of_thought():
    """Chain of Thought: How reasoning emerges from prompting."""
    print("=" * 60)
    print("DEMO 3: Chain of Thought Reasoning")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")

    problem = """
    A train travels at 60 mph for 2.5 hours, then speeds up to 80 mph for 1.5 hours.
    What is the total distance traveled?
    """

    # Without CoT
    response_direct = llm.invoke(problem + "\nAnswer directly.")
    print("Direct Answer:")
    print(response_direct.content)
    print()

    # With CoT
    response_cot = llm.invoke(problem + "\nThink through this step by step, showing your work.")
    print("Chain of Thought Answer:")
    print(response_cot.content)
    print()


def demo_temperature():
    """Temperature: Controlling randomness/creativity."""
    print("=" * 60)
    print("DEMO 4: Temperature Effects")
    print("=" * 60)

    prompt = "Generate a creative name for a coffee shop."

    # Low temperature - more deterministic
    llm_low = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    print("Temperature 0.1 (Deterministic):")
    for i in range(3):
        response = llm_low.invoke(prompt)
        print(f"  {i+1}. {response.content}")

    print()

    # High temperature - more creative/random
    llm_high = ChatOpenAI(model="gpt-4o-mini", temperature=1.2)
    print("Temperature 1.2 (Creative):")
    for i in range(3):
        response = llm_high.invoke(prompt)
        print(f"  {i+1}. {response.content}")

    print()


def demo_token_limits():
    """Understanding token limits and context windows."""
    print("=" * 60)
    print("DEMO 5: Token Limits")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=50)

    response = llm.invoke("Write a detailed essay about the history of artificial intelligence.")
    print("With max_tokens=50:")
    print(response.content)
    print()

    llm_longer = ChatOpenAI(model="gpt-4o-mini", max_tokens=200)
    response_longer = llm_longer.invoke("Write a detailed essay about the history of artificial intelligence.")
    print("With max_tokens=200:")
    print(response_longer.content)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LLM BASICS - Understanding the Reasoning Engine")
    print("=" * 60 + "\n")

    demo_simple_invocation()
    demo_system_context()
    demo_chain_of_thought()
    demo_temperature()
    demo_token_limits()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. LLMs are token predictors - everything else emerges from this
2. System messages shape behavior without changing the model
3. Chain of Thought prompting unlocks reasoning capabilities
4. Temperature controls creativity vs determinism
5. Token limits affect response length and cost
    """)
