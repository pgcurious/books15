"""
Module 2.1: LCEL Basics
=======================
LangChain Expression Language - the foundation of composable chains.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()


def demo_basic_chain():
    """The simplest chain: prompt | llm | parser."""
    print("=" * 60)
    print("DEMO 1: Basic Chain (prompt | llm | parser)")
    print("=" * 60)

    # Components
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    llm = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | llm | parser

    # Use the chain
    result = chain.invoke({"topic": "programming"})
    print(f"Result: {result}")
    print()


def demo_chain_steps():
    """Show what happens at each step."""
    print("=" * 60)
    print("DEMO 2: Understanding Chain Steps")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    llm = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()

    # Step by step
    input_data = {"text": "Hello, how are you?"}
    print(f"Input: {input_data}")

    # Step 1: Prompt
    prompt_result = prompt.invoke(input_data)
    print(f"\nAfter Prompt: {prompt_result}")

    # Step 2: LLM
    llm_result = llm.invoke(prompt_result)
    print(f"\nAfter LLM: {llm_result}")

    # Step 3: Parser
    final_result = parser.invoke(llm_result)
    print(f"\nAfter Parser: {final_result}")
    print()


def demo_runnable_lambda():
    """Custom functions as chain components."""
    print("=" * 60)
    print("DEMO 3: RunnableLambda (Custom Functions)")
    print("=" * 60)

    # Custom processing functions
    def add_context(text: str) -> str:
        return f"[Context: Technical discussion]\n{text}"

    def format_output(text: str) -> str:
        return f"📝 {text.strip()}"

    # Create runnables
    add_context_runnable = RunnableLambda(add_context)
    format_runnable = RunnableLambda(format_output)

    # Build chain
    prompt = ChatPromptTemplate.from_template("{input}")
    llm = ChatOpenAI(model="gpt-4o-mini")

    chain = add_context_runnable | prompt | llm | StrOutputParser() | format_runnable

    result = chain.invoke("Explain recursion briefly")
    print(result)
    print()


def demo_parallel_chains():
    """Running multiple chains in parallel."""
    print("=" * 60)
    print("DEMO 4: Parallel Chains")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create parallel chains
    joke_chain = ChatPromptTemplate.from_template("Tell a joke about {topic}") | llm | StrOutputParser()
    fact_chain = ChatPromptTemplate.from_template("Tell a fact about {topic}") | llm | StrOutputParser()

    # Combine in parallel
    parallel_chain = RunnableParallel(
        joke=joke_chain,
        fact=fact_chain
    )

    result = parallel_chain.invoke({"topic": "cats"})
    print(f"Joke: {result['joke']}")
    print(f"\nFact: {result['fact']}")
    print()


def demo_passthrough():
    """Passing data through while processing."""
    print("=" * 60)
    print("DEMO 5: RunnablePassthrough")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Chain that processes AND keeps original input
    chain = RunnableParallel(
        original=RunnablePassthrough(),
        processed=ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser()
    )

    result = chain.invoke({"text": "AI is transforming software development in many ways."})
    print(f"Original: {result['original']}")
    print(f"Processed: {result['processed']}")
    print()


def demo_branching():
    """Conditional routing in chains."""
    print("=" * 60)
    print("DEMO 6: Branching/Routing")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Different prompts for different needs
    technical_prompt = ChatPromptTemplate.from_template(
        "Give a technical explanation of: {question}"
    )
    simple_prompt = ChatPromptTemplate.from_template(
        "Explain like I'm 10 years old: {question}"
    )

    def route(input_dict):
        """Route based on complexity preference."""
        if input_dict.get("style") == "technical":
            return technical_prompt
        return simple_prompt

    # Create routing chain
    chain = RunnableLambda(route) | llm | StrOutputParser()

    # Test both routes
    print("Technical style:")
    print(chain.invoke({"question": "What is recursion?", "style": "technical"}))

    print("\nSimple style:")
    print(chain.invoke({"question": "What is recursion?", "style": "simple"}))
    print()


def demo_batch_and_stream():
    """Batch processing and streaming."""
    print("=" * 60)
    print("DEMO 7: Batch & Stream")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
    chain = prompt | llm | StrOutputParser()

    # Batch processing
    print("Batch processing:")
    countries = [{"country": "France"}, {"country": "Japan"}, {"country": "Brazil"}]
    results = chain.batch(countries)
    for country, result in zip(countries, results):
        print(f"  {country['country']}: {result}")

    # Streaming
    print("\nStreaming (token by token):")
    print("  ", end="")
    for chunk in chain.stream({"country": "Italy"}):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LCEL BASICS - Composable Chains")
    print("=" * 60 + "\n")

    demo_basic_chain()
    demo_chain_steps()
    demo_runnable_lambda()
    demo_parallel_chains()
    demo_passthrough()
    demo_branching()
    demo_batch_and_stream()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. The pipe operator (|) composes components left to right
2. Every component is a Runnable with invoke/batch/stream
3. RunnableLambda wraps any function as a chain component
4. RunnableParallel executes multiple chains concurrently
5. RunnablePassthrough forwards data through the chain
6. Chains are composable - build complex from simple
    """)
