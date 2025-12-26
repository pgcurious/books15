"""
Module 2.3: Debugging & Optimization Examples
=============================================
Techniques for debugging and optimizing AI agents.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain.globals import set_debug, set_verbose
from dotenv import load_dotenv
import logging
import time
import json
from typing import List, Dict, Any
from datetime import datetime

load_dotenv()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(level=logging.DEBUG):
    """Configure logging for debugging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('agent_debug.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# =============================================================================
# DEBUGGABLE AGENT
# =============================================================================

class DebuggableAgent:
    """An agent with built-in debugging capabilities."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.tools = self._create_tools()
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.call_log = []

    def _create_tools(self):
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for '{query}': [Mock data]"

        @tool
        def calculate(expression: str) -> str:
            """Calculate a math expression."""
            try:
                import math
                result = eval(expression, {"__builtins__": {}}, {"math": math})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"

        return [search, calculate]

    def _log(self, step: str, data: Any):
        """Log a step with data."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": str(data)[:500]  # Truncate for readability
        }
        self.call_log.append(entry)

        if self.verbose:
            logger.info(f"[{step}] {entry['data']}")

    def run_with_debugging(self, message: str) -> Dict[str, Any]:
        """Run with full debugging information."""
        start_time = time.time()
        self._log("INPUT", message)

        messages = [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": message}
        ]

        # Track iterations for loop detection
        iterations = 0
        max_iterations = 10
        tools_called = []

        while iterations < max_iterations:
            iterations += 1
            self._log("ITERATION", f"Starting iteration {iterations}")

            response = self.llm_with_tools.invoke(messages)
            self._log("LLM_RESPONSE", {
                "content": response.content[:200] if response.content else None,
                "tool_calls": len(response.tool_calls) if response.tool_calls else 0
            })

            if not response.tool_calls:
                # No more tool calls - we're done
                break

            # Execute tools
            messages.append(response)

            for call in response.tool_calls:
                self._log("TOOL_CALL", {"name": call["name"], "args": call["args"]})
                tools_called.append(call["name"])

                if call["name"] in self.tool_map:
                    result = self.tool_map[call["name"]].invoke(call["args"])
                else:
                    result = f"Unknown tool: {call['name']}"

                self._log("TOOL_RESULT", result)
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        # Get final response if we exited due to tool calls
        if response.tool_calls:
            response = self.llm.invoke(messages)

        elapsed_time = time.time() - start_time

        result = {
            "response": response.content,
            "iterations": iterations,
            "tools_called": tools_called,
            "elapsed_time": elapsed_time,
            "call_log": self.call_log.copy()
        }

        self._log("OUTPUT", {
            "response_length": len(response.content),
            "elapsed_time": elapsed_time
        })

        return result

    def get_debug_report(self) -> str:
        """Generate a debug report from the call log."""
        report = ["=" * 50, "DEBUG REPORT", "=" * 50, ""]

        for entry in self.call_log:
            report.append(f"[{entry['timestamp']}] {entry['step']}")
            report.append(f"  {entry['data']}")
            report.append("")

        return "\n".join(report)

    def clear_log(self):
        """Clear the call log."""
        self.call_log = []


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor and analyze agent performance."""

    def __init__(self):
        self.metrics = []

    def record(self, name: str, duration: float, tokens: int = 0, success: bool = True):
        """Record a metric."""
        self.metrics.append({
            "name": name,
            "duration": duration,
            "tokens": tokens,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {"message": "No metrics recorded"}

        durations = [m["duration"] for m in self.metrics]
        tokens = [m["tokens"] for m in self.metrics if m["tokens"] > 0]
        success_rate = sum(1 for m in self.metrics if m["success"]) / len(self.metrics)

        return {
            "total_calls": len(self.metrics),
            "success_rate": f"{success_rate:.1%}",
            "avg_duration": f"{sum(durations)/len(durations):.3f}s",
            "max_duration": f"{max(durations):.3f}s",
            "min_duration": f"{min(durations):.3f}s",
            "total_tokens": sum(tokens) if tokens else 0
        }


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_tool_directly():
    """Test individual tools without the agent."""
    print("=" * 50)
    print("TESTING TOOLS DIRECTLY")
    print("=" * 50)

    @tool
    def calculate(expression: str) -> str:
        """Calculate a math expression."""
        try:
            import math
            result = eval(expression, {"__builtins__": {}}, {"math": math})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    # Test cases
    test_cases = [
        ("2 + 2", "4"),
        ("10 / 3", "3.33"),
        ("math.sqrt(16)", "4"),
        ("invalid expression", "Error"),
    ]

    for expression, expected in test_cases:
        result = calculate.invoke({"expression": expression})
        status = "✓" if expected in result else "✗"
        print(f"{status} calculate('{expression}') -> {result}")


def test_prompt_variations():
    """Test different prompt variations."""
    print("=" * 50)
    print("TESTING PROMPT VARIATIONS")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompts = [
        # Vague prompt
        ("vague", "Be helpful. Answer: {question}"),
        # Specific prompt
        ("specific", "You are a math tutor. Give step-by-step solutions. Answer: {question}"),
        # Constrained prompt
        ("constrained", "Answer in exactly one sentence: {question}"),
    ]

    question = "What is 15% of 200?"

    for name, template in prompts:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question})
        print(f"\n{name.upper()} PROMPT:")
        print(f"Result: {result[:200]}...")


# =============================================================================
# LANGCHAIN DEBUG MODE
# =============================================================================

def demo_langchain_debug_mode():
    """Demonstrate LangChain's built-in debug mode."""
    print("=" * 50)
    print("LANGCHAIN DEBUG MODE")
    print("=" * 50)

    # Enable verbose mode (shows prompts and responses)
    set_verbose(True)

    # Enable debug mode (shows everything)
    # set_debug(True)  # Uncomment for maximum verbosity

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("Explain {topic} briefly")
    chain = prompt | llm | StrOutputParser()

    print("\nRunning with verbose=True:")
    result = chain.invoke({"topic": "recursion"})
    print(f"\nFinal result: {result}")

    # Reset
    set_verbose(False)
    set_debug(False)


# =============================================================================
# DEMO
# =============================================================================

def run_demo():
    """Run debugging demonstrations."""
    print("\n" + "=" * 60)
    print("DEBUGGING & OPTIMIZATION DEMO")
    print("=" * 60 + "\n")

    # Test 1: Direct tool testing
    test_tool_directly()
    print()

    # Test 2: Prompt variations
    test_prompt_variations()
    print()

    # Test 3: Debuggable agent
    print("=" * 50)
    print("DEBUGGABLE AGENT")
    print("=" * 50)

    agent = DebuggableAgent(verbose=True)

    result = agent.run_with_debugging("What is 25% of 80?")

    print(f"\nFinal Response: {result['response']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tools Called: {result['tools_called']}")
    print(f"Elapsed Time: {result['elapsed_time']:.3f}s")
    print()

    # Test 4: Performance monitoring
    print("=" * 50)
    print("PERFORMANCE MONITORING")
    print("=" * 50)

    monitor = PerformanceMonitor()

    # Simulate some calls
    for i in range(5):
        start = time.time()
        # Simulate work
        time.sleep(0.1)
        monitor.record(f"call_{i}", time.time() - start, tokens=100)

    print(json.dumps(monitor.get_summary(), indent=2))


if __name__ == "__main__":
    run_demo()
