"""
Module 2.1: Complete Agent Implementation
=========================================
A production-ready research assistant agent built with LangChain.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
import json
import math

load_dotenv()


# =============================================================================
# TOOLS - The agent's capabilities
# =============================================================================

@tool
def search_web(query: str) -> str:
    """
    Search the web for current information.
    Use this when you need up-to-date information, news, or facts you're unsure about.

    Args:
        query: The search query (be specific for better results)

    Returns:
        Search results as a formatted string
    """
    # Mock implementation - in production, use DuckDuckGo, Tavily, or similar
    mock_results = {
        "ai": "Recent AI developments include GPT-4o, Claude 3.5, and advances in multimodal models.",
        "weather": "Weather information requires real-time data. Please specify a location.",
        "python": "Python 3.12 is the latest stable release. Key features include better error messages.",
        "langchain": "LangChain is a framework for building LLM applications. Latest version has LCEL.",
    }

    for key, result in mock_results.items():
        if key in query.lower():
            return f"Search results for '{query}': {result}"

    return f"Search results for '{query}': Multiple sources found. Please refine your query."


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use this for calculations, conversions, or any math operations.

    Args:
        expression: Math expression (e.g., '2 + 2', 'sqrt(16)', '15 * 0.18')

    Returns:
        The calculation result
    """
    try:
        allowed_names = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
            "pow": pow,
            "abs": abs,
            "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}. Please check the expression format."


@tool
def get_current_datetime() -> str:
    """
    Get the current date and time.
    Use when the user asks about today's date or current time.

    Returns:
        Current date and time in a readable format
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


@tool
def analyze_text(text: str, analysis_type: str = "summary") -> str:
    """
    Analyze or transform text.
    Use for summarization, sentiment analysis, or key point extraction.

    Args:
        text: The text to analyze
        analysis_type: Type of analysis ('summary', 'sentiment', 'key_points')

    Returns:
        Analysis results
    """
    # This would use LLM in production for actual analysis
    word_count = len(text.split())

    if analysis_type == "summary":
        return f"Text contains {word_count} words. [Full summarization requires processing]"
    elif analysis_type == "sentiment":
        return f"Analyzed {word_count} words. [Sentiment analysis requires processing]"
    elif analysis_type == "key_points":
        return f"Text contains {word_count} words. [Key point extraction requires processing]"
    else:
        return f"Unknown analysis type: {analysis_type}"


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class ResearchAssistant:
    """
    A complete research assistant agent.

    Features:
    - Multi-tool capability (search, calculate, datetime, analyze)
    - Conversation memory with session management
    - Error handling and graceful degradation
    - Configurable behavior via system prompt

    Architecture follows the ReAct pattern:
    - Reason about what to do
    - Act using tools
    - Observe results
    - Repeat or respond
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the research assistant.

        Args:
            model: OpenAI model to use
            temperature: Response randomness (0-1)
            verbose: Whether to print debug information
        """
        self.verbose = verbose

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )

        # Set up tools
        self.tools = [search_web, calculate, get_current_datetime, analyze_text]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Session management
        self.sessions: Dict[str, ChatMessageHistory] = {}

        # System prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description.split(chr(10))[0]}"
            for t in self.tools
        ])

        return f"""You are a helpful research assistant with access to tools.

Available Tools:
{tool_descriptions}

Guidelines:
1. Use tools when they would genuinely help answer the question
2. For calculations, always use the calculate tool - don't compute manually
3. For current information (news, weather, etc.), use search_web
4. Think step-by-step for complex questions
5. Be concise but thorough in your responses
6. If you're unsure about something, say so
7. Reference conversation history when relevant

Remember: You're having a conversation. Be helpful and friendly."""

    def _get_session(self, session_id: str) -> ChatMessageHistory:
        """Get or create a session's chat history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
            if self.verbose:
                print(f"[DEBUG] Created new session: {session_id}")
        return self.sessions[session_id]

    def _execute_tools(self, tool_calls: List[Dict]) -> List[ToolMessage]:
        """Execute tool calls and return results."""
        results = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]

            if self.verbose:
                print(f"[TOOL] {tool_name}({json.dumps(tool_args)})")

            try:
                if tool_name in self.tool_map:
                    result = self.tool_map[tool_name].invoke(tool_args)
                else:
                    result = f"Error: Unknown tool '{tool_name}'"
            except Exception as e:
                result = f"Error executing {tool_name}: {str(e)}"

            if self.verbose:
                print(f"[RESULT] {result}")

            results.append(ToolMessage(
                content=str(result),
                tool_call_id=call["id"]
            ))

        return results

    def chat(
        self,
        message: str,
        session_id: str = "default"
    ) -> str:
        """
        Send a message and get a response.

        Args:
            message: The user's message
            session_id: Session identifier for conversation continuity

        Returns:
            The agent's response
        """
        session = self._get_session(session_id)

        # Build messages with history
        messages = [
            {"role": "system", "content": self.system_prompt},
            *[
                {"role": m.type, "content": m.content}
                for m in session.messages
            ],
            {"role": "user", "content": message}
        ]

        if self.verbose:
            print(f"[INPUT] {message}")
            print(f"[CONTEXT] {len(session.messages)} previous messages")

        try:
            # Get LLM response (may include tool calls)
            response = self.llm_with_tools.invoke(messages)

            # Handle tool calls if present
            if response.tool_calls:
                if self.verbose:
                    print(f"[TOOLS] {len(response.tool_calls)} tool(s) requested")

                # Add assistant message with tool calls
                messages.append(response)

                # Execute tools and add results
                tool_results = self._execute_tools(response.tool_calls)
                for result in tool_results:
                    messages.append(result)

                # Get final response with tool results
                final_response = self.llm.invoke(messages)
                response_text = final_response.content
            else:
                response_text = response.content

            # Update memory
            session.add_user_message(message)
            session.add_ai_message(response_text)

            if self.verbose:
                print(f"[OUTPUT] {response_text[:100]}...")

            return response_text

        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}. Please try again."
            if self.verbose:
                print(f"[ERROR] {e}")
            return error_msg

    def clear_session(self, session_id: str = "default") -> None:
        """Clear a session's conversation history."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def get_session_info(self, session_id: str = "default") -> Dict[str, Any]:
        """Get information about a session."""
        session = self._get_session(session_id)
        return {
            "session_id": session_id,
            "message_count": len(session.messages),
            "exists": session_id in self.sessions
        }


# =============================================================================
# DEMO
# =============================================================================

def run_demo():
    """Run a demonstration of the research assistant."""
    print("=" * 60)
    print("RESEARCH ASSISTANT DEMO")
    print("=" * 60)

    # Create agent with verbose mode
    assistant = ResearchAssistant(verbose=True)

    # Demo conversations
    demo_messages = [
        "Hi! What's today's date?",
        "Calculate a 18% tip on a bill of $67.50",
        "Search for the latest news about AI",
        "Based on what you found, what should I focus on learning?",
        "What have we discussed so far?",
    ]

    print("\n" + "-" * 60)
    for message in demo_messages:
        print(f"\nUSER: {message}")
        response = assistant.chat(message)
        print(f"\nASSISTANT: {response}")
        print("-" * 60)

    # Show session info
    print("\n" + "=" * 60)
    print("SESSION INFO")
    print("=" * 60)
    info = assistant.get_session_info()
    print(f"Session ID: {info['session_id']}")
    print(f"Messages: {info['message_count']}")


def run_interactive():
    """Run an interactive session."""
    print("=" * 60)
    print("INTERACTIVE RESEARCH ASSISTANT")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("=" * 60)

    assistant = ResearchAssistant(verbose=False)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                assistant.clear_session()
                print("Conversation cleared.")
                continue
            elif not user_input:
                continue

            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        run_demo()
