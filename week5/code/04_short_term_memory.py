"""
Module 5.2: Short-Term Memory Systems
=====================================
Demonstrates short-term memory patterns:
- Conversation Buffer Memory
- Conversation Window Memory
- Conversation Summary Memory
- Working Memory
"""

from typing import Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory
)

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# PATTERN 1: CONVERSATION BUFFER MEMORY
# ============================================================

class ConversationBufferAgent:
    """Agent with simple conversation buffer memory (keeps all history)."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use conversation history to provide contextual responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    def chat(self, user_input: str) -> str:
        """Process user input with conversation memory."""

        # Get history
        history = self.memory.load_memory_variables({})
        chat_history = history.get("chat_history", [])

        # Generate response
        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=user_input
        )
        response = self.llm.invoke(messages)

        # Save to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

        return response.content

    def get_history_size(self) -> int:
        """Get number of messages in history."""
        history = self.memory.load_memory_variables({})
        return len(history.get("chat_history", []))


def demo_buffer_memory():
    """Demonstrate conversation buffer memory."""
    print("=" * 60)
    print("DEMO 1: Conversation Buffer Memory")
    print("=" * 60)

    agent = ConversationBufferAgent()

    conversations = [
        "Hi! My name is Alice and I'm a software engineer.",
        "I work mainly with Python and TypeScript.",
        "I'm interested in learning about machine learning.",
        "What's my name and what do I do?",
        "What programming languages do I know?"
    ]

    print("\n--- Conversation ---")
    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = agent.chat(user_msg)
        print(f"Agent: {response}")
        print(f"  [History size: {agent.get_history_size()} messages]")

    print()


# ============================================================
# PATTERN 2: CONVERSATION WINDOW MEMORY
# ============================================================

class WindowedMemoryAgent:
    """Agent with windowed memory (keeps only last N exchanges)."""

    def __init__(self, window_size: int = 3):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.window_size = window_size

        # Only keep last N exchanges
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,
            memory_key="chat_history"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
            You only have access to recent conversation history.
            If asked about something not in recent history, say you don't have that context."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    def chat(self, user_input: str) -> str:
        """Process user input with windowed memory."""

        history = self.memory.load_memory_variables({})
        chat_history = history.get("chat_history", [])

        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=user_input
        )
        response = self.llm.invoke(messages)

        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

        return response.content

    def get_history_size(self) -> int:
        """Get current history size."""
        history = self.memory.load_memory_variables({})
        return len(history.get("chat_history", []))


def demo_window_memory():
    """Demonstrate windowed memory (forgets old context)."""
    print("=" * 60)
    print("DEMO 2: Conversation Window Memory (k=2)")
    print("=" * 60)

    agent = WindowedMemoryAgent(window_size=2)

    conversations = [
        "My name is Bob.",
        "I like pizza.",
        "My favorite color is blue.",
        "I have a dog named Max.",
        "What's my name?",  # Should not remember (outside window)
        "What's my favorite color?",  # Should remember
    ]

    print(f"\n[Window size: 2 exchanges]")
    print("\n--- Conversation ---")

    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = agent.chat(user_msg)
        print(f"Agent: {response}")
        print(f"  [History size: {agent.get_history_size()} messages]")

    print("\n[Note: The agent forgot the name because it was outside the window!]")
    print()


# ============================================================
# PATTERN 3: CONVERSATION SUMMARY MEMORY
# ============================================================

class SummaryMemoryAgent:
    """Agent that summarizes old conversations to save space."""

    def __init__(self, max_token_limit: int = 200):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Summarizes when context exceeds limit
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
            You have access to a summary of past conversations and recent messages.
            Use this context to provide helpful responses."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    def chat(self, user_input: str) -> str:
        """Chat with summary memory."""

        history = self.memory.load_memory_variables({})
        chat_history = history.get("chat_history", [])

        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=user_input
        )
        response = self.llm.invoke(messages)

        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

        return response.content

    def get_summary(self) -> str:
        """Get the current conversation summary."""
        return self.memory.moving_summary_buffer or "No summary yet."


def demo_summary_memory():
    """Demonstrate summary memory."""
    print("=" * 60)
    print("DEMO 3: Conversation Summary Memory")
    print("=" * 60)

    agent = SummaryMemoryAgent(max_token_limit=150)

    conversations = [
        "I'm planning a trip to Japan next month.",
        "I want to visit Tokyo, Kyoto, and Osaka.",
        "I'm interested in temples, food, and technology.",
        "My budget is around $3000 for two weeks.",
        "I don't speak Japanese but I'm willing to learn basics.",
        "What do you remember about my trip plans?"
    ]

    print(f"\n[Max token limit: 150 - older messages will be summarized]")
    print("\n--- Conversation ---")

    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = agent.chat(user_msg)
        print(f"Agent: {response[:200]}...")

        summary = agent.get_summary()
        if summary and summary != "No summary yet.":
            print(f"  [Current summary: {summary[:100]}...]")

    print()


# ============================================================
# PATTERN 4: WORKING MEMORY
# ============================================================

@dataclass
class WorkingMemoryItem:
    """A single item in working memory."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5  # 0-1 scale


class WorkingMemory:
    """
    Structured working memory for agents.

    Implements Miller's Law: ~7 items capacity.
    Uses importance + recency for eviction.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: dict[str, WorkingMemoryItem] = {}

    def store(self, key: str, value: Any, importance: float = 0.5):
        """Store an item in working memory."""

        # If at capacity, evict least important item
        if len(self.items) >= self.capacity and key not in self.items:
            self._evict_one()

        self.items[key] = WorkingMemoryItem(
            key=key,
            value=value,
            importance=importance
        )

        print(f"  [WM] Stored: {key} = {str(value)[:30]}... (importance: {importance})")

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve an item (and increase access count)."""
        if key in self.items:
            self.items[key].access_count += 1
            return self.items[key].value
        return None

    def _evict_one(self):
        """Evict the least important/accessed item."""
        if not self.items:
            return

        # Score = importance + recency + access_count
        def score(item: WorkingMemoryItem) -> float:
            recency = (datetime.now() - item.created_at).total_seconds()
            return item.importance + (item.access_count * 0.1) - (recency * 0.001)

        lowest_key = min(self.items.keys(), key=lambda k: score(self.items[k]))
        evicted = self.items.pop(lowest_key)
        print(f"  [WM] Evicted: {lowest_key} (importance: {evicted.importance})")

    def get_context(self) -> str:
        """Get working memory as context string."""
        if not self.items:
            return "Working memory is empty."

        lines = ["Current working memory:"]
        for key, item in sorted(self.items.items(), key=lambda x: -x[1].importance):
            lines.append(f"  - {key}: {item.value}")

        return "\n".join(lines)

    def clear(self):
        """Clear all items."""
        self.items.clear()


class WorkingMemoryAgent:
    """Agent with structured working memory."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.working_memory = WorkingMemory(capacity=5)
        self.conversation_history: List[dict] = []

    def process(self, user_input: str) -> str:
        """Process input using working memory."""

        # Extract and store important information
        self._update_working_memory(user_input)

        # Build prompt with working memory context
        wm_context = self.working_memory.get_context()
        history_context = self._format_history()

        prompt = f"""
        {wm_context}

        Recent conversation:
        {history_context}

        User: {user_input}

        Respond helpfully, using the working memory context when relevant.
        """

        response = self.llm.invoke(prompt)

        # Update conversation history
        self.conversation_history.append({"user": user_input, "assistant": response.content})
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]

        return response.content

    def _update_working_memory(self, user_input: str):
        """Extract and store important information from input."""

        # Simple extraction logic (in production, use NER)
        input_lower = user_input.lower()

        # Check for name mentions
        if "my name is" in input_lower or "i'm " in input_lower or "i am " in input_lower:
            self.working_memory.store("user_introduction", user_input, importance=0.9)

        # Check for preferences
        if "i like" in input_lower or "i love" in input_lower or "i prefer" in input_lower:
            self.working_memory.store("user_preference", user_input, importance=0.7)

        # Check for questions being asked
        if "?" in user_input:
            self.working_memory.store("last_question", user_input, importance=0.6)

        # Check for tasks/requests
        if any(word in input_lower for word in ["please", "can you", "could you", "help me"]):
            self.working_memory.store("current_task", user_input, importance=0.8)

    def _format_history(self) -> str:
        """Format recent conversation history."""
        if not self.conversation_history:
            return "No previous conversation."

        return "\n".join(
            f"User: {ex['user']}\nAssistant: {ex['assistant']}"
            for ex in self.conversation_history[-3:]
        )


def demo_working_memory():
    """Demonstrate working memory."""
    print("=" * 60)
    print("DEMO 4: Working Memory")
    print("=" * 60)

    agent = WorkingMemoryAgent()

    conversations = [
        "My name is Charlie.",
        "I like rock climbing and hiking.",
        "Can you help me plan a weekend trip?",
        "I prefer mountains over beaches.",
        "My budget is $500.",
        "I also enjoy photography.",
        "What do you know about me?",
    ]

    print(f"\n[Working memory capacity: 5 items]")
    print("\n--- Conversation ---")

    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = agent.process(user_msg)
        print(f"Agent: {response[:200]}...")
        print(f"  [WM items: {len(agent.working_memory.items)}]")

    print("\n--- Final Working Memory State ---")
    print(agent.working_memory.get_context())
    print()


# ============================================================
# COMPARISON: CHOOSING THE RIGHT MEMORY TYPE
# ============================================================

def demo_comparison():
    """Compare memory types."""
    print("=" * 60)
    print("SHORT-TERM MEMORY COMPARISON")
    print("=" * 60)

    comparison = """
    BUFFER MEMORY
    ─────────────
    Keeps: All messages
    Pros: Complete history, nothing lost
    Cons: Grows unbounded, expensive for long conversations
    Use when: Short conversations, need full context

    WINDOW MEMORY
    ─────────────
    Keeps: Last N exchanges
    Pros: Fixed size, predictable cost
    Cons: Loses old context
    Use when: Only recent context matters

    SUMMARY MEMORY
    ──────────────
    Keeps: Summary + recent messages
    Pros: Preserves key info, bounded size
    Cons: May lose details, summary overhead
    Use when: Long conversations, key facts matter

    WORKING MEMORY
    ──────────────
    Keeps: Structured key-value pairs
    Pros: Organized, importance-based retention
    Cons: Needs extraction logic
    Use when: Need structured context, specific facts

    RECOMMENDATION
    ──────────────
    1. Simple chatbot → Buffer or Window
    2. Long conversations → Summary
    3. Task-oriented agent → Working Memory
    4. Production system → Combine multiple types!
    """

    print(comparison)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SHORT-TERM MEMORY SYSTEMS")
    print("=" * 60 + "\n")

    demo_buffer_memory()
    demo_window_memory()
    demo_summary_memory()
    demo_working_memory()
    demo_comparison()

    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. BUFFER MEMORY keeps everything but grows unbounded
       - Good for short conversations
       - Use when you need full history

    2. WINDOW MEMORY keeps only recent context
       - Predictable size and cost
       - Good when only recent context matters

    3. SUMMARY MEMORY compresses old conversations
       - Preserves key information
       - Good for long conversations

    4. WORKING MEMORY stores structured information
       - Organized key-value storage
       - Good for task-oriented agents

    5. In production, COMBINE memory types:
       - Working memory for current task
       - Summary for conversation context
       - Long-term storage for persistence
    """)
