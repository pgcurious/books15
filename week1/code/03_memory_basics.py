"""
Module 1.2: Memory Basics
=========================
Understanding how memory enables persistence and learning in agents.
Memory transforms a stateless LLM into a contextual, learning system.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any
import json

load_dotenv()


# =============================================================================
# MEMORY IMPLEMENTATIONS
# =============================================================================

class ConversationMemory:
    """
    Simple conversation memory - stores the full conversation history.

    Analogy: Like a person's short-term memory during a conversation.
    They remember everything that was said in the current chat.
    """

    def __init__(self, max_messages: int = 100):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages

    def add_user_message(self, content: str):
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_if_needed()

    def add_ai_message(self, content: str):
        self.messages.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_if_needed()

    def get_messages_for_llm(self) -> List:
        """Convert to LangChain message format."""
        result = []
        for msg in self.messages:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            else:
                result.append(AIMessage(content=msg["content"]))
        return result

    def _trim_if_needed(self):
        """Remove old messages if we exceed the limit."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def clear(self):
        self.messages = []


class SummarizingMemory:
    """
    Memory that summarizes old conversations to save context space.

    Analogy: Like human memory - we don't remember every word of past
    conversations, but we remember the key points and conclusions.
    """

    def __init__(self, llm, summary_threshold: int = 10):
        self.llm = llm
        self.summary_threshold = summary_threshold
        self.summary: str = ""
        self.recent_messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})

        if len(self.recent_messages) >= self.summary_threshold:
            self._summarize()

    def _summarize(self):
        """Summarize old messages and clear them."""
        if not self.recent_messages:
            return

        # Build conversation text
        conversation = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in self.recent_messages[:-2]  # Keep last 2 messages
        ])

        prompt = f"""Summarize this conversation concisely, keeping key facts and context:

Previous Summary: {self.summary if self.summary else 'None'}

Recent Conversation:
{conversation}

New Summary:"""

        response = self.llm.invoke(prompt)
        self.summary = response.content

        # Keep only the last 2 messages
        self.recent_messages = self.recent_messages[-2:]

    def get_context(self) -> str:
        """Get the full context (summary + recent messages)."""
        recent = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in self.recent_messages
        ])

        if self.summary:
            return f"Summary of earlier conversation:\n{self.summary}\n\nRecent messages:\n{recent}"
        return recent


class FactMemory:
    """
    Semantic memory for storing facts and preferences.

    Analogy: Like a notebook where you write down important facts
    about people and topics. "John likes coffee" → retrieve later.
    """

    def __init__(self):
        self.facts: Dict[str, Dict[str, Any]] = {}

    def store_fact(self, key: str, value: str, category: str = "general"):
        """Store a fact with optional category."""
        self.facts[key] = {
            "value": value,
            "category": category,
            "stored_at": datetime.now().isoformat()
        }

    def recall_fact(self, key: str) -> str:
        """Recall a specific fact."""
        if key in self.facts:
            return self.facts[key]["value"]
        return None

    def search_facts(self, query: str) -> List[str]:
        """Search facts by key or value containing query."""
        results = []
        query_lower = query.lower()
        for key, data in self.facts.items():
            if query_lower in key.lower() or query_lower in data["value"].lower():
                results.append(f"{key}: {data['value']}")
        return results

    def get_facts_by_category(self, category: str) -> List[str]:
        """Get all facts in a category."""
        return [
            f"{k}: {v['value']}"
            for k, v in self.facts.items()
            if v["category"] == category
        ]

    def to_context_string(self) -> str:
        """Convert all facts to a context string for the LLM."""
        if not self.facts:
            return "No stored facts."
        return "\n".join([f"- {k}: {v['value']}" for k, v in self.facts.items()])


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_conversation_memory():
    """Show basic conversation memory in action."""
    print("=" * 60)
    print("DEMO 1: Conversation Memory")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")
    memory = ConversationMemory()

    def chat(user_input: str) -> str:
        memory.add_user_message(user_input)

        messages = [
            SystemMessage(content="You are a helpful assistant. Be concise."),
            *memory.get_messages_for_llm()
        ]

        response = llm.invoke(messages)
        memory.add_ai_message(response.content)

        return response.content

    # Demonstrate memory persistence
    conversations = [
        "Hi! My name is Alex and I'm learning about AI agents.",
        "What's 2 + 2?",
        "What's my name?",  # Should remember!
        "What did I say I was learning about?",  # Should remember!
    ]

    for user_input in conversations:
        print(f"User: {user_input}")
        response = chat(user_input)
        print(f"AI: {response}")
        print()

    print("✓ The AI remembered the name and topic from earlier!\n")


def demo_without_memory():
    """Show what happens without memory - for contrast."""
    print("=" * 60)
    print("DEMO 2: Without Memory (The Problem)")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # First message
    response1 = llm.invoke("My name is Alex.")
    print(f"User: My name is Alex.")
    print(f"AI: {response1.content}")
    print()

    # Second message - no context!
    response2 = llm.invoke("What's my name?")
    print(f"User: What's my name?")
    print(f"AI: {response2.content}")
    print()

    print("✗ Without memory, the LLM doesn't know the name!\n")


def demo_fact_memory():
    """Show semantic/fact memory for storing information."""
    print("=" * 60)
    print("DEMO 3: Fact Memory (Long-term Storage)")
    print("=" * 60)

    memory = FactMemory()

    # Store some facts
    memory.store_fact("user_name", "Alex", category="user_info")
    memory.store_fact("user_role", "Software Engineer", category="user_info")
    memory.store_fact("preferred_language", "Python", category="preferences")
    memory.store_fact("learning_topic", "AI Agents", category="context")

    print("Stored facts:")
    print(memory.to_context_string())
    print()

    # Recall specific fact
    print(f"Recall 'user_name': {memory.recall_fact('user_name')}")

    # Search facts
    print(f"Search 'python': {memory.search_facts('python')}")

    # Get by category
    print(f"User info category: {memory.get_facts_by_category('user_info')}")
    print()


def demo_memory_in_agent():
    """Show how memory integrates into a complete agent."""
    print("=" * 60)
    print("DEMO 4: Memory in Complete Agent")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")
    conversation = ConversationMemory()
    facts = FactMemory()

    system_template = """You are a helpful assistant with memory capabilities.

Known facts about the user:
{facts}

Use this information to personalize your responses. If you learn new facts
about the user (name, preferences, etc.), mention that you'll remember them.
"""

    def chat_with_full_memory(user_input: str) -> str:
        # Simple fact extraction (in production, use LLM for this)
        if "my name is" in user_input.lower():
            name = user_input.lower().split("my name is")[-1].strip().split()[0].title()
            facts.store_fact("user_name", name, "user_info")

        if "i love" in user_input.lower():
            thing = user_input.lower().split("i love")[-1].strip().rstrip(".")
            facts.store_fact("user_loves", thing, "preferences")

        conversation.add_user_message(user_input)

        messages = [
            SystemMessage(content=system_template.format(facts=facts.to_context_string())),
            *conversation.get_messages_for_llm()
        ]

        response = llm.invoke(messages)
        conversation.add_ai_message(response.content)

        return response.content

    # Demo conversation
    interactions = [
        "Hello! My name is Jordan and I love hiking.",
        "Can you recommend an activity for me?",
        "What do you remember about me?",
    ]

    for user_input in interactions:
        print(f"User: {user_input}")
        response = chat_with_full_memory(user_input)
        print(f"AI: {response}")
        print()

    print("Stored Facts:")
    print(facts.to_context_string())
    print()


def demo_context_window_limits():
    """Demonstrate context window limitations and solutions."""
    print("=" * 60)
    print("DEMO 5: Context Window Limits")
    print("=" * 60)

    print("""
Context windows have finite size:
- GPT-3.5: ~4K tokens (~3000 words)
- GPT-4: 8K-128K tokens
- Claude: 100K-200K tokens

When conversation exceeds the window, you have options:
1. Truncate old messages (lose information)
2. Summarize old messages (compress information)
3. Use external memory (vector database)

Example token counts:
- "Hello, how are you?" = ~5 tokens
- This entire demo file = ~1500 tokens
- A typical conversation = 500-2000 tokens
""")

    # Show summarization concept
    llm = ChatOpenAI(model="gpt-4o-mini")

    long_conversation = """
User: I'm planning a trip to Japan next month.
AI: That sounds exciting! Japan is wonderful. What cities are you considering?
User: Mainly Tokyo and Kyoto. I love temples and food.
AI: Great choices! Tokyo for modern culture and food, Kyoto for traditional temples.
User: How many days should I spend in each?
AI: I'd suggest 4-5 days in Tokyo and 2-3 days in Kyoto.
User: What about transportation?
AI: Get a JR Pass for train travel. It's economical for tourists.
User: Any food recommendations?
AI: Try ramen in Tokyo, kaiseki in Kyoto. Visit Tsukiji for sushi!
"""

    summary_prompt = f"Summarize this conversation in 2-3 sentences, keeping key facts:\n\n{long_conversation}"
    summary = llm.invoke(summary_prompt)

    print("Original conversation: ~150 words")
    print("Summarized conversation:")
    print(summary.content)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MEMORY BASICS - Enabling Persistence and Learning")
    print("=" * 60 + "\n")

    demo_without_memory()
    demo_conversation_memory()
    demo_fact_memory()
    demo_memory_in_agent()
    demo_context_window_limits()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. Without memory, LLMs are stateless - each call is independent
2. Conversation memory stores chat history for context
3. Fact memory stores important information long-term
4. Memory enables personalization and learning
5. Context windows limit memory - summarization helps
6. Memory is what makes agents feel "intelligent" over time
    """)
