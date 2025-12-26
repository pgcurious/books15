"""
Module 5.2: Long-Term Memory Systems
=====================================
Demonstrates long-term memory patterns:
- Semantic Memory (Vector Store)
- Episodic Memory
- Hybrid Retrieval
- Complete Memory-Enhanced Agent
"""

from typing import List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import math

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# PATTERN 1: SEMANTIC MEMORY (VECTOR STORE)
# ============================================================

class SemanticMemory:
    """Long-term semantic memory using vector embeddings."""

    def __init__(self, collection_name: str = "semantic_memory"):
        self.embeddings = OpenAIEmbeddings()

        # Using Chroma as vector store (in-memory for demo)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def store(self, content: str, metadata: dict = None) -> str:
        """Store content in semantic memory."""
        doc_id = str(uuid.uuid4())

        doc_metadata = {
            "stored_at": datetime.now().isoformat(),
            "memory_id": doc_id,
            **(metadata or {})
        }

        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )

        return doc_id

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve semantically similar memories."""
        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve with relevance scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)


class SemanticMemoryAgent:
    """Agent with semantic long-term memory."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory = SemanticMemory()
        self.conversation_buffer: List[dict] = []

    def chat(self, user_input: str) -> str:
        """Chat with semantic memory retrieval."""

        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(user_input, k=3)

        memory_context = ""
        if relevant_memories:
            memory_context = "Relevant memories:\n" + "\n".join(
                f"- {doc.page_content}"
                for doc in relevant_memories
            )

        # Build prompt
        buffer_context = self._format_buffer()

        prompt = f"""
        {memory_context}

        Recent conversation:
        {buffer_context}

        User: {user_input}

        Respond helpfully. Use memories if relevant, ignore if not.
        """

        response = self.llm.invoke(prompt)

        # Decide whether to store this interaction
        self._maybe_store(user_input, response.content)

        # Update buffer
        self.conversation_buffer.append({
            "user": user_input,
            "assistant": response.content
        })
        if len(self.conversation_buffer) > 5:
            self.conversation_buffer = self.conversation_buffer[-5:]

        return response.content

    def _maybe_store(self, user_input: str, response: str):
        """Store interaction if it contains important information."""
        # Simple heuristic: store if user shared personal info
        important_signals = [
            "my name", "i am", "i'm", "i work", "i like", "i love",
            "i prefer", "my favorite", "i live", "my goal"
        ]

        if any(signal in user_input.lower() for signal in important_signals):
            self.memory.store(
                f"User said: {user_input}",
                metadata={"type": "user_info"}
            )
            print(f"  [Memory] Stored: {user_input[:50]}...")

    def teach(self, knowledge: str) -> str:
        """Explicitly teach the agent something."""
        self.memory.store(knowledge, metadata={"type": "taught"})
        return f"Learned: {knowledge[:100]}..."

    def _format_buffer(self) -> str:
        if not self.conversation_buffer:
            return "No recent conversation."
        return "\n".join(
            f"User: {ex['user']}\nAssistant: {ex['assistant']}"
            for ex in self.conversation_buffer[-3:]
        )


def demo_semantic_memory():
    """Demonstrate semantic memory."""
    print("=" * 60)
    print("DEMO 1: Semantic Memory (Vector Store)")
    print("=" * 60)

    agent = SemanticMemoryAgent()

    # Teach the agent some facts
    print("\n--- Teaching Phase ---")
    print(agent.teach("The user prefers detailed technical explanations."))
    print(agent.teach("When discussing code, always include examples."))
    print(agent.teach("The user is experienced with Python but new to Go."))

    # Have a conversation
    print("\n--- Conversation ---")
    conversations = [
        "My name is David and I'm a backend developer.",
        "I'm interested in learning about microservices.",
        "Can you explain how to structure a microservice?",
        "What's my background and what should you keep in mind?"
    ]

    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = agent.chat(user_msg)
        print(f"Agent: {response[:200]}...")

    print()


# ============================================================
# PATTERN 2: EPISODIC MEMORY
# ============================================================

@dataclass
class Episode:
    """A complete episode/experience."""
    episode_id: str
    timestamp: datetime
    context: str  # What was happening
    actions: List[str]  # What actions were taken
    outcome: str  # What happened
    success: bool  # Was it successful
    lessons: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert episode to searchable text."""
        return f"""
        Context: {self.context}
        Actions: {', '.join(self.actions)}
        Outcome: {self.outcome}
        Success: {self.success}
        Lessons: {', '.join(self.lessons)}
        """


class EpisodicMemory:
    """Memory system for storing and retrieving episodes."""

    def __init__(self):
        self.episodes: List[Episode] = []
        self.semantic_index = SemanticMemory(collection_name="episodes")

    def record_episode(self, episode: Episode):
        """Record a new episode."""
        self.episodes.append(episode)

        # Index in semantic memory for retrieval
        self.semantic_index.store(
            episode.to_text(),
            metadata={
                "episode_id": episode.episode_id,
                "success": episode.success,
                "timestamp": episode.timestamp.isoformat()
            }
        )

        print(f"  [Episodic] Recorded episode: {episode.context[:50]}...")

    def recall_similar(self, situation: str, k: int = 3) -> List[Episode]:
        """Recall episodes similar to current situation."""
        docs = self.semantic_index.retrieve(situation, k=k)

        recalled = []
        for doc in docs:
            episode_id = doc.metadata.get("episode_id")
            for ep in self.episodes:
                if ep.episode_id == episode_id:
                    recalled.append(ep)
                    break

        return recalled

    def get_lessons_for(self, situation: str) -> List[str]:
        """Get lessons learned from similar situations."""
        similar = self.recall_similar(situation, k=3)

        lessons = []
        for ep in similar:
            prefix = "SUCCESS" if ep.success else "FAILURE (avoid)"
            for lesson in ep.lessons:
                lessons.append(f"{prefix}: {lesson}")

        return lessons


class EpisodicAgent:
    """Agent that learns from past episodes."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.episodic_memory = EpisodicMemory()
        self.current_actions: List[str] = []

    def process_task(self, task: str) -> str:
        """Process a task, learning from past episodes."""

        # Recall relevant past experiences
        lessons = self.episodic_memory.get_lessons_for(task)

        lessons_context = ""
        if lessons:
            lessons_context = "Lessons from past experiences:\n" + "\n".join(
                f"- {lesson}" for lesson in lessons
            )

        prompt = f"""
        Task: {task}

        {lessons_context}

        Complete the task, applying lessons from past experiences if relevant.
        """

        response = self.llm.invoke(prompt)

        self.current_actions.append(f"Completed: {task[:50]}")

        return response.content

    def complete_episode(self, context: str, outcome: str, success: bool):
        """Record the current episode."""

        # Generate lessons from the experience
        lessons_prompt = f"""
        Experience:
        - Context: {context}
        - Actions: {self.current_actions}
        - Outcome: {outcome}
        - Success: {success}

        What lessons should be learned? List 1-2 brief lessons.
        """

        lessons_response = self.llm.invoke(lessons_prompt)
        lessons = [
            l.strip().lstrip("-").strip()
            for l in lessons_response.content.split("\n")
            if l.strip() and not l.strip().startswith("Lesson")
        ][:2]

        episode = Episode(
            episode_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context=context,
            actions=self.current_actions.copy(),
            outcome=outcome,
            success=success,
            lessons=lessons
        )

        self.episodic_memory.record_episode(episode)
        self.current_actions = []

        return lessons


def demo_episodic_memory():
    """Demonstrate episodic memory."""
    print("=" * 60)
    print("DEMO 2: Episodic Memory")
    print("=" * 60)

    agent = EpisodicAgent()

    # Simulate some past experiences
    print("\n--- Recording Past Experiences ---")

    # Experience 1: Successful API integration
    agent.current_actions = ["Read API docs", "Wrote integration code", "Added error handling"]
    lessons1 = agent.complete_episode(
        context="Integrating a REST API for payment processing",
        outcome="Successfully integrated with proper error handling",
        success=True
    )
    print(f"  Lessons learned: {lessons1}")

    # Experience 2: Failed deployment
    agent.current_actions = ["Deployed without testing", "Production broke", "Had to rollback"]
    lessons2 = agent.complete_episode(
        context="Deploying a new feature to production",
        outcome="Deployment failed, caused downtime",
        success=False
    )
    print(f"  Lessons learned: {lessons2}")

    # Experience 3: Successful debugging
    agent.current_actions = ["Checked logs", "Added debug statements", "Found root cause"]
    lessons3 = agent.complete_episode(
        context="Debugging a memory leak in production",
        outcome="Found and fixed the memory leak",
        success=True
    )
    print(f"  Lessons learned: {lessons3}")

    # Now use the agent with episodic memory
    print("\n--- New Task with Episodic Memory ---")

    task = "I need to integrate a new payment API"
    print(f"\nTask: {task}")

    lessons = agent.episodic_memory.get_lessons_for(task)
    print("\nRetrieved lessons:")
    for lesson in lessons:
        print(f"  - {lesson}")

    response = agent.process_task(task)
    print(f"\nAgent response: {response[:200]}...")

    print()


# ============================================================
# PATTERN 3: HYBRID RETRIEVAL
# ============================================================

@dataclass
class MemoryItem:
    """A memory item with metadata."""
    id: str
    content: str
    timestamp: datetime
    importance: float = 0.5
    access_count: int = 0


@dataclass
class ScoredMemory:
    """A memory with computed scores."""
    memory: MemoryItem
    relevance_score: float
    recency_score: float
    importance_score: float
    final_score: float


class HybridRetriever:
    """
    Combines multiple retrieval strategies:
    - Relevance (semantic similarity)
    - Recency (time-based decay)
    - Importance (assigned weight)
    """

    def __init__(
        self,
        relevance_weight: float = 0.5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2
    ):
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name="hybrid_memory",
            embedding_function=self.embeddings
        )
        self.memories: dict[str, MemoryItem] = {}

    def add_memory(self, content: str, importance: float = 0.5) -> str:
        """Add a memory."""
        memory_id = str(uuid.uuid4())

        memory = MemoryItem(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            importance=importance
        )

        self.memories[memory_id] = memory

        self.vectorstore.add_texts(
            texts=[content],
            metadatas={"memory_id": memory_id},
            ids=[memory_id]
        )

        return memory_id

    def retrieve(self, query: str, k: int = 5) -> List[ScoredMemory]:
        """Retrieve using hybrid scoring."""

        # Get relevance scores from vector search
        results = self.vectorstore.similarity_search_with_score(query, k=k * 2)

        now = datetime.now()
        scored_memories = []

        for doc, distance in results:
            memory_id = doc.metadata.get("memory_id")
            if memory_id not in self.memories:
                continue

            memory = self.memories[memory_id]
            memory.access_count += 1

            # Relevance: convert distance to similarity (0-1)
            relevance = 1 / (1 + distance)

            # Recency: exponential decay with 24-hour half-life
            age_hours = (now - memory.timestamp).total_seconds() / 3600
            recency = math.exp(-age_hours / 24)

            # Importance: already 0-1
            importance = memory.importance

            # Combined score
            final = (
                self.relevance_weight * relevance +
                self.recency_weight * recency +
                self.importance_weight * importance
            )

            scored_memories.append(ScoredMemory(
                memory=memory,
                relevance_score=relevance,
                recency_score=recency,
                importance_score=importance,
                final_score=final
            ))

        # Sort by final score
        scored_memories.sort(key=lambda x: x.final_score, reverse=True)

        return scored_memories[:k]


def demo_hybrid_retrieval():
    """Demonstrate hybrid retrieval."""
    print("=" * 60)
    print("DEMO 3: Hybrid Retrieval")
    print("=" * 60)

    retriever = HybridRetriever(
        relevance_weight=0.5,
        recency_weight=0.3,
        importance_weight=0.2
    )

    # Add memories with different importance levels
    print("\n--- Adding Memories ---")
    memories = [
        ("User's name is Emma", 0.9),  # High importance
        ("User asked about Python yesterday", 0.5),
        ("User prefers concise answers", 0.8),
        ("Weather was sunny", 0.1),  # Low importance
        ("User is working on a web application", 0.7),
        ("User had coffee this morning", 0.1),
    ]

    for content, importance in memories:
        retriever.add_memory(content, importance)
        print(f"  Added: '{content}' (importance: {importance})")

    # Retrieve with hybrid scoring
    print("\n--- Hybrid Retrieval Results ---")
    query = "What should I know about the user?"

    results = retriever.retrieve(query, k=4)

    print(f"\nQuery: '{query}'")
    print(f"\nWeights: relevance={retriever.relevance_weight}, "
          f"recency={retriever.recency_weight}, importance={retriever.importance_weight}")

    print("\nResults:")
    for i, scored in enumerate(results, 1):
        print(f"\n  {i}. {scored.memory.content}")
        print(f"     Relevance: {scored.relevance_score:.3f}")
        print(f"     Recency:   {scored.recency_score:.3f}")
        print(f"     Importance: {scored.importance_score:.3f}")
        print(f"     FINAL:     {scored.final_score:.3f}")

    print()


# ============================================================
# COMPLETE MEMORY-ENHANCED AGENT
# ============================================================

class CompleteMemoryAgent:
    """
    Agent with complete memory system:
    - Short-term: Conversation buffer
    - Long-term: Semantic + episodic
    - Hybrid retrieval
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Short-term
        self.conversation_buffer: List[dict] = []

        # Long-term
        self.semantic_memory = SemanticMemory(collection_name="complete_agent")

        # Track what we've learned
        self.learned_facts: List[str] = []

    def chat(self, user_input: str) -> str:
        """Chat with full memory system."""

        # 1. Retrieve relevant long-term memories
        relevant = self.semantic_memory.retrieve(user_input, k=3)
        memory_context = ""
        if relevant:
            memory_context = "Remembered:\n" + "\n".join(
                f"- {doc.page_content}" for doc in relevant
            )

        # 2. Get conversation buffer
        buffer_context = ""
        if self.conversation_buffer:
            buffer_context = "Recent:\n" + "\n".join(
                f"User: {ex['user']}\nYou: {ex['assistant']}"
                for ex in self.conversation_buffer[-3:]
            )

        # 3. Generate response
        prompt = f"""
        {memory_context}

        {buffer_context}

        User: {user_input}

        Respond helpfully. Use your memories if relevant.
        """

        response = self.llm.invoke(prompt)

        # 4. Update memories
        self._update_memories(user_input, response.content)

        return response.content

    def _update_memories(self, user_input: str, response: str):
        """Update all memory systems."""

        # Update conversation buffer
        self.conversation_buffer.append({
            "user": user_input,
            "assistant": response
        })
        if len(self.conversation_buffer) > 10:
            self.conversation_buffer = self.conversation_buffer[-10:]

        # Extract and store important info
        important_patterns = ["my name", "i am", "i work", "i like", "i need"]
        if any(p in user_input.lower() for p in important_patterns):
            self.semantic_memory.store(
                f"User shared: {user_input}",
                metadata={"type": "user_info"}
            )
            self.learned_facts.append(user_input)
            print(f"  [Memory] Stored to long-term: {user_input[:40]}...")

    def teach(self, fact: str):
        """Explicitly teach the agent."""
        self.semantic_memory.store(fact, metadata={"type": "taught"})
        self.learned_facts.append(fact)
        print(f"  [Memory] Learned: {fact[:50]}...")


def demo_complete_agent():
    """Demonstrate complete memory agent."""
    print("=" * 60)
    print("DEMO 4: Complete Memory-Enhanced Agent")
    print("=" * 60)

    agent = CompleteMemoryAgent()

    # Teach some facts
    print("\n--- Teaching Phase ---")
    agent.teach("User prefers Python for backend development")
    agent.teach("User is building a SaaS product")
    agent.teach("User values clean, maintainable code")

    # Have a conversation
    print("\n--- Conversation ---")
    conversations = [
        "Hi! I'm Frank and I'm a startup founder.",
        "I'm trying to decide between Django and FastAPI.",
        "What do you remember about me?",
        "Based on what you know, which would you recommend?"
    ]

    for msg in conversations:
        print(f"\nUser: {msg}")
        response = agent.chat(msg)
        print(f"Agent: {response[:250]}...")

    print("\n--- Learned Facts ---")
    for fact in agent.learned_facts:
        print(f"  - {fact}")

    print()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LONG-TERM MEMORY SYSTEMS")
    print("=" * 60 + "\n")

    demo_semantic_memory()
    demo_episodic_memory()
    demo_hybrid_retrieval()
    demo_complete_agent()

    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. SEMANTIC MEMORY stores knowledge as embeddings
       - Enables similarity-based retrieval
       - Good for facts, concepts, learned information

    2. EPISODIC MEMORY stores complete experiences
       - Records context, actions, outcomes, lessons
       - Enables learning from past successes/failures

    3. HYBRID RETRIEVAL combines multiple signals
       - Relevance (semantic similarity)
       - Recency (time decay)
       - Importance (assigned weight)

    4. PRODUCTION AGENTS should combine memory types:
       - Short-term for conversation context
       - Long-term semantic for knowledge
       - Long-term episodic for experiences

    5. Key design decisions:
       - What to store (selection criteria)
       - How to retrieve (scoring strategy)
       - How to inject (prompt engineering)
    """)
