# Module 5.2: Agent Memory Systems

> "The true art of memory is the art of attention." — Samuel Johnson

## What You'll Learn

- Why memory transforms agents from stateless to intelligent
- Short-term memory: conversation context and working memory
- Long-term memory: vector stores, episodic memory, and persistent knowledge
- Memory retrieval strategies and when to use them
- Shared memory for multi-agent coordination
- Building agents that learn from experience

---

## First Principles: What Is Memory?

Let's build our understanding from the ground up. What does it mean for an agent to "remember"?

### The Stateless Problem

By default, LLM-based agents are **stateless**:

```
WITHOUT MEMORY

User: "My name is Alice"
Agent: "Nice to meet you, Alice!"

User: "What's my name?"
Agent: "I don't know your name. Could you tell me?"

     ┌─────────────────────────────────────────────┐
     │                                             │
     │  Each call is INDEPENDENT                   │
     │                                             │
     │  Call 1: "My name is Alice" ──► Response    │
     │                              ────┘          │
     │             (forgotten)                     │
     │                                             │
     │  Call 2: "What's my name?" ──► Response     │
     │                            ────┘            │
     │             (no context)                    │
     │                                             │
     └─────────────────────────────────────────────┘
```

This is like a goldfish—every moment is new, no continuity.

### Memory as Context Injection

At its core, memory is simply **injecting relevant past information into the current context**:

```
WITH MEMORY

User: "My name is Alice"
Agent: "Nice to meet you, Alice!"
     │
     └──► STORED: {"user_name": "Alice"}

User: "What's my name?"
     │
     └──► RETRIEVED: {"user_name": "Alice"}
     │
     └──► Injected into prompt:
         "Previous context: User's name is Alice.
          Current question: What's my name?"

Agent: "Your name is Alice!"
```

### The Memory Formula

```
Effective Memory = Storage + Retrieval + Injection

Where:
├── Storage: How we save information
│   ├── What to store (selection)
│   ├── How to structure it (format)
│   └── Where to put it (backend)
│
├── Retrieval: How we find relevant memories
│   ├── When to retrieve (triggers)
│   ├── What to retrieve (relevance)
│   └── How much to retrieve (quantity)
│
└── Injection: How we add memories to context
    ├── Where in the prompt (position)
    ├── How to format (representation)
    └── How to prioritize (ordering)
```

### Types of Memory: A Taxonomy

```
                        AGENT MEMORY
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  SHORT   │      │  LONG    │      │  SHARED  │
    │  TERM    │      │  TERM    │      │  (Multi- │
    │          │      │          │      │  Agent)  │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                  │
    ┌────┴────┐       ┌────┴────┐        ┌───┴────┐
    │         │       │         │        │        │
    ▼         ▼       ▼         ▼        ▼        ▼
Conversa-  Working  Semantic  Episodic  Black-  Message
tion       Memory   Memory    Memory    board   History
Buffer                                  State
```

---

## Analogical Thinking: Memory as Human Cognition

Understanding agent memory becomes intuitive when we map it to human memory systems.

### The Human Memory Analogy

| Human Memory | Agent Equivalent | Duration | Capacity | Use Case |
|--------------|-----------------|----------|----------|----------|
| **Sensory** | Token window | Milliseconds | Very limited | Immediate processing |
| **Working** | Conversation buffer | Seconds-minutes | ~7 items | Current task |
| **Short-term** | Session state | Minutes-hours | Limited | Ongoing conversation |
| **Long-term Declarative** | Vector store | Days-years | Unlimited | Facts and knowledge |
| **Long-term Episodic** | Interaction logs | Days-years | Unlimited | Past experiences |
| **Long-term Procedural** | Fine-tuned weights | Permanent | Embedded | Skills and habits |

### The Library Analogy

Think of agent memory as a library system:

```
THE LIBRARY OF AGENT MEMORY

┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  DESK (Working Memory)                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Currently open books, notes, the immediate task                    │  │
│  │ Limited space - only a few items at once                           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  READING ROOM (Short-term Memory)                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Recent research, conversation history                               │  │
│  │ Easily accessible, relatively fresh                                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  STACKS (Long-term Memory)                                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │  │
│  │ │ Vector Store    │  │ Episodic        │  │ Document        │      │  │
│  │ │ (Semantic)      │  │ Memory          │  │ Store           │      │  │
│  │ │                 │  │                 │  │                 │      │  │
│  │ │ "Quantum        │  │ "On 2024-01-15  │  │ Full PDFs,      │      │  │
│  │ │  computing is   │  │  user asked     │  │ manuals,        │      │  │
│  │ │  about qubits"  │  │  about X..."    │  │ code files      │      │  │
│  │ └─────────────────┘  └─────────────────┘  └─────────────────┘      │  │
│  │                                                                     │  │
│  │ Requires SEARCH to find (can't access directly)                     │  │
│  │ Indexed for efficient retrieval                                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  CARD CATALOG (Memory Index)                                             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Embeddings, metadata, search indices                               │  │
│  │ "How do I find what I'm looking for?"                              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Just like a library, an agent can't hold everything on its desk. It must:
1. Keep immediate needs close (working memory)
2. Have good organization (indexing)
3. Know how to search (retrieval)
4. Bring relevant items to the desk (injection)

---

## Emergence Thinking: Memory Enables Learning

Memory is the foundation for emergent learning in agent systems.

### The Learning Loop

```
EMERGENT LEARNING FROM MEMORY

    ┌─────────────────────────────────────────────────────────┐
    │                                                          │
    │                    EXPERIENCE                            │
    │                        │                                 │
    │                        ▼                                 │
    │              ┌─────────────────┐                         │
    │              │     STORE       │                         │
    │              │   experience    │                         │
    │              │   with outcome  │                         │
    │              └────────┬────────┘                         │
    │                       │                                  │
    │                       ▼                                  │
    │              ┌─────────────────┐                         │
    │              │    RETRIEVE     │                         │
    │              │    similar      │                         │
    │              │    experiences  │                         │
    │              └────────┬────────┘                         │
    │                       │                                  │
    │                       ▼                                  │
    │              ┌─────────────────┐                         │
    │              │     APPLY       │                         │
    │              │    lessons      │                         │
    │              │    learned      │                         │
    │              └────────┬────────┘                         │
    │                       │                                  │
    │                       ▼                                  │
    │                 NEW EXPERIENCE                           │
    │                       │                                  │
    │                       └──────────────────────────────────┤
    │                                                          │
    └──────────────────────────────────────────────────────────┘

Simple rules:
- Store experiences with outcomes
- Retrieve similar past experiences
- Use past outcomes to guide current decisions

Emergent behavior:
- Agent improves over time
- Agent develops "intuition" (pattern matching)
- Agent avoids past mistakes
- Agent reuses successful strategies
```

### Individual Memories, Collective Intelligence

In multi-agent systems, memory enables collective learning:

```
COLLECTIVE MEMORY EMERGENCE

Agent A learns: "API X requires authentication"
     │
     └──► Shares to collective memory

Agent B encounters API X
     │
     └──► Retrieves: "API X requires authentication"
     │
     └──► Avoids the mistake Agent A made

Result: The TEAM learns, even though individual
        agents never directly communicated

This is emergent collective intelligence!
```

---

## Short-Term Memory: Conversation and Working Memory

Short-term memory keeps track of the immediate context.

### Conversation Buffer Memory

The simplest form—store all recent messages.

```python
# code/04_conversation_buffer.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

class ConversationAgent:
    """Agent with simple conversation buffer memory."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the conversation history to provide contextual responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def chat(self, user_input: str) -> str:
        """Process user input with conversation memory."""

        # Get history
        history = self.memory.load_memory_variables({})

        # Generate response
        response = self.chain.invoke({
            "chat_history": history.get("chat_history", []),
            "input": user_input
        })

        # Save to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

        return response.content

    def get_history(self) -> list:
        """Get the conversation history."""
        return self.memory.load_memory_variables({}).get("chat_history", [])


# Demo
if __name__ == "__main__":
    agent = ConversationAgent()

    conversations = [
        "My name is Alice and I'm a software engineer.",
        "I work mainly with Python and TypeScript.",
        "What's my name and what languages do I use?"
    ]

    print("=" * 60)
    print("CONVERSATION BUFFER MEMORY DEMO")
    print("=" * 60)

    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = agent.chat(user_msg)
        print(f"Agent: {response}")
```

### Conversation Buffer Window Memory

Keep only the last N messages to manage context size.

```python
# code/04_buffer_window.py

from langchain.memory import ConversationBufferWindowMemory

class WindowedMemoryAgent:
    """Agent with windowed conversation memory."""

    def __init__(self, window_size: int = 5):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Only keep last N exchanges
        self.memory = ConversationBufferWindowMemory(
            k=window_size,  # Number of exchanges to keep
            return_messages=True,
            memory_key="chat_history"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
            You have access to recent conversation history.
            If asked about something not in recent history, acknowledge you may not have that context."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def chat(self, user_input: str) -> str:
        """Process user input with windowed memory."""

        history = self.memory.load_memory_variables({})

        response = self.chain.invoke({
            "chat_history": history.get("chat_history", []),
            "input": user_input
        })

        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

        return response.content
```

### Conversation Summary Memory

Summarize old conversations to preserve meaning while saving space.

```python
# code/04_summary_memory.py

from langchain.memory import ConversationSummaryBufferMemory

class SummaryMemoryAgent:
    """Agent that summarizes old conversations."""

    def __init__(self, max_token_limit: int = 1000):
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

        response = self.llm.invoke(
            self.prompt.format_messages(
                chat_history=history.get("chat_history", []),
                input=user_input
            )
        )

        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

        return response.content

    def get_summary(self) -> str:
        """Get the current conversation summary."""
        return self.memory.moving_summary_buffer
```

### Working Memory: Structured Short-Term State

For complex tasks, maintain structured working memory.

```python
# code/04_working_memory.py

from typing import TypedDict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class WorkingMemoryItem:
    """A single item in working memory."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5

class WorkingMemory:
    """Structured working memory for agents."""

    def __init__(self, capacity: int = 7):
        """Initialize with Miller's magic number as default."""
        self.capacity = capacity
        self.items: dict[str, WorkingMemoryItem] = {}

    def store(self, key: str, value: Any, importance: float = 0.5):
        """Store an item in working memory."""

        # If at capacity, evict least important/accessed item
        if len(self.items) >= self.capacity and key not in self.items:
            self._evict_one()

        self.items[key] = WorkingMemoryItem(
            key=key,
            value=value,
            importance=importance
        )

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve an item from working memory."""
        if key in self.items:
            self.items[key].access_count += 1
            return self.items[key].value
        return None

    def _evict_one(self):
        """Evict the least important item."""
        if not self.items:
            return

        # Score by importance and recency
        def score(item: WorkingMemoryItem) -> float:
            recency = (datetime.now() - item.created_at).total_seconds()
            return item.importance + (item.access_count * 0.1) - (recency * 0.001)

        # Find lowest scored item
        lowest_key = min(self.items.keys(), key=lambda k: score(self.items[k]))
        del self.items[lowest_key]

    def get_context(self) -> str:
        """Get working memory as context string."""
        if not self.items:
            return "Working memory is empty."

        lines = ["Current working memory:"]
        for key, item in self.items.items():
            lines.append(f"- {key}: {item.value}")

        return "\n".join(lines)

    def clear(self):
        """Clear working memory."""
        self.items.clear()


class WorkingMemoryAgent:
    """Agent with structured working memory."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.working_memory = WorkingMemory(capacity=7)
        self.conversation_history = []

    def process(self, user_input: str) -> str:
        """Process input using working memory."""

        # Extract entities/facts to potentially store
        self._update_working_memory(user_input)

        # Build prompt with working memory context
        prompt = f"""
        {self.working_memory.get_context()}

        Recent conversation:
        {self._format_history()}

        User: {user_input}

        Respond helpfully, using the working memory context when relevant.
        """

        response = self.llm.invoke(prompt)

        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })

        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return response.content

    def _update_working_memory(self, user_input: str):
        """Extract and store relevant information."""

        # Simple extraction (in production, use NER or structured extraction)
        extraction_prompt = f"""
        Extract any important facts from this message that should be remembered.
        Format as key:value pairs, one per line.
        If nothing important, respond with NONE.

        Message: {user_input}
        """

        response = self.llm.invoke(extraction_prompt)

        if "NONE" not in response.content:
            for line in response.content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    self.working_memory.store(
                        key.strip(),
                        value.strip(),
                        importance=0.7
                    )

    def _format_history(self) -> str:
        """Format conversation history."""
        if not self.conversation_history:
            return "No previous conversation."

        return "\n".join(
            f"{msg['role'].title()}: {msg['content']}"
            for msg in self.conversation_history[-6:]
        )
```

---

## Long-Term Memory: Persistent Knowledge

Long-term memory persists across sessions and enables learning over time.

### Vector Store Memory (Semantic Memory)

Store embeddings for semantic search.

```python
# code/05_vector_memory.py

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datetime import datetime
from typing import List
import uuid

class SemanticMemory:
    """Long-term semantic memory using vector embeddings."""

    def __init__(self, collection_name: str = "agent_memory"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def store(self, content: str, metadata: dict = None):
        """Store content in semantic memory."""

        # Create document with metadata
        doc_metadata = {
            "stored_at": datetime.now().isoformat(),
            "memory_id": str(uuid.uuid4()),
            **(metadata or {})
        }

        # Split if content is long
        chunks = self.text_splitter.split_text(content)

        documents = [
            Document(page_content=chunk, metadata=doc_metadata)
            for chunk in chunks
        ]

        # Add to vector store
        self.vectorstore.add_documents(documents)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant memories."""
        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve with relevance scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)


class SemanticMemoryAgent:
    """Agent with semantic long-term memory."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory = SemanticMemory()
        self.conversation_buffer = []

    def chat(self, user_input: str) -> str:
        """Chat with semantic memory retrieval."""

        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(user_input, k=3)

        # Format memories for context
        memory_context = ""
        if relevant_memories:
            memory_context = "Relevant memories:\n" + "\n".join(
                f"- {doc.page_content}"
                for doc in relevant_memories
            )

        # Build prompt
        prompt = f"""
        {memory_context}

        Current conversation:
        {self._format_buffer()}

        User: {user_input}

        Respond helpfully. If the memories are relevant, use them. If not, ignore them.
        """

        response = self.llm.invoke(prompt)

        # Store this interaction
        self._store_interaction(user_input, response.content)

        # Update buffer
        self.conversation_buffer.append({"user": user_input, "assistant": response.content})
        if len(self.conversation_buffer) > 5:
            self.conversation_buffer = self.conversation_buffer[-5:]

        return response.content

    def _store_interaction(self, user_input: str, response: str):
        """Store the interaction in long-term memory."""

        # Decide if this interaction is worth storing
        importance_check = self.llm.invoke(f"""
        Rate the importance of storing this interaction (1-10).
        Only respond with a number.

        User: {user_input}
        Response: {response}
        """)

        try:
            importance = int(importance_check.content.strip())
        except:
            importance = 5

        if importance >= 6:
            self.memory.store(
                f"User asked: {user_input}\nResponse: {response}",
                metadata={"importance": importance, "type": "interaction"}
            )

    def _format_buffer(self) -> str:
        if not self.conversation_buffer:
            return "No recent conversation."

        return "\n".join(
            f"User: {ex['user']}\nAssistant: {ex['assistant']}"
            for ex in self.conversation_buffer
        )

    def teach(self, knowledge: str):
        """Explicitly teach the agent something."""
        self.memory.store(
            knowledge,
            metadata={"type": "taught", "importance": 9}
        )
        return f"Learned: {knowledge[:100]}..."
```

### Episodic Memory

Store complete episodes/experiences for later recall.

```python
# code/05_episodic_memory.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import json

@dataclass
class Episode:
    """A complete episode/experience."""
    episode_id: str
    timestamp: datetime
    context: str  # What was happening
    actions: List[str]  # What the agent did
    outcome: str  # What happened
    success: bool  # Was it successful
    lessons: List[str] = field(default_factory=list)  # What was learned
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "actions": self.actions,
            "outcome": self.outcome,
            "success": self.success,
            "lessons": self.lessons,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Episode':
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class EpisodicMemory:
    """Memory system for storing and retrieving episodes."""

    def __init__(self):
        self.episodes: List[Episode] = []
        self.semantic_memory = SemanticMemory(collection_name="episodic")

    def record_episode(self, episode: Episode):
        """Record a new episode."""
        self.episodes.append(episode)

        # Also store in semantic memory for retrieval
        episode_text = f"""
        Context: {episode.context}
        Actions: {', '.join(episode.actions)}
        Outcome: {episode.outcome}
        Success: {episode.success}
        Lessons: {', '.join(episode.lessons)}
        """

        self.semantic_memory.store(
            episode_text,
            metadata={
                "episode_id": episode.episode_id,
                "success": episode.success,
                "tags": ",".join(episode.tags)
            }
        )

    def recall_similar(self, situation: str, k: int = 3) -> List[Episode]:
        """Recall episodes similar to the current situation."""
        docs = self.semantic_memory.retrieve(situation, k=k)

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
        similar_episodes = self.recall_similar(situation, k=5)

        lessons = []
        for ep in similar_episodes:
            if ep.success:
                lessons.append(f"SUCCESS: {', '.join(ep.lessons)}")
            else:
                lessons.append(f"FAILURE (avoid): {', '.join(ep.lessons)}")

        return lessons


class EpisodicAgent:
    """Agent that learns from past episodes."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.episodic_memory = EpisodicMemory()
        self.current_episode_actions = []

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

        Complete the task, applying lessons from past experiences.
        """

        response = self.llm.invoke(prompt)

        self.current_episode_actions.append(f"Responded: {response.content[:100]}")

        return response.content

    def complete_episode(self, context: str, outcome: str, success: bool):
        """Complete and record the current episode."""

        # Extract lessons
        lessons_prompt = f"""
        Context: {context}
        Actions taken: {self.current_episode_actions}
        Outcome: {outcome}
        Success: {success}

        What lessons should be learned from this experience?
        List 1-3 key lessons, one per line.
        """

        lessons_response = self.llm.invoke(lessons_prompt)
        lessons = [l.strip() for l in lessons_response.content.split("\n") if l.strip()]

        episode = Episode(
            episode_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context=context,
            actions=self.current_episode_actions,
            outcome=outcome,
            success=success,
            lessons=lessons
        )

        self.episodic_memory.record_episode(episode)
        self.current_episode_actions = []

        return episode
```

---

## Shared Memory for Multi-Agent Systems

When multiple agents need to coordinate, they need shared memory.

### Blackboard Pattern Implementation

```python
# code/06_shared_memory.py

from typing import TypedDict, Annotated, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
import operator

@dataclass
class BlackboardEntry:
    """An entry on the shared blackboard."""
    key: str
    value: Any
    author: str
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    tags: List[str] = field(default_factory=list)

class SharedBlackboard:
    """Thread-safe shared blackboard for multi-agent coordination."""

    def __init__(self):
        self.entries: dict[str, BlackboardEntry] = {}
        self.history: List[BlackboardEntry] = []
        self.lock = Lock()
        self.subscribers: dict[str, List[callable]] = {}

    def write(self, key: str, value: Any, author: str, tags: List[str] = None):
        """Write to the blackboard."""
        with self.lock:
            version = 1
            if key in self.entries:
                version = self.entries[key].version + 1

            entry = BlackboardEntry(
                key=key,
                value=value,
                author=author,
                version=version,
                tags=tags or []
            )

            self.entries[key] = entry
            self.history.append(entry)

            # Notify subscribers
            self._notify_subscribers(key, entry)

    def read(self, key: str) -> Optional[Any]:
        """Read from the blackboard."""
        with self.lock:
            if key in self.entries:
                return self.entries[key].value
            return None

    def read_by_author(self, author: str) -> List[BlackboardEntry]:
        """Read all entries by a specific author."""
        with self.lock:
            return [e for e in self.entries.values() if e.author == author]

    def read_by_tag(self, tag: str) -> List[BlackboardEntry]:
        """Read all entries with a specific tag."""
        with self.lock:
            return [e for e in self.entries.values() if tag in e.tags]

    def subscribe(self, key: str, callback: callable):
        """Subscribe to changes on a key."""
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(callback)

    def _notify_subscribers(self, key: str, entry: BlackboardEntry):
        """Notify subscribers of a change."""
        if key in self.subscribers:
            for callback in self.subscribers[key]:
                callback(entry)

    def get_state_summary(self) -> str:
        """Get a summary of the current blackboard state."""
        with self.lock:
            lines = ["Blackboard State:"]
            for key, entry in self.entries.items():
                lines.append(f"  [{entry.author}] {key}: {str(entry.value)[:100]}")
            return "\n".join(lines)


class BlackboardAgent:
    """Agent that communicates via shared blackboard."""

    def __init__(self, name: str, blackboard: SharedBlackboard):
        self.name = name
        self.blackboard = blackboard
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def contribute(self, task: str, contribution_type: str):
        """Make a contribution to the blackboard."""

        # Read current state
        state = self.blackboard.get_state_summary()

        prompt = f"""
        You are {self.name}. Your task is to contribute {contribution_type} to a shared project.

        Current blackboard state:
        {state}

        Task: {task}

        Provide your {contribution_type} contribution.
        """

        response = self.llm.invoke(prompt)

        # Write to blackboard
        self.blackboard.write(
            key=f"{self.name}_{contribution_type}",
            value=response.content,
            author=self.name,
            tags=[contribution_type, "contribution"]
        )

        return response.content

    def review_and_respond(self, target_author: str):
        """Review another agent's work and respond."""

        # Get their contributions
        their_work = self.blackboard.read_by_author(target_author)

        if not their_work:
            return "No work found to review."

        work_summary = "\n".join(
            f"- {e.key}: {str(e.value)[:200]}"
            for e in their_work
        )

        prompt = f"""
        You are {self.name}. Review {target_author}'s work and provide feedback.

        Their work:
        {work_summary}

        Provide constructive feedback or build upon their work.
        """

        response = self.llm.invoke(prompt)

        # Write review to blackboard
        self.blackboard.write(
            key=f"{self.name}_review_of_{target_author}",
            value=response.content,
            author=self.name,
            tags=["review", target_author]
        )

        return response.content


# Demo: Multi-agent collaboration via blackboard
def demo_blackboard_collaboration():
    """Demonstrate multi-agent collaboration using shared blackboard."""

    blackboard = SharedBlackboard()

    # Create agents
    researcher = BlackboardAgent("Researcher", blackboard)
    analyst = BlackboardAgent("Analyst", blackboard)
    writer = BlackboardAgent("Writer", blackboard)

    task = "Create a brief report on renewable energy trends"

    print("=" * 60)
    print("BLACKBOARD COLLABORATION DEMO")
    print("=" * 60)

    # Phase 1: Research
    print("\n--- Phase 1: Research ---")
    researcher.contribute(task, "research_findings")
    print(f"Researcher contributed: {blackboard.read('Researcher_research_findings')[:200]}...")

    # Phase 2: Analysis
    print("\n--- Phase 2: Analysis ---")
    analyst.review_and_respond("Researcher")
    analyst.contribute(task, "analysis")
    print(f"Analyst contributed: {blackboard.read('Analyst_analysis')[:200]}...")

    # Phase 3: Writing
    print("\n--- Phase 3: Writing ---")
    writer.review_and_respond("Analyst")
    writer.contribute(task, "draft_report")
    print(f"Writer contributed: {blackboard.read('Writer_draft_report')[:200]}...")

    # Final state
    print("\n--- Final Blackboard State ---")
    print(blackboard.get_state_summary())
```

### Message Queue Memory

For asynchronous multi-agent communication.

```python
# code/06_message_queue.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from collections import deque
from enum import Enum

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentMessage:
    """A message between agents."""
    sender: str
    receiver: str  # Can be "*" for broadcast
    content: str
    message_type: str
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For request-response pairing

class MessageQueue:
    """Shared message queue for agent communication."""

    def __init__(self):
        self.queues: dict[str, deque] = {}
        self.broadcast_queue: deque = deque(maxlen=100)

    def send(self, message: AgentMessage):
        """Send a message."""
        if message.receiver == "*":
            # Broadcast
            self.broadcast_queue.append(message)
        else:
            # Direct message
            if message.receiver not in self.queues:
                self.queues[message.receiver] = deque()

            self.queues[message.receiver].append(message)

    def receive(self, agent_name: str, block: bool = False) -> Optional[AgentMessage]:
        """Receive a message for an agent."""

        # Check direct messages first (prioritized)
        if agent_name in self.queues and self.queues[agent_name]:
            # Get highest priority message
            messages = list(self.queues[agent_name])
            messages.sort(key=lambda m: m.priority.value, reverse=True)
            msg = messages[0]
            self.queues[agent_name].remove(msg)
            return msg

        # Check broadcasts
        for msg in self.broadcast_queue:
            if msg.sender != agent_name:  # Don't receive own broadcasts
                return msg

        return None

    def receive_all(self, agent_name: str) -> List[AgentMessage]:
        """Receive all pending messages for an agent."""
        messages = []

        # Direct messages
        if agent_name in self.queues:
            messages.extend(self.queues[agent_name])
            self.queues[agent_name].clear()

        # Broadcasts (from others)
        messages.extend([
            m for m in self.broadcast_queue
            if m.sender != agent_name
        ])

        # Sort by priority and time
        messages.sort(
            key=lambda m: (m.priority.value, m.timestamp.timestamp()),
            reverse=True
        )

        return messages


class MessageQueueAgent:
    """Agent that communicates via message queue."""

    def __init__(self, name: str, queue: MessageQueue):
        self.name = name
        self.queue = queue
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def send_message(self, receiver: str, content: str,
                     msg_type: str = "info",
                     priority: MessagePriority = MessagePriority.NORMAL):
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=msg_type,
            priority=priority
        )
        self.queue.send(message)

    def broadcast(self, content: str, msg_type: str = "announcement"):
        """Broadcast a message to all agents."""
        message = AgentMessage(
            sender=self.name,
            receiver="*",
            content=content,
            message_type=msg_type
        )
        self.queue.send(message)

    def check_messages(self) -> List[AgentMessage]:
        """Check for new messages."""
        return self.queue.receive_all(self.name)

    def process_messages(self):
        """Process all pending messages."""
        messages = self.check_messages()

        responses = []
        for msg in messages:
            response = self._handle_message(msg)
            if response:
                responses.append(response)

        return responses

    def _handle_message(self, message: AgentMessage) -> Optional[str]:
        """Handle a single message."""

        prompt = f"""
        You are {self.name}. You received a message:

        From: {message.sender}
        Type: {message.message_type}
        Priority: {message.priority.name}
        Content: {message.content}

        How do you respond or act on this message?
        """

        response = self.llm.invoke(prompt)

        # If requires response, send one back
        if message.requires_response:
            self.send_message(
                receiver=message.sender,
                content=response.content,
                msg_type="response",
                priority=message.priority
            )

        return response.content
```

---

## Memory Retrieval Strategies

How you retrieve memories is as important as what you store.

### Recency-Based Retrieval

```python
def retrieve_by_recency(memories: List[Memory], k: int = 5) -> List[Memory]:
    """Retrieve the most recent memories."""
    sorted_memories = sorted(
        memories,
        key=lambda m: m.timestamp,
        reverse=True
    )
    return sorted_memories[:k]
```

### Relevance-Based Retrieval

```python
def retrieve_by_relevance(
    query: str,
    vector_store: VectorStore,
    k: int = 5
) -> List[Document]:
    """Retrieve the most relevant memories."""
    return vector_store.similarity_search(query, k=k)
```

### Importance-Based Retrieval

```python
def retrieve_by_importance(
    memories: List[Memory],
    k: int = 5
) -> List[Memory]:
    """Retrieve the most important memories."""
    sorted_memories = sorted(
        memories,
        key=lambda m: m.importance,
        reverse=True
    )
    return sorted_memories[:k]
```

### Hybrid Retrieval (Best Practice)

```python
# code/05_hybrid_retrieval.py

from dataclasses import dataclass
from datetime import datetime
from typing import List
import math

@dataclass
class ScoredMemory:
    memory: any
    relevance_score: float
    recency_score: float
    importance_score: float
    final_score: float

class HybridRetriever:
    """Combines multiple retrieval strategies."""

    def __init__(
        self,
        relevance_weight: float = 0.5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2
    ):
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight

    def retrieve(
        self,
        query: str,
        memories: List,
        vector_store,
        k: int = 5
    ) -> List[ScoredMemory]:
        """Retrieve using hybrid scoring."""

        # Get relevance scores from vector search
        relevance_results = vector_store.similarity_search_with_score(query, k=k*2)
        relevance_map = {
            doc.metadata.get("memory_id"): score
            for doc, score in relevance_results
        }

        # Score all memories
        scored = []
        now = datetime.now()

        for memory in memories:
            # Relevance (from vector search, normalized)
            relevance = relevance_map.get(memory.id, 0)
            relevance_norm = 1 / (1 + relevance)  # Convert distance to similarity

            # Recency (exponential decay)
            age_hours = (now - memory.timestamp).total_seconds() / 3600
            recency = math.exp(-age_hours / 24)  # Half-life of 24 hours

            # Importance (already normalized 0-1)
            importance = getattr(memory, 'importance', 0.5)

            # Combined score
            final = (
                self.relevance_weight * relevance_norm +
                self.recency_weight * recency +
                self.importance_weight * importance
            )

            scored.append(ScoredMemory(
                memory=memory,
                relevance_score=relevance_norm,
                recency_score=recency,
                importance_score=importance,
                final_score=final
            ))

        # Sort by final score and return top k
        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored[:k]
```

---

## Complete Memory-Enhanced Agent

Let's put it all together into a complete memory-enhanced agent.

```python
# code/05_complete_memory_agent.py

from typing import Optional, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime
import uuid

class CompleteMemoryAgent:
    """
    Agent with complete memory system:
    - Short-term: Conversation buffer
    - Working: Structured current context
    - Long-term: Semantic vector store + episodic memory
    """

    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Short-term: conversation buffer (last 10 exchanges)
        self.conversation_memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="chat_history"
        )

        # Working memory: current task context
        self.working_memory = {}

        # Long-term: semantic memory (vector store)
        self.semantic_memory = Chroma(
            collection_name=f"agent_{self.agent_id}",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"./memory_{self.agent_id}"
        )

        # Long-term: episodic memory (experiences)
        self.episodes = []

    def chat(self, user_input: str) -> str:
        """Process user input with full memory system."""

        # 1. Retrieve relevant long-term memories
        relevant_memories = self._retrieve_memories(user_input)

        # 2. Get conversation history
        history = self.conversation_memory.load_memory_variables({})

        # 3. Build context
        context = self._build_context(
            user_input,
            relevant_memories,
            history.get("chat_history", [])
        )

        # 4. Generate response
        response = self.llm.invoke(context)

        # 5. Update memories
        self._update_memories(user_input, response.content)

        return response.content

    def _retrieve_memories(self, query: str) -> List[str]:
        """Retrieve relevant memories from long-term storage."""

        memories = []

        # Semantic search
        try:
            docs = self.semantic_memory.similarity_search(query, k=3)
            memories.extend([doc.page_content for doc in docs])
        except:
            pass  # Empty store on first run

        # Episodic search (simple keyword matching for demo)
        for episode in self.episodes[-10:]:  # Last 10 episodes
            if any(word in episode.get("context", "").lower()
                   for word in query.lower().split()):
                memories.append(f"Past experience: {episode.get('lesson', '')}")

        return memories

    def _build_context(self, user_input: str, memories: List[str], history: list) -> str:
        """Build the full context for the LLM."""

        # Format memories
        memories_text = ""
        if memories:
            memories_text = "Relevant memories:\n" + "\n".join(
                f"- {m}" for m in memories[:5]
            )

        # Format working memory
        working_text = ""
        if self.working_memory:
            working_text = "Current context:\n" + "\n".join(
                f"- {k}: {v}" for k, v in self.working_memory.items()
            )

        # Format history
        history_text = ""
        if history:
            history_text = "Recent conversation:\n" + "\n".join(
                f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content}"
                for i, msg in enumerate(history[-6:])
            )

        prompt = f"""
        You are a helpful AI assistant with memory capabilities.

        {memories_text}

        {working_text}

        {history_text}

        User: {user_input}

        Provide a helpful response, using your memories when relevant.
        If you learn something new and important, acknowledge it.
        """

        return prompt

    def _update_memories(self, user_input: str, response: str):
        """Update all memory systems."""

        # Update conversation memory
        self.conversation_memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        # Extract and store important information
        self._extract_and_store(user_input, response)

    def _extract_and_store(self, user_input: str, response: str):
        """Extract important information and store in long-term memory."""

        extraction_prompt = f"""
        Analyze this exchange and identify any important facts to remember.

        User: {user_input}
        Response: {response}

        If there are important facts (names, preferences, information shared),
        list them one per line.
        If nothing important, respond with: NONE
        """

        extraction = self.llm.invoke(extraction_prompt)

        if "NONE" not in extraction.content:
            # Store in semantic memory
            self.semantic_memory.add_texts(
                texts=[extraction.content],
                metadatas=[{
                    "timestamp": datetime.now().isoformat(),
                    "type": "extracted_fact"
                }]
            )

    def teach(self, knowledge: str) -> str:
        """Explicitly teach the agent something."""
        self.semantic_memory.add_texts(
            texts=[knowledge],
            metadatas=[{
                "timestamp": datetime.now().isoformat(),
                "type": "taught_knowledge",
                "importance": "high"
            }]
        )
        return f"I've learned: {knowledge}"

    def set_context(self, key: str, value: str):
        """Set working memory context."""
        self.working_memory[key] = value

    def clear_context(self):
        """Clear working memory."""
        self.working_memory.clear()

    def record_episode(self, context: str, outcome: str, lesson: str):
        """Record an episode for future reference."""
        self.episodes.append({
            "context": context,
            "outcome": outcome,
            "lesson": lesson,
            "timestamp": datetime.now().isoformat()
        })


# Demo
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("COMPLETE MEMORY AGENT DEMO")
    print("=" * 60)

    agent = CompleteMemoryAgent()

    # Teach the agent
    print("\n--- Teaching Phase ---")
    print(agent.teach("The user prefers concise responses."))
    print(agent.teach("When discussing code, always include examples."))

    # Set context
    agent.set_context("current_project", "Building a multi-agent system")
    agent.set_context("user_expertise", "intermediate Python developer")

    # Conversation
    print("\n--- Conversation ---")

    exchanges = [
        "Hi, my name is Alex and I'm working on an AI project.",
        "I'm trying to implement memory for my agents.",
        "What's my name and what am I working on?",
        "Can you suggest a good approach for agent memory?"
    ]

    for msg in exchanges:
        print(f"\nUser: {msg}")
        response = agent.chat(msg)
        print(f"Agent: {response}")

    # Record episode
    agent.record_episode(
        context="User asked about agent memory",
        outcome="Provided comprehensive answer",
        lesson="User appreciates detailed technical explanations"
    )

    print("\n" + "=" * 60)
    print("Memory system demonstration complete!")
```

---

## Key Takeaways

### 1. Memory Transforms Agents from Stateless to Intelligent
Without memory, agents forget everything. With memory, they can learn, adapt, and provide personalized experiences.

### 2. Different Memory Types Serve Different Purposes
- **Short-term**: Current conversation context
- **Working**: Structured task-specific state
- **Long-term semantic**: Retrievable knowledge
- **Long-term episodic**: Past experiences and lessons

### 3. Memory is Storage + Retrieval + Injection
All three components must work together. Great storage with poor retrieval is useless.

### 4. Hybrid Retrieval Beats Single Strategies
Combine recency, relevance, and importance for the best results.

### 5. Shared Memory Enables Multi-Agent Coordination
Blackboard patterns and message queues let agents collaborate without direct communication.

### 6. Memory Enables Emergent Learning
With memory, simple interaction rules can produce complex, adaptive behavior over time.

---

## What's Next?

In **Module 5.3: Building Agent Teams**, we'll bring everything together:
- Combining collaboration patterns with memory systems
- Building production-ready agent teams
- Monitoring and debugging multi-agent systems
- Real-world case study: A complete research team

You now have the building blocks. Let's assemble the team!

[Continue to Module 5.3 →](03_building_agent_teams.md)
