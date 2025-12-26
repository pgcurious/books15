"""
Module 5.1: Communication Patterns for Multi-Agent Systems
============================================================
Demonstrates different communication patterns:
- Direct Messaging
- Publish/Subscribe (Pub/Sub)
- Blackboard Pattern
"""

from typing import TypedDict, List, Callable, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
import time
import uuid

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# PATTERN 1: DIRECT MESSAGING
# ============================================================

@dataclass
class DirectMessage:
    """A direct message between two agents."""
    sender: str
    receiver: str
    content: str
    message_type: str = "info"  # "request", "response", "info"
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


class DirectMessageAgent:
    """Agent that uses direct messaging."""

    def __init__(self, name: str):
        self.name = name
        self.inbox: List[DirectMessage] = []
        self.outbox: List[DirectMessage] = []
        self.peers: dict[str, 'DirectMessageAgent'] = {}

    def register_peer(self, peer: 'DirectMessageAgent'):
        """Register a peer agent for direct communication."""
        self.peers[peer.name] = peer
        peer.peers[self.name] = self

    def send(self, receiver: str, content: str, msg_type: str = "info") -> bool:
        """Send a direct message to another agent."""
        if receiver not in self.peers:
            print(f"[{self.name}] Error: Unknown peer '{receiver}'")
            return False

        message = DirectMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=msg_type,
            correlation_id=str(uuid.uuid4())
        )

        self.outbox.append(message)
        self.peers[receiver].inbox.append(message)
        print(f"[{self.name}] Sent to {receiver}: {content[:50]}...")
        return True

    def receive(self) -> Optional[DirectMessage]:
        """Receive the next message from inbox."""
        if self.inbox:
            return self.inbox.pop(0)
        return None

    def receive_all(self) -> List[DirectMessage]:
        """Receive all messages from inbox."""
        messages = self.inbox.copy()
        self.inbox.clear()
        return messages


def demo_direct_messaging():
    """Demonstrate direct messaging between agents."""
    print("=" * 60)
    print("DEMO 1: Direct Messaging")
    print("=" * 60)

    # Create agents
    alice = DirectMessageAgent("Alice")
    bob = DirectMessageAgent("Bob")
    charlie = DirectMessageAgent("Charlie")

    # Register peers
    alice.register_peer(bob)
    alice.register_peer(charlie)
    bob.register_peer(charlie)

    # Send messages
    print("\n--- Sending Messages ---")
    alice.send("Bob", "Can you help me analyze this data?", "request")
    bob.send("Alice", "Sure, send me the details.", "response")
    alice.send("Charlie", "FYI: Bob is helping with analysis.", "info")

    # Receive messages
    print("\n--- Receiving Messages ---")
    for msg in bob.receive_all():
        print(f"[Bob] Received from {msg.sender}: {msg.content}")

    for msg in charlie.receive_all():
        print(f"[Charlie] Received from {msg.sender}: {msg.content}")

    print()


# ============================================================
# PATTERN 2: PUBLISH/SUBSCRIBE
# ============================================================

class MessageBus:
    """Publish-subscribe message bus for agent communication."""

    def __init__(self):
        self.subscribers: dict[str, List[Callable]] = defaultdict(list)
        self.message_log: List[dict] = []

    def subscribe(self, channel: str, handler: Callable):
        """Subscribe to a channel."""
        self.subscribers[channel].append(handler)
        print(f"  Subscribed handler to channel: {channel}")

    def publish(self, channel: str, message: Any, sender: str):
        """Publish a message to a channel."""
        full_message = {
            "channel": channel,
            "sender": sender,
            "content": message,
            "timestamp": datetime.now()
        }
        self.message_log.append(full_message)

        # Notify all subscribers
        for handler in self.subscribers[channel]:
            handler(full_message)

        print(f"  [{sender}] Published to #{channel}: {str(message)[:50]}...")


class PubSubAgent:
    """Agent that communicates via publish-subscribe."""

    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus = bus
        self.received_messages: List[dict] = []

    def subscribe_to(self, channel: str):
        """Subscribe to a channel."""
        self.bus.subscribe(channel, self._handle_message)

    def publish_to(self, channel: str, message: Any):
        """Publish to a channel."""
        self.bus.publish(channel, message, self.name)

    def _handle_message(self, message: dict):
        """Handle incoming message."""
        if message["sender"] != self.name:  # Don't process own messages
            self.received_messages.append(message)
            self.on_message(message)

    def on_message(self, message: dict):
        """Override in subclass to handle messages."""
        print(f"  [{self.name}] Received from #{message['channel']}: {message['content']}")


def demo_pubsub():
    """Demonstrate publish-subscribe messaging."""
    print("=" * 60)
    print("DEMO 2: Publish/Subscribe")
    print("=" * 60)

    # Create message bus
    bus = MessageBus()

    # Create agents
    print("\n--- Setting Up Agents ---")
    researcher = PubSubAgent("Researcher", bus)
    analyst = PubSubAgent("Analyst", bus)
    writer = PubSubAgent("Writer", bus)
    monitor = PubSubAgent("Monitor", bus)

    # Set up subscriptions
    analyst.subscribe_to("research")
    writer.subscribe_to("analysis")
    monitor.subscribe_to("research")
    monitor.subscribe_to("analysis")
    monitor.subscribe_to("output")

    # Publish messages
    print("\n--- Publishing Messages ---")
    researcher.publish_to("research", {"topic": "AI Safety", "findings": ["point1", "point2"]})
    analyst.publish_to("analysis", {"insights": ["insight1"], "confidence": 0.85})
    writer.publish_to("output", {"draft": "Final report content..."})

    # Show what each agent received
    print("\n--- Messages Received ---")
    for agent in [researcher, analyst, writer, monitor]:
        print(f"  {agent.name}: {len(agent.received_messages)} messages")

    print()


# ============================================================
# PATTERN 3: BLACKBOARD
# ============================================================

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
        self.watchers: dict[str, List[Callable]] = defaultdict(list)

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

            # Notify watchers
            self._notify_watchers(key, entry)

            print(f"  [{author}] Wrote to blackboard: {key} = {str(value)[:50]}...")

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

    def watch(self, key: str, callback: Callable):
        """Watch for changes to a key."""
        self.watchers[key].append(callback)

    def _notify_watchers(self, key: str, entry: BlackboardEntry):
        """Notify watchers of a change."""
        for callback in self.watchers[key]:
            callback(entry)

    def get_state_summary(self) -> str:
        """Get a summary of the current blackboard state."""
        with self.lock:
            lines = ["Blackboard State:"]
            for key, entry in self.entries.items():
                lines.append(f"  [{entry.author}] {key}: {str(entry.value)[:50]}")
            return "\n".join(lines)


class BlackboardAgent:
    """Agent that communicates via shared blackboard."""

    def __init__(self, name: str, blackboard: SharedBlackboard):
        self.name = name
        self.blackboard = blackboard

    def contribute(self, key: str, value: Any, tags: List[str] = None):
        """Contribute to the blackboard."""
        self.blackboard.write(key, value, self.name, tags)

    def read_all_work(self) -> dict:
        """Read all work on the blackboard."""
        return {
            key: entry.value
            for key, entry in self.blackboard.entries.items()
        }

    def read_others_work(self) -> List[BlackboardEntry]:
        """Read work from other agents."""
        return [
            e for e in self.blackboard.entries.values()
            if e.author != self.name
        ]


def demo_blackboard():
    """Demonstrate blackboard pattern."""
    print("=" * 60)
    print("DEMO 3: Blackboard Pattern")
    print("=" * 60)

    # Create shared blackboard
    blackboard = SharedBlackboard()

    # Create agents
    researcher = BlackboardAgent("Researcher", blackboard)
    analyst = BlackboardAgent("Analyst", blackboard)
    writer = BlackboardAgent("Writer", blackboard)

    # Agents contribute to the blackboard
    print("\n--- Phase 1: Research ---")
    researcher.contribute(
        "research_findings",
        ["Finding 1: AI is transformative", "Finding 2: Safety is crucial"],
        tags=["research", "phase1"]
    )

    print("\n--- Phase 2: Analysis ---")
    # Analyst reads research and contributes analysis
    research = blackboard.read("research_findings")
    analyst.contribute(
        "analysis",
        {"summary": f"Analyzed {len(research)} findings", "confidence": 0.9},
        tags=["analysis", "phase2"]
    )

    print("\n--- Phase 3: Writing ---")
    # Writer reads all and contributes draft
    all_work = writer.read_all_work()
    writer.contribute(
        "draft_report",
        f"Report based on {len(all_work)} contributions...",
        tags=["writing", "phase3"]
    )

    # Show final state
    print("\n--- Final Blackboard State ---")
    print(blackboard.get_state_summary())

    print()


# ============================================================
# COMPARISON: WHEN TO USE EACH PATTERN
# ============================================================

def demo_comparison():
    """Compare when to use each pattern."""
    print("=" * 60)
    print("PATTERN COMPARISON")
    print("=" * 60)

    comparison = """
    DIRECT MESSAGING
    ────────────────
    Best for:
    • Clear sender-receiver relationships
    • Simple request-response patterns
    • Small number of agents (2-4)
    • When you need guaranteed delivery

    Example: Agent A asks Agent B to analyze data


    PUBLISH/SUBSCRIBE
    ─────────────────
    Best for:
    • Many agents that need to stay informed
    • Loose coupling between agents
    • Event-driven architectures
    • When multiple agents care about the same events

    Example: Research agent publishes findings,
             multiple agents (analyst, writer, monitor) receive


    BLACKBOARD
    ──────────
    Best for:
    • Complex state that multiple agents need
    • Phases where agents build on each other's work
    • Need for audit trail / history
    • Collaborative problem-solving

    Example: Research team building a report together
    """

    print(comparison)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-AGENT COMMUNICATION PATTERNS")
    print("=" * 60 + "\n")

    demo_direct_messaging()
    demo_pubsub()
    demo_blackboard()
    demo_comparison()

    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. DIRECT MESSAGING is best for simple, targeted communication

    2. PUBLISH/SUBSCRIBE decouples senders from receivers

    3. BLACKBOARD enables collaborative state building

    4. Choose based on your coordination needs:
       - Few agents, clear relationships → Direct
       - Many agents, events → Pub/Sub
       - Shared work product → Blackboard

    5. Patterns can be combined in a single system
    """)
