"""
Module 5.1: Delegation Patterns for Multi-Agent Systems
========================================================
Demonstrates different delegation strategies:
- Capability-Based Delegation
- Supervisor-Based Delegation
- Self-Delegation (Recursive)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# PATTERN 1: CAPABILITY-BASED DELEGATION
# ============================================================

@dataclass
class AgentCapability:
    """Describes a capability an agent has."""
    name: str
    description: str
    keywords: List[str]
    confidence_threshold: float = 0.7


class CapabilityAwareAgent:
    """Agent that knows its capabilities and can delegate based on them."""

    def __init__(self, name: str, capabilities: List[AgentCapability]):
        self.name = name
        self.capabilities = capabilities
        self.peer_registry: dict[str, 'CapabilityAwareAgent'] = {}

    def register_peer(self, peer: 'CapabilityAwareAgent'):
        """Register a peer agent."""
        self.peer_registry[peer.name] = peer
        print(f"  [{self.name}] Registered peer: {peer.name}")

    def can_handle(self, task: str) -> Tuple[bool, float, str]:
        """Check if this agent can handle a task."""
        task_lower = task.lower()

        best_capability = None
        best_confidence = 0.0

        for capability in self.capabilities:
            keyword_matches = sum(
                1 for kw in capability.keywords
                if kw in task_lower
            )
            if keyword_matches > 0:
                confidence = min(1.0, keyword_matches * 0.3 + 0.4)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_capability = capability

        if best_capability and best_confidence >= best_capability.confidence_threshold:
            return True, best_confidence, best_capability.name

        return False, 0.0, ""

    def find_best_peer(self, task: str) -> Optional['CapabilityAwareAgent']:
        """Find the best peer to handle a task."""
        best_peer = None
        best_confidence = 0.0

        for peer in self.peer_registry.values():
            can_handle, confidence, _ = peer.can_handle(task)
            if can_handle and confidence > best_confidence:
                best_peer = peer
                best_confidence = confidence

        return best_peer

    def process(self, task: str) -> str:
        """Process a task, delegating if necessary."""
        can_handle, confidence, capability = self.can_handle(task)

        if can_handle:
            print(f"  [{self.name}] Handling task using '{capability}' (confidence: {confidence:.2f})")
            return self._do_task(task)
        else:
            # Try to delegate
            peer = self.find_best_peer(task)
            if peer:
                print(f"  [{self.name}] Delegating to {peer.name}")
                return peer.process(task)
            else:
                print(f"  [{self.name}] Cannot handle and no peer available")
                return f"Unable to handle: {task}"

    def _do_task(self, task: str) -> str:
        """Actually perform the task (simulated)."""
        return f"[{self.name}] Completed: {task[:50]}..."


def demo_capability_delegation():
    """Demonstrate capability-based delegation."""
    print("=" * 60)
    print("DEMO 1: Capability-Based Delegation")
    print("=" * 60)

    # Create specialized agents
    print("\n--- Creating Agents ---")

    research_agent = CapabilityAwareAgent(
        name="Researcher",
        capabilities=[
            AgentCapability(
                name="web_search",
                description="Search the web for information",
                keywords=["search", "find", "look up", "research", "information"]
            ),
            AgentCapability(
                name="summarize",
                description="Summarize documents",
                keywords=["summarize", "summary", "overview", "brief"]
            )
        ]
    )

    code_agent = CapabilityAwareAgent(
        name="Coder",
        capabilities=[
            AgentCapability(
                name="write_code",
                description="Write code in various languages",
                keywords=["code", "program", "function", "implement", "python", "javascript"]
            ),
            AgentCapability(
                name="debug",
                description="Debug and fix code issues",
                keywords=["debug", "fix", "error", "bug", "issue"]
            )
        ]
    )

    math_agent = CapabilityAwareAgent(
        name="Mathematician",
        capabilities=[
            AgentCapability(
                name="calculate",
                description="Perform mathematical calculations",
                keywords=["calculate", "compute", "math", "equation", "formula"]
            ),
            AgentCapability(
                name="statistics",
                description="Statistical analysis",
                keywords=["statistics", "average", "mean", "correlation", "data analysis"]
            )
        ]
    )

    # Register peers
    print("\n--- Registering Peers ---")
    research_agent.register_peer(code_agent)
    research_agent.register_peer(math_agent)
    code_agent.register_peer(research_agent)
    code_agent.register_peer(math_agent)
    math_agent.register_peer(research_agent)
    math_agent.register_peer(code_agent)

    # Process various tasks
    tasks = [
        "Search for information about quantum computing",
        "Write a Python function to sort a list",
        "Calculate the average of these numbers: 10, 20, 30",
        "Debug this error in my code: TypeError",
        "Summarize this research paper",
        "Cook me a meal"  # No agent can handle this
    ]

    print("\n--- Processing Tasks ---")
    for task in tasks:
        print(f"\nTask: {task}")
        # Start with research agent
        result = research_agent.process(task)
        print(f"Result: {result}")

    print()


# ============================================================
# PATTERN 2: SUPERVISOR-BASED DELEGATION
# ============================================================

class SupervisorAgent:
    """Supervisor that routes tasks to appropriate workers."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.workers = {}

    def register_worker(self, name: str, description: str, handler):
        """Register a worker with the supervisor."""
        self.workers[name] = {
            "description": description,
            "handler": handler
        }
        print(f"  Registered worker: {name} - {description}")

    def route_task(self, task: str) -> str:
        """Route a task to the appropriate worker."""

        # Create worker descriptions for the LLM
        worker_list = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.workers.items()
        )

        prompt = ChatPromptTemplate.from_template("""
        You are a supervisor agent. Route this task to the best worker.

        Available workers:
        {workers}

        Task: {task}

        Respond with ONLY the worker name (e.g., "researcher").
        If no worker fits, respond with "none".
        """)

        response = self.llm.invoke(
            prompt.format(workers=worker_list, task=task)
        )

        selected = response.content.strip().lower()
        print(f"  [Supervisor] Routing to: {selected}")

        if selected in self.workers:
            return self.workers[selected]["handler"](task)
        elif selected == "none":
            return "No appropriate worker found for this task."
        else:
            # Fuzzy match
            for name in self.workers:
                if name in selected:
                    return self.workers[name]["handler"](task)
            return f"Unknown worker: {selected}"


def demo_supervisor_delegation():
    """Demonstrate supervisor-based delegation."""
    print("=" * 60)
    print("DEMO 2: Supervisor-Based Delegation")
    print("=" * 60)

    supervisor = SupervisorAgent()

    # Create worker handlers
    def researcher_handler(task: str) -> str:
        return f"[Researcher] Researched: {task[:40]}..."

    def analyst_handler(task: str) -> str:
        return f"[Analyst] Analyzed: {task[:40]}..."

    def writer_handler(task: str) -> str:
        return f"[Writer] Wrote about: {task[:40]}..."

    def coder_handler(task: str) -> str:
        return f"[Coder] Coded: {task[:40]}..."

    # Register workers
    print("\n--- Registering Workers ---")
    supervisor.register_worker(
        "researcher",
        "Finds information, searches documents, gathers data",
        researcher_handler
    )
    supervisor.register_worker(
        "analyst",
        "Analyzes data, finds patterns, evaluates evidence",
        analyst_handler
    )
    supervisor.register_worker(
        "writer",
        "Writes content, creates documents, drafts reports",
        writer_handler
    )
    supervisor.register_worker(
        "coder",
        "Writes code, fixes bugs, implements features",
        coder_handler
    )

    # Route various tasks
    tasks = [
        "Find information about renewable energy",
        "Write a blog post about AI",
        "Analyze the sales data trends",
        "Implement a sorting algorithm in Python"
    ]

    print("\n--- Routing Tasks ---")
    for task in tasks:
        print(f"\nTask: {task}")
        result = supervisor.route_task(task)
        print(f"Result: {result}")

    print()


# ============================================================
# PATTERN 3: SELF-DELEGATION (RECURSIVE)
# ============================================================

class RecursiveAgent:
    """Agent that can break down tasks and delegate to itself."""

    def __init__(self, max_depth: int = 3):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.max_depth = max_depth

    def process(self, task: str, depth: int = 0) -> str:
        """Process a task, possibly breaking it down recursively."""

        indent = "  " * depth

        if depth >= self.max_depth:
            print(f"{indent}[Depth {depth}] Max depth reached, solving directly")
            return self._direct_solve(task)

        # Decide if task should be broken down
        print(f"{indent}[Depth {depth}] Evaluating: {task[:50]}...")
        subtasks = self._decompose_if_needed(task)

        if len(subtasks) == 1 and subtasks[0] == task:
            # No decomposition needed, solve directly
            print(f"{indent}[Depth {depth}] Solving directly")
            return self._direct_solve(task)
        else:
            # Recursively solve subtasks
            print(f"{indent}[Depth {depth}] Breaking into {len(subtasks)} subtasks")
            results = []
            for i, subtask in enumerate(subtasks):
                print(f"{indent}  Subtask {i+1}: {subtask[:40]}...")
                result = self.process(subtask, depth + 1)
                results.append(result)

            # Combine results
            return self._combine_results(task, results)

    def _decompose_if_needed(self, task: str) -> List[str]:
        """Break task into subtasks if complex."""
        prompt = f"""
        Task: {task}

        Should this task be broken into smaller subtasks?
        If YES, list 2-3 specific subtasks (one per line, starting with "-").
        If NO, respond with exactly: NO_DECOMPOSITION

        Be concise. Only decompose if truly necessary.
        """

        response = self.llm.invoke(prompt)

        if "NO_DECOMPOSITION" in response.content:
            return [task]
        else:
            subtasks = []
            for line in response.content.split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    subtasks.append(line[1:].strip())
                elif line and not any(x in line.lower() for x in ["subtask", "here", "following"]):
                    subtasks.append(line)

            return subtasks if subtasks else [task]

    def _direct_solve(self, task: str) -> str:
        """Directly solve a task without decomposition."""
        prompt = f"Complete this specific task concisely: {task}"
        response = self.llm.invoke(prompt)
        return response.content[:200]  # Truncate for demo

    def _combine_results(self, original_task: str, results: List[str]) -> str:
        """Combine subtask results into final answer."""
        prompt = f"""
        Original task: {original_task}

        Subtask results:
        {chr(10).join(f'- {r[:150]}' for r in results)}

        Combine these into a brief, coherent response (2-3 sentences).
        """

        response = self.llm.invoke(prompt)
        return response.content


def demo_recursive_delegation():
    """Demonstrate self-delegation with task decomposition."""
    print("=" * 60)
    print("DEMO 3: Self-Delegation (Recursive)")
    print("=" * 60)

    agent = RecursiveAgent(max_depth=2)

    tasks = [
        "What is 2 + 2?",  # Simple, no decomposition needed
        "Research the history and current state of electric vehicles",  # Complex, will decompose
    ]

    for task in tasks:
        print(f"\n--- Task: {task} ---\n")
        result = agent.process(task)
        print(f"\nFinal Result: {result[:300]}...")

    print()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-AGENT DELEGATION PATTERNS")
    print("=" * 60 + "\n")

    demo_capability_delegation()
    demo_supervisor_delegation()
    demo_recursive_delegation()

    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. CAPABILITY-BASED DELEGATION
       - Agents know their strengths and limitations
       - Peer-to-peer discovery of who can help
       - Decentralized decision making

    2. SUPERVISOR-BASED DELEGATION
       - Central authority makes routing decisions
       - Easier to debug and understand
       - Single point of control (and failure)

    3. SELF-DELEGATION (RECURSIVE)
       - Agent breaks complex tasks into subtasks
       - Solves subtasks recursively
       - Good for hierarchical problem decomposition

    4. Choose based on your needs:
       - Capability: When agents should self-organize
       - Supervisor: When you need central control
       - Recursive: When tasks are hierarchical
    """)
