"""
Module 5.1: Multi-Agent Architectural Patterns
==============================================
Demonstrates major architectural patterns:
- Supervisor Architecture
- Peer-to-Peer Architecture
- Assembly Line Architecture
"""

from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import operator

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# ARCHITECTURE 1: SUPERVISOR PATTERN
# ============================================================

class SupervisorState(TypedDict):
    """State for supervisor-based system."""
    query: str
    plan: List[str]
    current_step: int
    worker_outputs: Annotated[List[dict], operator.add]
    final_response: str


class SupervisorArchitecture:
    """
    Supervisor architecture where a central agent coordinates workers.

    Structure:
                    ┌─────────────┐
                    │ SUPERVISOR  │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       ┌────────┐    ┌────────┐    ┌────────┐
       │Worker A│    │Worker B│    │Worker C│
       └────────┘    └────────┘    └────────┘
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.workers = {
            "researcher": "Find and gather information",
            "analyst": "Analyze data and find patterns",
            "writer": "Create written content"
        }

    def supervisor_plan(self, state: SupervisorState) -> dict:
        """Supervisor creates a plan."""
        print("\n[Supervisor] Creating plan...")

        worker_list = "\n".join(f"- {k}: {v}" for k, v in self.workers.items())

        prompt = ChatPromptTemplate.from_template("""
        Create a plan to answer this query using available workers.

        Workers:
        {workers}

        Query: {query}

        Create a plan with 2-3 steps. Format each step as:
        WORKER_NAME: specific task

        Example:
        researcher: Find information about X
        analyst: Analyze the findings
        """)

        response = self.llm.invoke(
            prompt.format(workers=worker_list, query=state["query"])
        )

        # Parse plan
        plan = []
        for line in response.content.split("\n"):
            line = line.strip()
            if ":" in line:
                plan.append(line)

        print(f"[Supervisor] Plan created with {len(plan)} steps")
        for i, step in enumerate(plan):
            print(f"  Step {i+1}: {step}")

        return {"plan": plan, "current_step": 0}

    def execute_worker(self, worker_name: str, task: str, context: str) -> str:
        """Execute a single worker."""
        print(f"\n[{worker_name.title()}] Working on: {task[:50]}...")

        prompt = f"""
        You are a {worker_name}. {self.workers.get(worker_name, '')}

        Task: {task}

        Context from previous steps:
        {context if context else 'None yet.'}

        Provide a concise response (2-3 sentences).
        """

        response = self.llm.invoke(prompt)
        return response.content

    def execute_step(self, state: SupervisorState) -> dict:
        """Execute the current step."""
        current = state["current_step"]

        if current >= len(state["plan"]):
            return state

        step = state["plan"][current]

        # Parse worker and task
        parts = step.split(":", 1)
        worker_name = parts[0].strip().lower()
        task = parts[1].strip() if len(parts) > 1 else step

        # Get context from previous outputs
        context = "\n".join(
            f"- {o['worker']}: {o['output'][:200]}"
            for o in state.get("worker_outputs", [])
        )

        # Execute
        output = self.execute_worker(worker_name, task, context)
        print(f"[{worker_name.title()}] Output: {output[:100]}...")

        return {
            "worker_outputs": [{"worker": worker_name, "output": output}],
            "current_step": current + 1
        }

    def should_continue(self, state: SupervisorState) -> str:
        """Check if more steps to execute."""
        if state["current_step"] < len(state["plan"]):
            return "execute"
        return "compile"

    def compile_response(self, state: SupervisorState) -> dict:
        """Compile final response."""
        print("\n[Supervisor] Compiling final response...")

        outputs = "\n".join(
            f"- {o['worker']}: {o['output']}"
            for o in state["worker_outputs"]
        )

        prompt = f"""
        Compile these worker outputs into a final response.

        Query: {state['query']}

        Worker outputs:
        {outputs}

        Provide a coherent final answer (3-4 sentences).
        """

        response = self.llm.invoke(prompt)
        return {"final_response": response.content}

    def build_graph(self):
        """Build the supervisor graph."""
        graph = StateGraph(SupervisorState)

        graph.add_node("plan", self.supervisor_plan)
        graph.add_node("execute", self.execute_step)
        graph.add_node("compile", self.compile_response)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute",
            self.should_continue,
            {"execute": "execute", "compile": "compile"}
        )
        graph.add_edge("compile", END)

        return graph.compile()

    def run(self, query: str) -> str:
        """Run the supervisor system."""
        graph = self.build_graph()
        result = graph.invoke({
            "query": query,
            "plan": [],
            "current_step": 0,
            "worker_outputs": [],
            "final_response": ""
        })
        return result["final_response"]


def demo_supervisor():
    """Demonstrate supervisor architecture."""
    print("=" * 60)
    print("DEMO 1: Supervisor Architecture")
    print("=" * 60)

    system = SupervisorArchitecture()
    query = "What are the main benefits and challenges of remote work?"

    print(f"\nQuery: {query}")
    result = system.run(query)

    print("\n" + "-" * 40)
    print("FINAL RESPONSE:")
    print("-" * 40)
    print(result)
    print()


# ============================================================
# ARCHITECTURE 2: PEER-TO-PEER PATTERN
# ============================================================

class PeerToPeerState(TypedDict):
    """State for peer-to-peer system."""
    task: str
    contributions: Annotated[List[dict], operator.add]
    round_number: int
    max_rounds: int
    final_output: str


class PeerToPeerArchitecture:
    """
    Peer-to-peer architecture where agents collaborate without central control.

    Structure:
         ┌─────────┐     ┌─────────┐
         │ Agent A │◄───►│ Agent B │
         └────┬────┘     └────┬────┘
              │               │
              └───────┬───────┘
                      ▼
              ┌─────────────┐
              │  Agent C    │
              └─────────────┘
    """

    def __init__(self, max_rounds: int = 2):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.max_rounds = max_rounds
        self.agents = {
            "explorer": "Explores new ideas and possibilities",
            "critic": "Critiques and finds weaknesses",
            "synthesizer": "Combines and reconciles different views"
        }

    def agent_contribute(self, agent_name: str, state: PeerToPeerState) -> dict:
        """Single agent makes a contribution."""

        # Get other agents' contributions
        others = [c for c in state["contributions"] if c["agent"] != agent_name]
        others_text = "\n".join(
            f"- {c['agent']}: {c['content'][:200]}"
            for c in others
        ) if others else "None yet."

        prompt = f"""
        You are the {agent_name}. {self.agents[agent_name]}

        Task: {state['task']}
        Round: {state['round_number'] + 1} of {state['max_rounds']}

        Other agents' contributions:
        {others_text}

        Provide your unique contribution (2-3 sentences).
        Build on others' work if relevant.
        """

        response = self.llm.invoke(prompt)

        return {
            "agent": agent_name,
            "content": response.content,
            "round": state["round_number"]
        }

    def contribution_round(self, state: PeerToPeerState) -> dict:
        """All agents contribute in a round."""
        print(f"\n[Round {state['round_number'] + 1}]")

        new_contributions = []
        for agent_name in self.agents.keys():
            print(f"  [{agent_name.title()}] Contributing...")
            contribution = self.agent_contribute(agent_name, state)
            new_contributions.append(contribution)
            print(f"  [{agent_name.title()}]: {contribution['content'][:80]}...")

        return {
            "contributions": new_contributions,
            "round_number": state["round_number"] + 1
        }

    def should_continue(self, state: PeerToPeerState) -> str:
        """Check if more rounds needed."""
        if state["round_number"] < state["max_rounds"]:
            return "contribute"
        return "synthesize"

    def final_synthesis(self, state: PeerToPeerState) -> dict:
        """Create final synthesis from all contributions."""
        print("\n[Synthesis] Combining all contributions...")

        all_contributions = "\n".join(
            f"- {c['agent']} (round {c['round'] + 1}): {c['content']}"
            for c in state["contributions"]
        )

        prompt = f"""
        Synthesize all agent contributions into a final answer.

        Task: {state['task']}

        All contributions:
        {all_contributions}

        Create a balanced final response (3-4 sentences).
        """

        response = self.llm.invoke(prompt)
        return {"final_output": response.content}

    def build_graph(self):
        """Build the peer-to-peer graph."""
        graph = StateGraph(PeerToPeerState)

        graph.add_node("contribute", self.contribution_round)
        graph.add_node("synthesize", self.final_synthesis)

        graph.set_entry_point("contribute")
        graph.add_conditional_edges(
            "contribute",
            self.should_continue,
            {"contribute": "contribute", "synthesize": "synthesize"}
        )
        graph.add_edge("synthesize", END)

        return graph.compile()

    def run(self, task: str) -> str:
        """Run the peer-to-peer system."""
        graph = self.build_graph()
        result = graph.invoke({
            "task": task,
            "contributions": [],
            "round_number": 0,
            "max_rounds": self.max_rounds,
            "final_output": ""
        })
        return result["final_output"]


def demo_peer_to_peer():
    """Demonstrate peer-to-peer architecture."""
    print("=" * 60)
    print("DEMO 2: Peer-to-Peer Architecture")
    print("=" * 60)

    system = PeerToPeerArchitecture(max_rounds=2)
    task = "Should AI systems be given more autonomy in critical decisions?"

    print(f"\nTask: {task}")
    result = system.run(task)

    print("\n" + "-" * 40)
    print("FINAL SYNTHESIS:")
    print("-" * 40)
    print(result)
    print()


# ============================================================
# ARCHITECTURE 3: ASSEMBLY LINE PATTERN
# ============================================================

class AssemblyLineState(TypedDict):
    """State for assembly line."""
    input_data: str
    stage_outputs: Annotated[List[dict], operator.add]
    current_stage: str
    final_product: str


class AssemblyLineArchitecture:
    """
    Assembly line architecture with sequential processing.

    Structure:
    Input → [Stage 1] → [Stage 2] → [Stage 3] → Output
             Extract      Process     Polish
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.stages = [
            ("extract", "Extract key information and facts"),
            ("process", "Analyze and organize the information"),
            ("polish", "Create a polished final output")
        ]

    def run_stage(self, stage_name: str, stage_desc: str,
                  state: AssemblyLineState) -> dict:
        """Run a single stage."""
        print(f"\n[Stage: {stage_name.upper()}] {stage_desc}")

        # Get previous stage output
        previous = ""
        if state["stage_outputs"]:
            previous = state["stage_outputs"][-1]["output"]

        prompt = f"""
        Stage: {stage_name} - {stage_desc}

        Input: {state['input_data']}

        Previous stage output:
        {previous if previous else 'This is the first stage.'}

        Perform this stage's task. Be concise (2-3 sentences).
        """

        response = self.llm.invoke(prompt)
        output = response.content

        print(f"  Output: {output[:100]}...")

        return {
            "stage_outputs": [{"stage": stage_name, "output": output}],
            "current_stage": stage_name
        }

    def build_graph(self):
        """Build the assembly line graph."""
        graph = StateGraph(AssemblyLineState)

        # Add stage nodes
        for stage_name, stage_desc in self.stages:
            graph.add_node(
                stage_name,
                lambda s, name=stage_name, desc=stage_desc: self.run_stage(name, desc, s)
            )

        # Add final compilation node
        def compile_final(state: AssemblyLineState) -> dict:
            final = state["stage_outputs"][-1]["output"] if state["stage_outputs"] else ""
            return {"final_product": final}

        graph.add_node("compile", compile_final)

        # Connect stages
        graph.set_entry_point(self.stages[0][0])
        for i in range(len(self.stages) - 1):
            graph.add_edge(self.stages[i][0], self.stages[i + 1][0])
        graph.add_edge(self.stages[-1][0], "compile")
        graph.add_edge("compile", END)

        return graph.compile()

    def run(self, input_data: str) -> str:
        """Run the assembly line."""
        graph = self.build_graph()
        result = graph.invoke({
            "input_data": input_data,
            "stage_outputs": [],
            "current_stage": "",
            "final_product": ""
        })
        return result["final_product"]


def demo_assembly_line():
    """Demonstrate assembly line architecture."""
    print("=" * 60)
    print("DEMO 3: Assembly Line Architecture")
    print("=" * 60)

    system = AssemblyLineArchitecture()
    input_data = "The rapid growth of renewable energy sources, particularly solar and wind power, is transforming global electricity generation. By 2023, renewable energy accounted for over 30% of global electricity production, up from just 20% in 2010."

    print(f"\nInput: {input_data[:80]}...")
    result = system.run(input_data)

    print("\n" + "-" * 40)
    print("FINAL PRODUCT:")
    print("-" * 40)
    print(result)
    print()


# ============================================================
# ARCHITECTURE COMPARISON
# ============================================================

def demo_comparison():
    """Compare the architectures."""
    print("=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)

    comparison = """
    SUPERVISOR ARCHITECTURE
    ───────────────────────
    Structure: Central coordinator + workers
    Control: Centralized
    Best for: Clear authority, predictable workflows
    Pros: Easy to debug, clear responsibility
    Cons: Supervisor is bottleneck, single point of failure

    PEER-TO-PEER ARCHITECTURE
    ─────────────────────────
    Structure: Equal agents communicating directly
    Control: Distributed
    Best for: Creative tasks, diverse perspectives
    Pros: No bottleneck, resilient
    Cons: Harder to debug, may need conflict resolution

    ASSEMBLY LINE ARCHITECTURE
    ──────────────────────────
    Structure: Sequential stages
    Control: Flow-based
    Best for: Predictable transformations, document processing
    Pros: Simple, predictable, easy to test each stage
    Cons: Inflexible, slow (sequential)

    CHOOSING AN ARCHITECTURE
    ────────────────────────
    1. Need central control? → Supervisor
    2. Need multiple perspectives? → Peer-to-Peer
    3. Have clear stages? → Assembly Line
    4. Complex routing? → Supervisor with dynamic routing
    5. Need resilience? → Peer-to-Peer
    """

    print(comparison)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-AGENT ARCHITECTURAL PATTERNS")
    print("=" * 60 + "\n")

    demo_supervisor()
    demo_peer_to_peer()
    demo_assembly_line()
    demo_comparison()

    print("=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. SUPERVISOR is best when you need control and predictability
       - Central decision-making
       - Clear chain of command
       - Easy debugging

    2. PEER-TO-PEER is best for creative and resilient systems
       - Agents self-organize
       - Multiple perspectives
       - No single point of failure

    3. ASSEMBLY LINE is best for sequential transformations
       - Clear stages
       - Predictable flow
       - Easy to test

    4. Architectures can be combined:
       - Supervisor managing assembly lines
       - Peer-to-peer within assembly line stages
       - Hierarchical supervisors

    5. Choose based on your specific requirements:
       - Control needs
       - Resilience requirements
       - Problem structure
    """)
