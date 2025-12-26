"""
Module 5.3: Complete Multi-Agent Research Team
=============================================
A production-ready multi-agent system demonstrating:
- Specialized agents (planner, researcher, analyst, writer, reviewer)
- Shared memory system
- LangGraph orchestration
- Error handling and monitoring
"""

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass, field
from datetime import datetime
import operator
import uuid
import time

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# STATE DEFINITION
# ============================================================

class ResearchTeamState(TypedDict):
    """Complete state for the research team."""
    # Input
    task_id: str
    query: str

    # Planning
    research_plan: List[str]
    current_step: int

    # Research
    research_findings: Annotated[List[dict], operator.add]
    sources: Annotated[List[str], operator.add]

    # Analysis
    analysis: str
    key_insights: List[str]

    # Writing
    draft: str

    # Review
    review_feedback: str
    final_report: str

    # Control
    current_phase: str
    status: str  # running, completed, failed

    # Metrics
    start_time: float
    phase_times: dict


# ============================================================
# SHARED MEMORY SYSTEM
# ============================================================

class TeamMemory:
    """Shared memory for the research team."""

    def __init__(self, team_id: str = "research_team"):
        self.team_id = team_id
        self.embeddings = OpenAIEmbeddings()

        # Knowledge store for past research
        self.knowledge_store = Chroma(
            collection_name=f"{team_id}_knowledge",
            embedding_function=self.embeddings
        )

        # Working memory for current task
        self.working: dict = {}

    def store_knowledge(self, content: str, source: str, metadata: dict = None):
        """Store research knowledge."""
        self.knowledge_store.add_texts(
            texts=[content],
            metadatas=[{
                "source": source,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }]
        )

    def retrieve_relevant(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant past knowledge."""
        try:
            docs = self.knowledge_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except:
            return []

    def set_working(self, key: str, value):
        """Set working memory."""
        self.working[key] = value

    def get_working(self, key: str, default=None):
        """Get from working memory."""
        return self.working.get(key, default)


# ============================================================
# MONITORING SYSTEM
# ============================================================

@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls > 0 else 0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.calls if self.calls > 0 else 0


class TeamMonitor:
    """Monitor for the research team."""

    def __init__(self):
        self.agent_metrics: dict[str, AgentMetrics] = {}
        self.task_count = 0
        self.completed_tasks = 0

    def record_call(self, agent_name: str, success: bool, duration: float):
        """Record an agent call."""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(name=agent_name)

        metrics = self.agent_metrics[agent_name]
        metrics.calls += 1
        metrics.total_time += duration

        if success:
            metrics.successes += 1
        else:
            metrics.failures += 1

    def get_report(self) -> str:
        """Generate a metrics report."""
        lines = ["=== Team Performance Report ==="]

        for name, m in self.agent_metrics.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Calls: {m.calls}")
            lines.append(f"  Success Rate: {m.success_rate:.1%}")
            lines.append(f"  Avg Time: {m.avg_time:.2f}s")

        return "\n".join(lines)


# ============================================================
# AGENT DEFINITIONS
# ============================================================

class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, llm, memory: TeamMemory, monitor: TeamMonitor):
        self.name = name
        self.llm = llm
        self.memory = memory
        self.monitor = monitor

    def invoke(self, prompt: str) -> str:
        """Invoke LLM with monitoring."""
        start = time.time()
        success = True

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            success = False
            raise
        finally:
            self.monitor.record_call(self.name, success, time.time() - start)


class PlannerAgent(BaseAgent):
    """Plans the research approach."""

    def plan(self, state: ResearchTeamState) -> dict:
        """Create a research plan."""
        print(f"\n[Planner] Creating research plan...")

        # Check for relevant past research
        past = self.memory.retrieve_relevant(state["query"], k=2)
        past_context = f"Relevant past research:\n{chr(10).join(past)}" if past else ""

        prompt = f"""
        You are a research planner. Create a research plan.

        Query: {state['query']}

        {past_context}

        Create 3-4 specific research questions to investigate.
        Format as numbered list (1., 2., etc.)
        """

        response = self.invoke(prompt)

        # Parse steps
        steps = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                steps.append(line.lstrip("0123456789.)- "))

        if not steps:
            steps = [state["query"]]

        print(f"[Planner] Created {len(steps)} research steps")

        return {
            "research_plan": steps,
            "current_step": 0,
            "current_phase": "research",
            "phase_times": {
                **state.get("phase_times", {}),
                "planning": time.time() - state["start_time"]
            }
        }


class ResearcherAgent(BaseAgent):
    """Conducts research."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.search = DuckDuckGoSearchRun()
        except:
            self.search = None

    def research(self, state: ResearchTeamState) -> dict:
        """Conduct research on current step."""
        current = state["current_step"]

        if current >= len(state["research_plan"]):
            return {"current_phase": "analysis"}

        question = state["research_plan"][current]
        print(f"\n[Researcher] Step {current + 1}: {question[:50]}...")

        # Search for information
        search_results = ""
        if self.search:
            try:
                search_results = self.search.run(question)[:1500]
            except:
                search_results = "Search unavailable."

        prompt = f"""
        Research question: {question}

        Search results:
        {search_results if search_results else 'No search results available.'}

        Provide:
        1. Key findings (3-4 bullet points)
        2. Confidence level (high/medium/low)
        """

        response = self.invoke(prompt)

        # Store in team memory
        self.memory.store_knowledge(response, source="research", metadata={"question": question})

        print(f"[Researcher] Completed step {current + 1}")

        return {
            "research_findings": [{
                "question": question,
                "findings": response,
                "step": current
            }],
            "sources": [f"Research: {question}"],
            "current_step": current + 1
        }


class AnalystAgent(BaseAgent):
    """Analyzes research findings."""

    def analyze(self, state: ResearchTeamState) -> dict:
        """Analyze all findings."""
        print(f"\n[Analyst] Analyzing {len(state['research_findings'])} findings...")

        findings_text = "\n\n".join(
            f"Q: {f['question']}\n{f['findings']}"
            for f in state["research_findings"]
        )

        prompt = f"""
        Original query: {state['query']}

        Research findings:
        {findings_text}

        Provide:
        1. Executive Summary (2-3 sentences)
        2. Key Insights (numbered list, 3-5 items)
        3. Conclusions
        """

        response = self.invoke(prompt)

        # Extract key insights
        insights = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                insights.append(line.lstrip("0123456789.)- "))

        print(f"[Analyst] Identified {len(insights)} key insights")

        return {
            "analysis": response,
            "key_insights": insights[:5],
            "current_phase": "writing",
            "phase_times": {
                **state.get("phase_times", {}),
                "analysis": time.time() - state["start_time"]
            }
        }


class WriterAgent(BaseAgent):
    """Writes the research report."""

    def write(self, state: ResearchTeamState) -> dict:
        """Write the draft report."""
        print(f"\n[Writer] Drafting report...")

        prompt = f"""
        Write a research report.

        Query: {state['query']}

        Analysis:
        {state['analysis']}

        Create a well-structured report with:
        - Title
        - Executive Summary
        - Key Findings
        - Conclusions
        - Recommendations

        Keep it concise but comprehensive.
        """

        response = self.invoke(prompt)

        print(f"[Writer] Draft completed ({len(response)} chars)")

        return {
            "draft": response,
            "current_phase": "review",
            "phase_times": {
                **state.get("phase_times", {}),
                "writing": time.time() - state["start_time"]
            }
        }


class ReviewerAgent(BaseAgent):
    """Reviews and improves the report."""

    def review(self, state: ResearchTeamState) -> dict:
        """Review and finalize the report."""
        print(f"\n[Reviewer] Reviewing draft...")

        prompt = f"""
        Review and improve this research report.

        Original Query: {state['query']}

        Draft:
        {state['draft']}

        Provide:
        1. Brief feedback (1-2 sentences)
        2. The improved final report

        Separate with "---FINAL---"
        """

        response = self.invoke(prompt)

        # Split feedback and final
        if "---FINAL---" in response:
            parts = response.split("---FINAL---")
            feedback = parts[0].strip()
            final = parts[1].strip()
        else:
            feedback = "Approved with minor edits."
            final = response

        print(f"[Reviewer] Review complete")

        return {
            "review_feedback": feedback,
            "final_report": final,
            "current_phase": "complete",
            "status": "completed",
            "phase_times": {
                **state.get("phase_times", {}),
                "review": time.time() - state["start_time"]
            }
        }


# ============================================================
# TEAM ORCHESTRATION
# ============================================================

class ResearchTeam:
    """Complete research team with orchestration."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory = TeamMemory()
        self.monitor = TeamMonitor()

        # Create agents
        self.planner = PlannerAgent("Planner", self.llm, self.memory, self.monitor)
        self.researcher = ResearcherAgent("Researcher", self.llm, self.memory, self.monitor)
        self.analyst = AnalystAgent("Analyst", self.llm, self.memory, self.monitor)
        self.writer = WriterAgent("Writer", self.llm, self.memory, self.monitor)
        self.reviewer = ReviewerAgent("Reviewer", self.llm, self.memory, self.monitor)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        graph = StateGraph(ResearchTeamState)

        # Add nodes
        graph.add_node("plan", self.planner.plan)
        graph.add_node("research", self.researcher.research)
        graph.add_node("analyze", self.analyst.analyze)
        graph.add_node("write", self.writer.write)
        graph.add_node("review", self.reviewer.review)

        # Add edges
        graph.set_entry_point("plan")
        graph.add_edge("plan", "research")

        # Research continues until all steps done
        graph.add_conditional_edges(
            "research",
            self._should_continue_research,
            {"research": "research", "analyze": "analyze"}
        )

        graph.add_edge("analyze", "write")
        graph.add_edge("write", "review")
        graph.add_edge("review", END)

        return graph.compile()

    def _should_continue_research(self, state: ResearchTeamState) -> str:
        """Check if more research steps remain."""
        if state["current_step"] < len(state["research_plan"]):
            return "research"
        return "analyze"

    def run(self, query: str) -> dict:
        """Run the research team."""
        print("\n" + "=" * 60)
        print("RESEARCH TEAM STARTING")
        print("=" * 60)
        print(f"\nQuery: {query}")

        initial_state = {
            "task_id": str(uuid.uuid4()),
            "query": query,
            "research_plan": [],
            "current_step": 0,
            "research_findings": [],
            "sources": [],
            "analysis": "",
            "key_insights": [],
            "draft": "",
            "review_feedback": "",
            "final_report": "",
            "current_phase": "planning",
            "status": "running",
            "start_time": time.time(),
            "phase_times": {}
        }

        result = self.graph.invoke(initial_state)

        total_time = time.time() - result["start_time"]

        print("\n" + "=" * 60)
        print("RESEARCH COMPLETE")
        print("=" * 60)

        return {
            "task_id": result["task_id"],
            "query": result["query"],
            "final_report": result["final_report"],
            "key_insights": result["key_insights"],
            "sources": result["sources"],
            "status": result["status"],
            "total_time": total_time,
            "phase_times": result["phase_times"],
            "metrics": self.monitor.get_report()
        }


# ============================================================
# DEMO: SIMPLER TEAM FOR QUICK TESTING
# ============================================================

def demo_simple_team():
    """Quick demo with simplified agents."""
    print("=" * 60)
    print("DEMO: Simple Research Team")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Simple agent functions
    def planner(query: str) -> List[str]:
        response = llm.invoke(f"Create 2 research questions for: {query}")
        return [line.strip() for line in response.content.split("\n") if line.strip()][:2]

    def researcher(question: str) -> str:
        response = llm.invoke(f"Research briefly: {question}")
        return response.content

    def analyst(findings: List[str]) -> str:
        response = llm.invoke(f"Analyze these findings:\n{chr(10).join(findings)}")
        return response.content

    def writer(analysis: str, query: str) -> str:
        response = llm.invoke(f"Write a brief report on '{query}' based on: {analysis}")
        return response.content

    # Run the pipeline
    query = "What are the benefits of meditation?"

    print(f"\nQuery: {query}")
    print("\n--- Planning ---")
    plan = planner(query)
    for i, q in enumerate(plan, 1):
        print(f"  {i}. {q[:60]}...")

    print("\n--- Researching ---")
    findings = []
    for q in plan:
        result = researcher(q)
        findings.append(result)
        print(f"  Found: {result[:60]}...")

    print("\n--- Analyzing ---")
    analysis = analyst(findings)
    print(f"  Analysis: {analysis[:100]}...")

    print("\n--- Writing ---")
    report = writer(analysis, query)
    print(f"\n{report[:500]}...")

    print()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-AGENT RESEARCH TEAM")
    print("=" * 60 + "\n")

    # Quick demo
    demo_simple_team()

    # Full team demo (uncomment to run - takes longer)
    print("\n" + "=" * 60)
    print("FULL TEAM DEMO")
    print("=" * 60)

    team = ResearchTeam()
    query = "What are the environmental and economic impacts of electric vehicles?"

    result = team.run(query)

    print(f"\nTask ID: {result['task_id']}")
    print(f"Status: {result['status']}")
    print(f"Total Time: {result['total_time']:.2f}s")

    print("\n--- Key Insights ---")
    for insight in result['key_insights'][:5]:
        print(f"  • {insight}")

    print("\n--- Final Report Preview ---")
    print(result['final_report'][:800])
    if len(result['final_report']) > 800:
        print("\n... [truncated]")

    print("\n--- Team Performance ---")
    print(result['metrics'])

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. SPECIALIZED AGENTS excel at their specific tasks
       - Planner: Creates structured research plans
       - Researcher: Gathers information
       - Analyst: Synthesizes and extracts insights
       - Writer: Creates polished output
       - Reviewer: Ensures quality

    2. SHARED MEMORY enables coordination
       - Working memory for current task
       - Long-term memory for past research
       - All agents can access and contribute

    3. LANGGRAPH ORCHESTRATION handles the workflow
       - Defines agent order and dependencies
       - Manages state transitions
       - Supports conditional routing

    4. MONITORING is essential for production
       - Track success/failure rates
       - Measure performance
       - Identify bottlenecks

    5. DESIGN PRINCIPLES:
       - Single responsibility per agent
       - Clear interfaces between agents
       - Graceful error handling
       - Observable and debuggable
    """)
