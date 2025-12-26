# Module 13.1: Autonomous Agent Foundations

> "The question is not whether intelligent machines can have any emotions, but whether machines can be intelligent without any emotions." — Marvin Minsky

## Introduction

What does it mean for an AI agent to be truly autonomous? In this module, we'll apply first principles thinking to deconstruct autonomy, understand its essential components, and build agents that can operate independently while remaining aligned with our intentions.

---

## Learning Objectives

By the end of this module, you will:
- Define autonomy from first principles (not just what it looks like, but what it *is*)
- Understand the spectrum from automation to true autonomy
- Implement goal hierarchies and sub-goal generation
- Build agents that make decisions without explicit instructions
- Create systems that maintain alignment while operating independently

---

## First Principles: Deconstructing Autonomy

### What Autonomy Is NOT

Let's clear away misconceptions:

| Misconception | Reality |
|--------------|---------|
| "Autonomy means no rules" | True autonomy operates *within* principles |
| "Autonomous = unpredictable" | Good autonomous systems are predictable in *values*, flexible in *methods* |
| "More autonomy is always better" | Right-sized autonomy matches the task |
| "Autonomy removes human control" | It shifts control from actions to goals |

### The Fundamental Components of Autonomy

From first principles, what must an autonomous system have?

```
Autonomy = Goals + Perception + Decision-Making + Action + Feedback

Where:
├── Goals: What the agent is trying to achieve
├── Perception: Understanding the current state
├── Decision-Making: Choosing actions without explicit instruction
├── Action: Capability to affect the environment
└── Feedback: Evaluating outcomes against goals
```

**Key Insight:** Remove any of these, and you don't have autonomy:
- Without goals → random behavior
- Without perception → blind action
- Without decision-making → automation
- Without action → passive observation
- Without feedback → no improvement

---

## The Autonomy Spectrum

### Level 0: Scripted Automation

```python
# Level 0: Pure automation - no decisions
def automated_report():
    """Does exactly the same thing every time."""
    data = fetch_data()
    report = format_report(data)
    send_email(report)
    return "Report sent"

# No autonomy: every step is predetermined
```

**Characteristics:**
- Fixed sequence of actions
- No adaptation to context
- Fails if environment changes
- Predictable but brittle

### Level 1: Reactive Agency

```python
# Level 1: Reacts to environment
def reactive_agent(query: str):
    """Chooses actions based on input, but strategies are fixed."""
    if "weather" in query.lower():
        return weather_tool(query)
    elif "calculate" in query.lower():
        return calculator_tool(query)
    else:
        return search_tool(query)

# Some autonomy: chooses which tool, but logic is hardcoded
```

**Characteristics:**
- Responds to input variability
- Fixed decision logic
- Can handle multiple paths
- Still fundamentally predetermined

### Level 2: Adaptive Agency

```python
# Level 2: Learns and adapts
class AdaptiveAgent:
    def __init__(self):
        self.strategy_weights = {"direct": 0.5, "search_first": 0.5}
        self.history = []

    def execute(self, query: str) -> str:
        # Choose strategy based on learned weights
        strategy = self.select_strategy(query)
        result = self.run_strategy(strategy, query)

        # Update weights based on outcome
        self.update_from_feedback(strategy, result)
        return result

    def select_strategy(self, query: str) -> str:
        """Probabilistic selection based on learned effectiveness."""
        # Not just if-else, but learned preferences
        return weighted_random_choice(self.strategy_weights)

    def update_from_feedback(self, strategy: str, result: dict):
        """Adjust strategy weights based on success."""
        if result["success"]:
            self.strategy_weights[strategy] *= 1.1
        else:
            self.strategy_weights[strategy] *= 0.9
        self.normalize_weights()
```

**Characteristics:**
- Learns from experience
- Adjusts strategies over time
- Handles novelty better
- Still operates within defined strategy space

### Level 3: Goal-Directed Autonomy

```python
# Level 3: Sets sub-goals, operates toward objectives
class AutonomousAgent:
    def __init__(self, llm, tools, goal: str):
        self.llm = llm
        self.tools = tools
        self.primary_goal = goal
        self.sub_goals = []
        self.context = AgentContext()

    async def run_autonomously(self, time_budget: int = 3600):
        """Operates independently toward goal."""
        start_time = time.time()

        while not self.goal_achieved() and self.within_budget(start_time, time_budget):
            # Perceive current state
            current_state = await self.perceive()

            # Generate or select sub-goal
            sub_goal = await self.generate_subgoal(current_state)

            # Plan and execute
            plan = await self.create_plan(sub_goal)
            result = await self.execute_plan(plan)

            # Evaluate and adapt
            await self.evaluate_and_learn(sub_goal, result)

            # Check if should escalate
            if self.should_escalate(result):
                await self.request_human_input()

        return self.compile_results()

    async def generate_subgoal(self, state: dict) -> str:
        """LLM generates appropriate sub-goal given current state."""
        prompt = f"""
        Primary Goal: {self.primary_goal}
        Current State: {state}
        Completed Sub-goals: {self.completed_subgoals}

        What is the most important next sub-goal to pursue?
        Consider: dependencies, available resources, progress so far.

        Return a specific, actionable sub-goal.
        """
        return await self.llm.generate(prompt)
```

**Characteristics:**
- Generates own sub-goals
- Operates for extended periods
- Adapts strategy in real-time
- Knows when to ask for help

---

## Building Goal Hierarchies

### First Principles: What Is a Goal?

A goal is a desired state of the world that guides action selection.

```
Goal Properties:
├── Measurable: Can determine if achieved
├── Decomposable: Can be broken into sub-goals
├── Prioritizable: Some goals matter more
└── Contextual: Interpretation depends on situation
```

### Goal Hierarchy Architecture

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"

@dataclass
class Goal:
    description: str
    success_criteria: str
    priority: int = 5  # 1-10 scale
    status: GoalStatus = GoalStatus.PENDING
    parent: Optional['Goal'] = None
    sub_goals: List['Goal'] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3

    def is_leaf(self) -> bool:
        """Leaf goals have no sub-goals and are directly actionable."""
        return len(self.sub_goals) == 0

    def completion_ratio(self) -> float:
        """How much of this goal (including sub-goals) is complete."""
        if self.is_leaf():
            return 1.0 if self.status == GoalStatus.COMPLETED else 0.0

        if not self.sub_goals:
            return 0.0

        return sum(g.completion_ratio() for g in self.sub_goals) / len(self.sub_goals)

    def get_next_actionable(self) -> Optional['Goal']:
        """Find the next leaf goal to work on."""
        if self.status == GoalStatus.COMPLETED:
            return None

        if self.is_leaf() and self.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS]:
            return self

        # Find first incomplete sub-goal
        for sub_goal in sorted(self.sub_goals, key=lambda g: -g.priority):
            next_goal = sub_goal.get_next_actionable()
            if next_goal:
                return next_goal

        return None


class GoalManager:
    """Manages goal hierarchy and sub-goal generation."""

    def __init__(self, llm):
        self.llm = llm
        self.root_goals: List[Goal] = []

    async def decompose_goal(self, goal: Goal, context: dict) -> List[Goal]:
        """Use LLM to break a goal into sub-goals."""
        prompt = f"""
        Goal to decompose: {goal.description}
        Success criteria: {goal.success_criteria}
        Current context: {context}

        Break this goal into 2-5 specific, actionable sub-goals.
        Each sub-goal should:
        1. Be concrete and measurable
        2. Be achievable with available tools
        3. Contribute clearly to the parent goal

        Format each sub-goal as:
        - Description: [what to do]
        - Success Criteria: [how to know it's done]
        - Priority: [1-10]
        """

        response = await self.llm.generate(prompt)
        sub_goals = self.parse_subgoals(response, parent=goal)
        goal.sub_goals = sub_goals
        return sub_goals

    async def should_decompose(self, goal: Goal, context: dict) -> bool:
        """Determine if a goal needs to be broken down further."""
        prompt = f"""
        Goal: {goal.description}
        Context: {context}

        Can this goal be accomplished in a single action or tool call?
        Answer: YES (single action) or NO (needs decomposition)
        """
        response = await self.llm.generate(prompt)
        return "NO" in response.upper()
```

### Dynamic Sub-Goal Generation

The power of autonomous agents is generating appropriate sub-goals at runtime:

```python
class AutonomousGoalAgent:
    """Agent that generates and pursues sub-goals autonomously."""

    def __init__(self, llm, tools: List[Tool], guardrails: Guardrails):
        self.llm = llm
        self.tools = tools
        self.guardrails = guardrails
        self.goal_manager = GoalManager(llm)
        self.execution_history = []

    async def pursue_goal(self, primary_goal: str, time_limit: int = 3600) -> dict:
        """Autonomously work toward a goal."""

        # Create root goal
        root = Goal(
            description=primary_goal,
            success_criteria=await self.infer_success_criteria(primary_goal),
            priority=10
        )
        self.goal_manager.root_goals.append(root)

        start_time = time.time()

        while time.time() - start_time < time_limit:
            # Get next actionable goal
            current_goal = root.get_next_actionable()

            if not current_goal:
                if root.status == GoalStatus.COMPLETED:
                    return {"status": "success", "message": "Goal achieved"}
                else:
                    return {"status": "blocked", "message": "No actionable goals"}

            # Check if goal needs decomposition
            context = self.get_current_context()
            if await self.goal_manager.should_decompose(current_goal, context):
                await self.goal_manager.decompose_goal(current_goal, context)
                continue

            # Execute the leaf goal
            current_goal.status = GoalStatus.IN_PROGRESS
            result = await self.execute_goal(current_goal)

            # Update status based on result
            if result["success"]:
                current_goal.status = GoalStatus.COMPLETED
                self.propagate_completion(current_goal)
            else:
                current_goal.attempts += 1
                if current_goal.attempts >= current_goal.max_attempts:
                    current_goal.status = GoalStatus.BLOCKED
                    await self.handle_blocked_goal(current_goal)

            # Learn from this execution
            self.execution_history.append({
                "goal": current_goal.description,
                "result": result,
                "context": context
            })

        return {
            "status": "timeout",
            "progress": root.completion_ratio(),
            "message": f"Time limit reached. {root.completion_ratio()*100:.1f}% complete"
        }

    async def execute_goal(self, goal: Goal) -> dict:
        """Execute a leaf goal using available tools."""

        # First, check guardrails
        if not await self.guardrails.allow_goal(goal):
            return {"success": False, "reason": "Blocked by guardrails"}

        # Generate execution plan
        prompt = f"""
        Goal: {goal.description}
        Success Criteria: {goal.success_criteria}
        Available Tools: {[t.name for t in self.tools]}

        What is the best single action to accomplish this goal?
        """

        action = await self.llm.generate(prompt)

        # Execute the action
        try:
            result = await self.execute_action(action)

            # Check if success criteria met
            success = await self.check_success(goal, result)
            return {"success": success, "result": result}

        except Exception as e:
            return {"success": False, "error": str(e)}
```

---

## Decision-Making Without Explicit Instructions

### The Core Challenge

Traditional programs have explicit instructions for every situation. Autonomous agents must decide what to do when they encounter situations not explicitly covered.

### First Principles: How Do Humans Decide?

```
Human Decision-Making:
├── Values: What matters to us
├── Goals: What we're trying to achieve
├── Context: Current situation
├── Options: Available actions
├── Consequences: Predicted outcomes
└── Selection: Choose best option given values/goals
```

We can implement this same structure:

```python
class AutonomousDecisionMaker:
    """Makes decisions based on values and goals, not explicit rules."""

    def __init__(self, llm, values: List[str], tools: List[Tool]):
        self.llm = llm
        self.values = values  # e.g., ["accuracy", "efficiency", "safety"]
        self.tools = tools

    async def decide(self, situation: str, goal: str) -> dict:
        """Make a decision without explicit instructions."""

        # 1. Understand the situation
        understanding = await self.analyze_situation(situation)

        # 2. Generate options
        options = await self.generate_options(situation, goal)

        # 3. Evaluate each option against values and goal
        evaluations = []
        for option in options:
            eval_result = await self.evaluate_option(option, understanding, goal)
            evaluations.append(eval_result)

        # 4. Select best option
        best_option = self.select_best(evaluations)

        # 5. Explain the decision (for transparency)
        explanation = await self.explain_decision(best_option, evaluations)

        return {
            "decision": best_option["action"],
            "reasoning": explanation,
            "confidence": best_option["score"],
            "alternatives_considered": len(options)
        }

    async def generate_options(self, situation: str, goal: str) -> List[dict]:
        """Generate possible courses of action."""
        prompt = f"""
        Situation: {situation}
        Goal: {goal}
        Available Tools: {[t.name for t in self.tools]}

        Generate 3-5 different approaches to handle this situation.
        For each approach, describe:
        1. What action to take
        2. What tools to use
        3. Expected outcome
        """

        response = await self.llm.generate(prompt)
        return self.parse_options(response)

    async def evaluate_option(self, option: dict, context: dict, goal: str) -> dict:
        """Evaluate an option against values and goals."""
        prompt = f"""
        Option: {option['action']}
        Expected Outcome: {option['expected_outcome']}
        Goal: {goal}
        Values to consider: {self.values}
        Context: {context}

        Rate this option on:
        1. Goal alignment (0-10): How well does it serve the goal?
        2. Value alignment (0-10): How well does it align with values?
        3. Feasibility (0-10): How likely is it to succeed?
        4. Risk (0-10, lower is better): What could go wrong?

        Provide scores and brief reasoning.
        """

        response = await self.llm.generate(prompt)
        scores = self.parse_evaluation(response)

        # Calculate weighted score
        scores["total"] = (
            scores["goal_alignment"] * 0.4 +
            scores["value_alignment"] * 0.3 +
            scores["feasibility"] * 0.2 +
            (10 - scores["risk"]) * 0.1
        )

        return {"option": option, "scores": scores}

    def select_best(self, evaluations: List[dict]) -> dict:
        """Select the best option based on evaluations."""
        return max(evaluations, key=lambda e: e["scores"]["total"])
```

---

## Maintaining Alignment While Operating Independently

### The Alignment Challenge

As agents operate more independently, how do we ensure they remain aligned with our intentions?

### Analogical Thinking: Constitutional AI

Just as a constitution provides principles for human governance without specifying every law, we can give agents constitutional principles:

```python
class ConstitutionalAgent:
    """Agent that operates according to constitutional principles."""

    CONSTITUTION = """
    1. HELPFULNESS: Prioritize being genuinely helpful to the user's real needs.
    2. ACCURACY: Prefer being uncertain to being confidently wrong.
    3. SAFETY: Never take actions that could cause harm.
    4. TRANSPARENCY: Explain reasoning when making important decisions.
    5. BOUNDARIES: Recognize and respect the limits of your authority.
    6. ESCALATION: Ask for human input when facing novel ethical dilemmas.
    """

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    async def evaluate_action(self, proposed_action: str, context: str) -> dict:
        """Evaluate whether an action aligns with constitutional principles."""
        prompt = f"""
        Constitution:
        {self.CONSTITUTION}

        Proposed Action: {proposed_action}
        Context: {context}

        Evaluate this action against each constitutional principle.
        For each principle, state: ALIGNED, UNCLEAR, or CONFLICT

        If any CONFLICT exists, the action should not be taken.
        If UNCLEAR exists, consider asking for clarification.
        """

        evaluation = await self.llm.generate(prompt)
        return self.parse_constitutional_eval(evaluation)

    async def act(self, intent: str, context: str) -> dict:
        """Take action with constitutional oversight."""

        # Generate proposed action
        proposed = await self.plan_action(intent, context)

        # Constitutional review
        eval_result = await self.evaluate_action(proposed, context)

        if eval_result["has_conflict"]:
            return {
                "action_taken": False,
                "reason": "Constitutional conflict",
                "conflicts": eval_result["conflicts"]
            }

        if eval_result["has_unclear"]:
            return {
                "action_taken": False,
                "reason": "Needs clarification",
                "unclear_points": eval_result["unclear"]
            }

        # Action is constitutional - proceed
        result = await self.execute_action(proposed)

        return {
            "action_taken": True,
            "action": proposed,
            "result": result,
            "constitutional_review": "passed"
        }
```

### Value Alignment Through Reward Modeling

```python
class ValueAlignedAgent:
    """Agent that maintains alignment through explicit value modeling."""

    def __init__(self, llm, tools, value_model):
        self.llm = llm
        self.tools = tools
        self.value_model = value_model  # Scores actions on alignment

    async def choose_action(self, options: List[str], context: str) -> str:
        """Choose the most value-aligned action."""
        scored_options = []

        for option in options:
            # Get value alignment score
            value_score = await self.value_model.score(option, context)

            # Get effectiveness score
            effectiveness = await self.estimate_effectiveness(option, context)

            # Combined score balances values and effectiveness
            combined = value_score * 0.6 + effectiveness * 0.4

            scored_options.append({
                "option": option,
                "value_score": value_score,
                "effectiveness": effectiveness,
                "combined": combined
            })

        # Choose highest combined score, but never below value threshold
        valid_options = [o for o in scored_options if o["value_score"] >= 0.7]

        if not valid_options:
            raise AlignmentError("No sufficiently aligned options available")

        return max(valid_options, key=lambda o: o["combined"])["option"]
```

---

## Complete Example: Autonomous Research Agent

Let's bring it all together with a complete autonomous research agent:

```python
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# --- Tools ---

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation would use actual search API
    return f"Search results for: {query}"

@tool
def read_document(url: str) -> str:
    """Read and extract content from a document."""
    return f"Content from: {url}"

@tool
def write_notes(content: str, filename: str) -> str:
    """Save notes to a file."""
    return f"Saved to {filename}"

# --- Autonomous Agent ---

class AutonomousResearchAgent:
    """
    An autonomous research agent that:
    1. Takes a high-level research goal
    2. Decomposes it into sub-goals
    3. Pursues sub-goals independently
    4. Learns from successes and failures
    5. Operates within constitutional principles
    """

    CONSTITUTION = """
    1. Seek accurate, well-sourced information
    2. Acknowledge uncertainty and limitations
    3. Respect intellectual property and cite sources
    4. Stay focused on the research goal
    5. Escalate when findings are ambiguous or concerning
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.tools = [web_search, read_document, write_notes]
        self.memory = ResearchMemory()
        self.goal_tree = None

    async def research(self, topic: str, depth: str = "moderate") -> Dict:
        """
        Conduct autonomous research on a topic.

        Args:
            topic: What to research
            depth: "shallow", "moderate", or "deep"

        Returns:
            Research findings and methodology
        """
        print(f"Starting autonomous research on: {topic}")

        # 1. Understand the research goal
        research_goal = await self.formulate_goal(topic, depth)
        self.goal_tree = GoalTree(research_goal)

        # 2. Generate initial sub-goals
        await self.decompose_initial_goals()

        # 3. Research loop
        max_iterations = {"shallow": 5, "moderate": 15, "deep": 30}[depth]

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Get next goal to pursue
            current_goal = self.goal_tree.get_next_actionable()

            if not current_goal:
                print("All goals completed or blocked")
                break

            print(f"Pursuing: {current_goal.description}")

            # Decide how to pursue this goal
            action = await self.decide_action(current_goal)

            # Constitutional check
            if not await self.constitutional_check(action):
                print("Action blocked by constitution")
                current_goal.status = "blocked"
                continue

            # Execute
            result = await self.execute(action)

            # Update goal status
            if await self.goal_achieved(current_goal, result):
                current_goal.status = "completed"
                self.memory.add_finding(current_goal.description, result)
            else:
                current_goal.attempts += 1
                if current_goal.attempts >= 3:
                    # Try different approach or decompose further
                    await self.handle_stuck_goal(current_goal)

            # Check if we should add new sub-goals based on findings
            new_goals = await self.identify_new_goals(result)
            for goal in new_goals:
                self.goal_tree.add_goal(goal)

        # 4. Compile findings
        return await self.compile_research_report()

    async def formulate_goal(self, topic: str, depth: str) -> Goal:
        """Formulate a clear research goal from topic."""
        prompt = f"""
        Research Topic: {topic}
        Depth: {depth}

        Formulate a clear research goal including:
        1. Specific questions to answer
        2. What would constitute successful research
        3. Scope boundaries (what's in/out)

        Be specific and measurable.
        """
        response = await self.llm.ainvoke(prompt)
        return self.parse_goal(response.content)

    async def decide_action(self, goal: Goal) -> Dict:
        """Decide what action to take for a goal."""
        prompt = f"""
        Current Goal: {goal.description}
        Available Tools: {[t.name for t in self.tools]}
        Previous Findings: {self.memory.get_relevant(goal.description)}

        What is the best single action to make progress on this goal?
        Consider what information we already have vs. what we need.

        Return:
        - tool: which tool to use
        - arguments: what arguments to pass
        - reasoning: why this action
        """
        response = await self.llm.ainvoke(prompt)
        return self.parse_action(response.content)

    async def constitutional_check(self, action: Dict) -> bool:
        """Verify action aligns with constitution."""
        prompt = f"""
        Constitution:
        {self.CONSTITUTION}

        Proposed Action:
        Tool: {action['tool']}
        Arguments: {action['arguments']}
        Reasoning: {action['reasoning']}

        Does this action align with all constitutional principles?
        Answer YES or NO with brief explanation.
        """
        response = await self.llm.ainvoke(prompt)
        return "YES" in response.content.upper()

    async def compile_research_report(self) -> Dict:
        """Compile final research report."""
        findings = self.memory.get_all_findings()

        prompt = f"""
        Research Findings:
        {findings}

        Goal Completion Status:
        {self.goal_tree.get_status_summary()}

        Compile a research report including:
        1. Executive Summary
        2. Key Findings
        3. Methodology
        4. Limitations
        5. Recommendations for further research
        """

        response = await self.llm.ainvoke(prompt)

        return {
            "report": response.content,
            "goals_completed": self.goal_tree.completion_ratio(),
            "sources": self.memory.get_sources(),
            "methodology": self.memory.get_action_log()
        }


# --- Supporting Classes ---

@dataclass
class Goal:
    description: str
    success_criteria: str
    priority: int = 5
    status: str = "pending"
    attempts: int = 0
    parent: Optional['Goal'] = None
    children: List['Goal'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class GoalTree:
    def __init__(self, root_goal: Goal):
        self.root = root_goal

    def get_next_actionable(self) -> Optional[Goal]:
        """Find next goal to work on using DFS."""
        return self._find_actionable(self.root)

    def _find_actionable(self, goal: Goal) -> Optional[Goal]:
        if goal.status == "completed":
            return None

        # If no children, this is actionable
        if not goal.children:
            if goal.status in ["pending", "in_progress"]:
                return goal
            return None

        # Check children
        for child in sorted(goal.children, key=lambda g: -g.priority):
            actionable = self._find_actionable(child)
            if actionable:
                return actionable

        return None

    def completion_ratio(self) -> float:
        """Calculate overall completion."""
        return self._calc_completion(self.root)

    def _calc_completion(self, goal: Goal) -> float:
        if not goal.children:
            return 1.0 if goal.status == "completed" else 0.0
        return sum(self._calc_completion(c) for c in goal.children) / len(goal.children)


class ResearchMemory:
    def __init__(self):
        self.findings = []
        self.sources = []
        self.action_log = []

    def add_finding(self, goal: str, result: str):
        self.findings.append({"goal": goal, "result": result})

    def get_relevant(self, query: str) -> List[Dict]:
        # Simple relevance - in production, use embeddings
        return self.findings[-5:]  # Last 5 findings

    def get_all_findings(self) -> List[Dict]:
        return self.findings

    def get_sources(self) -> List[str]:
        return self.sources

    def get_action_log(self) -> List[Dict]:
        return self.action_log


# --- Usage ---

async def main():
    agent = AutonomousResearchAgent()

    result = await agent.research(
        topic="Current state of autonomous AI agents in enterprise applications",
        depth="moderate"
    )

    print("\n" + "="*60)
    print("RESEARCH REPORT")
    print("="*60)
    print(result["report"])
    print(f"\nGoals completed: {result['goals_completed']*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Exercises

### Exercise 1: Goal Decomposition
Take the goal "Plan a company offsite" and manually decompose it into a goal hierarchy. Then implement a `GoalDecomposer` class that uses an LLM to do this automatically.

### Exercise 2: Decision Without Instructions
Create a `NoveDecisionMaker` that can handle situations it wasn't explicitly programmed for by reasoning from first principles and values.

### Exercise 3: Constitutional Constraints
Design a constitution for a customer service agent. Implement the constitutional checking system and test it with edge cases.

### Exercise 4: Autonomy Levels
Implement versions of a "meeting scheduler" agent at each autonomy level (0-3). Compare their capabilities and limitations.

---

## Key Takeaways

1. **Autonomy is a spectrum**, not a binary state. Right-size autonomy to the task.

2. **True autonomy requires goals, perception, decision-making, action, and feedback**. Remove any component and autonomy degrades.

3. **Goal hierarchies enable independence**. Agents can generate sub-goals rather than requiring explicit instructions.

4. **Constitutional principles maintain alignment**. Define values, not rules.

5. **Transparency enables trust**. Autonomous agents should explain their decisions.

---

## Next Steps

In [Module 13.2: Self-Improving Agent Systems](02_self_improving_systems.md), we'll explore how agents can learn from their experiences and get better over time—the key to truly useful autonomy.

---

*"The first principle is that you must not fool yourself—and you are the easiest person to fool."* — Richard Feynman

Apply this to agents: the first principle of autonomous systems is robust self-evaluation.
