# Module 13.2: Self-Improving Agent Systems

> "Intelligence is the ability to adapt to change." — Stephen Hawking

## Introduction

The most powerful autonomous agents don't just execute—they learn. In this module, we'll explore how self-improvement emerges from the interaction of experience, evaluation, and adaptation. Using emergence thinking, we'll build agents that get measurably better over time without explicit reprogramming.

---

## Learning Objectives

By the end of this module, you will:
- Understand how learning emerges from simple feedback loops
- Implement experience accumulation and retrieval systems
- Build strategy libraries that evolve based on outcomes
- Create evaluation frameworks that drive improvement
- Design memory architectures that support continuous learning

---

## Emergence Thinking: How Learning Happens

### The Fundamental Learning Loop

At its core, all learning follows this pattern:

```
      ┌─────────────────────────────────┐
      │                                 │
      ▼                                 │
   ┌──────┐    ┌─────────┐    ┌────────┴───┐
   │ Act  │───▶│ Observe │───▶│  Evaluate  │
   └──────┘    └─────────┘    └────────────┘
      ▲                             │
      │                             │
      │        ┌──────────┐         │
      └────────│  Update  │◀────────┘
               └──────────┘

Simple components:
├── Act: Take an action
├── Observe: See the outcome
├── Evaluate: Judge against goals
└── Update: Modify future behavior

Emergence: The COMBINATION creates learning
```

**Key Insight:** No single component "learns." Learning emerges from their interaction.

### Analogical Thinking: Learning Like a Scientist

Think of a self-improving agent like a scientist conducting experiments:

| Scientific Method | Agent Self-Improvement |
|------------------|----------------------|
| Hypothesis | Strategy selection |
| Experiment | Action execution |
| Data collection | Outcome observation |
| Analysis | Performance evaluation |
| Theory refinement | Strategy update |
| Literature review | Experience retrieval |
| Peer review | External validation |

---

## Experience Accumulation

### First Principles: What Is Experience?

Experience is structured information about past actions and their outcomes that can inform future decisions.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

@dataclass
class Experience:
    """A single learning experience."""

    # What happened
    situation: str          # The context/input
    action_taken: str       # What the agent did
    outcome: str            # What resulted
    timestamp: datetime = field(default_factory=datetime.now)

    # Evaluation
    success: bool = False
    quality_score: float = 0.0  # 0-1 scale
    feedback: Optional[str] = None

    # Metadata for retrieval
    tags: List[str] = field(default_factory=list)
    strategy_used: Optional[str] = None

    # For learning
    what_worked: Optional[str] = None
    what_failed: Optional[str] = None
    lesson_learned: Optional[str] = None

    def to_embedding_text(self) -> str:
        """Convert to text for embedding-based retrieval."""
        return f"""
        Situation: {self.situation}
        Action: {self.action_taken}
        Outcome: {self.outcome}
        Success: {self.success}
        Lesson: {self.lesson_learned or 'None recorded'}
        """

    def to_prompt_context(self) -> str:
        """Format for inclusion in prompts."""
        return f"""
        [Past Experience]
        When: {self.situation}
        I tried: {self.action_taken}
        Result: {self.outcome} ({'success' if self.success else 'failure'})
        Lesson: {self.lesson_learned or 'N/A'}
        """
```

### Experience Memory Architecture

```python
from abc import ABC, abstractmethod
import chromadb
from sentence_transformers import SentenceTransformer

class ExperienceMemory(ABC):
    """Abstract base for experience storage."""

    @abstractmethod
    def store(self, experience: Experience) -> str:
        """Store an experience, return ID."""
        pass

    @abstractmethod
    def retrieve_similar(self, situation: str, k: int = 5) -> List[Experience]:
        """Retrieve k most relevant experiences."""
        pass

    @abstractmethod
    def retrieve_by_strategy(self, strategy: str, k: int = 5) -> List[Experience]:
        """Get experiences using a specific strategy."""
        pass


class VectorExperienceMemory(ExperienceMemory):
    """Experience memory using vector similarity search."""

    def __init__(self, collection_name: str = "agent_experiences"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def store(self, experience: Experience) -> str:
        """Store experience with embedding."""
        exp_id = f"exp_{datetime.now().timestamp()}"

        # Create embedding from the experience
        embedding = self.encoder.encode(experience.to_embedding_text()).tolist()

        # Store in vector DB
        self.collection.add(
            ids=[exp_id],
            embeddings=[embedding],
            documents=[experience.to_embedding_text()],
            metadatas=[{
                "success": experience.success,
                "quality_score": experience.quality_score,
                "strategy": experience.strategy_used or "unknown",
                "timestamp": experience.timestamp.isoformat(),
                "tags": json.dumps(experience.tags)
            }]
        )

        return exp_id

    def retrieve_similar(self, situation: str, k: int = 5) -> List[Experience]:
        """Find experiences from similar situations."""
        query_embedding = self.encoder.encode(situation).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"]
        )

        return self._parse_results(results)

    def retrieve_successful(self, situation: str, k: int = 3) -> List[Experience]:
        """Find successful experiences from similar situations."""
        query_embedding = self.encoder.encode(situation).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,  # Get more, then filter
            where={"success": True},
            include=["documents", "metadatas"]
        )

        return self._parse_results(results)[:k]

    def get_strategy_performance(self, strategy: str) -> Dict[str, float]:
        """Calculate performance metrics for a strategy."""
        results = self.collection.get(
            where={"strategy": strategy},
            include=["metadatas"]
        )

        if not results["metadatas"]:
            return {"success_rate": 0.0, "avg_quality": 0.0, "sample_size": 0}

        metadatas = results["metadatas"]
        successes = sum(1 for m in metadatas if m.get("success"))
        qualities = [m.get("quality_score", 0) for m in metadatas]

        return {
            "success_rate": successes / len(metadatas),
            "avg_quality": sum(qualities) / len(qualities),
            "sample_size": len(metadatas)
        }
```

---

## Strategy Libraries

### First Principles: What Is a Strategy?

A strategy is a reusable approach to handling a class of situations. Strategies abstract away from specific actions to capture *patterns* of effective behavior.

```python
@dataclass
class Strategy:
    """A reusable approach to handling situations."""

    name: str
    description: str
    applicable_when: str  # Conditions for using this strategy
    approach: str         # How to execute the strategy
    examples: List[str] = field(default_factory=list)

    # Performance tracking
    times_used: int = 0
    successes: int = 0
    total_quality: float = 0.0

    # Adaptive parameters
    confidence: float = 0.5  # Initial uncertainty
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.times_used, 1)

    @property
    def avg_quality(self) -> float:
        return self.total_quality / max(self.times_used, 1)

    def update_from_outcome(self, success: bool, quality: float):
        """Update strategy metrics based on outcome."""
        self.times_used += 1
        if success:
            self.successes += 1
        self.total_quality += quality
        self.last_updated = datetime.now()

        # Update confidence based on sample size
        # More data = more confident in our estimates
        self.confidence = min(0.95, 0.5 + (self.times_used * 0.05))


class StrategyLibrary:
    """Collection of strategies that evolves over time."""

    def __init__(self, llm):
        self.llm = llm
        self.strategies: Dict[str, Strategy] = {}
        self.selection_history: List[Dict] = []

    def add_strategy(self, strategy: Strategy):
        """Add a new strategy to the library."""
        self.strategies[strategy.name] = strategy

    async def select_strategy(self, situation: str) -> Strategy:
        """Select the best strategy for a situation."""

        if not self.strategies:
            raise ValueError("No strategies available")

        # Get applicable strategies
        applicable = await self.get_applicable_strategies(situation)

        if not applicable:
            # No good matches - might need to create new strategy
            return await self.create_new_strategy(situation)

        # Use Upper Confidence Bound (UCB) for exploration/exploitation
        selected = self.ucb_select(applicable)

        self.selection_history.append({
            "situation": situation,
            "selected": selected.name,
            "alternatives": [s.name for s in applicable]
        })

        return selected

    async def get_applicable_strategies(self, situation: str) -> List[Strategy]:
        """Find strategies applicable to this situation."""
        prompt = f"""
        Situation: {situation}

        Available Strategies:
        {self._format_strategies()}

        Which strategies are applicable to this situation?
        Return a comma-separated list of strategy names, or "NONE" if none apply.
        """

        response = await self.llm.ainvoke(prompt)
        names = [n.strip() for n in response.content.split(",")]

        return [self.strategies[n] for n in names if n in self.strategies]

    def ucb_select(self, strategies: List[Strategy]) -> Strategy:
        """
        Select strategy using Upper Confidence Bound algorithm.
        Balances exploitation (use what works) with exploration (try uncertain options).
        """
        import math

        total_selections = sum(s.times_used for s in strategies) + 1

        def ucb_score(strategy: Strategy) -> float:
            if strategy.times_used == 0:
                return float('inf')  # Always try untested strategies

            exploitation = strategy.success_rate * 0.7 + strategy.avg_quality * 0.3
            exploration = math.sqrt(2 * math.log(total_selections) / strategy.times_used)

            return exploitation + exploration

        return max(strategies, key=ucb_score)

    async def create_new_strategy(self, situation: str) -> Strategy:
        """Create a new strategy for an unhandled situation type."""
        prompt = f"""
        I encountered a situation that none of my existing strategies handle well.

        Situation: {situation}

        Existing strategies:
        {self._format_strategies()}

        Design a new strategy to handle this type of situation.
        Include:
        - name: A short descriptive name
        - description: What this strategy does
        - applicable_when: Conditions for using it
        - approach: Step-by-step how to execute it
        """

        response = await self.llm.ainvoke(prompt)
        new_strategy = self._parse_strategy(response.content)

        self.add_strategy(new_strategy)
        return new_strategy

    async def refine_strategy(self, strategy_name: str, experiences: List[Experience]):
        """Improve a strategy based on accumulated experience."""
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return

        successful = [e for e in experiences if e.success]
        failed = [e for e in experiences if not e.success]

        prompt = f"""
        Strategy: {strategy.name}
        Current Approach: {strategy.approach}

        Recent Successful Uses:
        {self._format_experiences(successful)}

        Recent Failures:
        {self._format_experiences(failed)}

        Based on this experience, suggest improvements to the strategy:
        1. What patterns lead to success?
        2. What patterns lead to failure?
        3. How should the approach be modified?
        """

        response = await self.llm.ainvoke(prompt)
        improvements = response.content

        # Update strategy with improvements
        strategy.approach = await self.integrate_improvements(
            strategy.approach,
            improvements
        )

    def _format_strategies(self) -> str:
        return "\n".join([
            f"- {s.name}: {s.description} (success rate: {s.success_rate:.1%})"
            for s in self.strategies.values()
        ])
```

---

## The Self-Improvement Loop

### Emergence: From Components to Learning

```python
class SelfImprovingAgent:
    """
    An agent that improves through the emergent interaction of:
    - Experience memory (what happened)
    - Strategy library (how to act)
    - Evaluator (what worked)
    - Updater (how to change)
    """

    def __init__(self, llm, tools: List):
        self.llm = llm
        self.tools = tools

        # The four components that create emergent learning
        self.experience_memory = VectorExperienceMemory()
        self.strategy_library = StrategyLibrary(llm)
        self.evaluator = OutcomeEvaluator(llm)
        self.updater = StrategyUpdater(llm)

        # Metrics for tracking improvement
        self.performance_history: List[Dict] = []

    async def execute_with_learning(self, task: str) -> Dict:
        """Execute a task while learning from the experience."""

        # 1. Retrieve relevant past experiences
        similar_experiences = self.experience_memory.retrieve_similar(task, k=5)
        successful_approaches = self.experience_memory.retrieve_successful(task, k=3)

        # 2. Select strategy informed by experience
        strategy = await self.strategy_library.select_strategy(task)

        # 3. Plan action using strategy + experience context
        action_plan = await self.plan_action(
            task=task,
            strategy=strategy,
            past_experiences=similar_experiences,
            successful_approaches=successful_approaches
        )

        # 4. Execute the action
        start_time = datetime.now()
        result = await self.execute_action(action_plan)
        execution_time = (datetime.now() - start_time).total_seconds()

        # 5. Evaluate the outcome
        evaluation = await self.evaluator.evaluate(
            task=task,
            action=action_plan,
            result=result,
            strategy=strategy
        )

        # 6. Create and store experience
        experience = Experience(
            situation=task,
            action_taken=action_plan["action"],
            outcome=str(result),
            success=evaluation["success"],
            quality_score=evaluation["quality_score"],
            strategy_used=strategy.name,
            what_worked=evaluation.get("what_worked"),
            what_failed=evaluation.get("what_failed"),
            lesson_learned=evaluation.get("lesson")
        )
        self.experience_memory.store(experience)

        # 7. Update strategy based on outcome
        strategy.update_from_outcome(
            success=evaluation["success"],
            quality=evaluation["quality_score"]
        )

        # 8. Periodically refine strategies
        if strategy.times_used % 10 == 0:
            recent_experiences = self.experience_memory.retrieve_by_strategy(
                strategy.name, k=10
            )
            await self.strategy_library.refine_strategy(
                strategy.name,
                recent_experiences
            )

        # 9. Track performance for measuring improvement
        self.performance_history.append({
            "timestamp": datetime.now(),
            "task": task,
            "strategy": strategy.name,
            "success": evaluation["success"],
            "quality": evaluation["quality_score"],
            "execution_time": execution_time
        })

        return {
            "result": result,
            "evaluation": evaluation,
            "strategy_used": strategy.name,
            "experiences_consulted": len(similar_experiences)
        }

    async def plan_action(
        self,
        task: str,
        strategy: Strategy,
        past_experiences: List[Experience],
        successful_approaches: List[Experience]
    ) -> Dict:
        """Plan an action informed by strategy and experience."""

        experience_context = "\n".join([
            e.to_prompt_context() for e in past_experiences
        ])

        successful_context = "\n".join([
            f"- Success: {e.action_taken} -> {e.outcome}"
            for e in successful_approaches
        ])

        prompt = f"""
        Task: {task}

        Strategy to use: {strategy.name}
        Strategy approach: {strategy.approach}

        Relevant past experiences:
        {experience_context}

        Previously successful approaches for similar tasks:
        {successful_context}

        Based on the strategy and past experiences, plan the specific action to take.
        Learn from what worked and avoid what failed.

        Return:
        - action: The specific action to take
        - tool: Which tool to use
        - arguments: Tool arguments
        - reasoning: Why this approach (referencing experience if relevant)
        """

        response = await self.llm.ainvoke(prompt)
        return self._parse_action_plan(response.content)

    def get_improvement_metrics(self) -> Dict:
        """Calculate metrics showing improvement over time."""
        if len(self.performance_history) < 10:
            return {"status": "insufficient_data"}

        # Split into early and recent
        midpoint = len(self.performance_history) // 2
        early = self.performance_history[:midpoint]
        recent = self.performance_history[midpoint:]

        early_success_rate = sum(1 for p in early if p["success"]) / len(early)
        recent_success_rate = sum(1 for p in recent if p["success"]) / len(recent)

        early_avg_quality = sum(p["quality"] for p in early) / len(early)
        recent_avg_quality = sum(p["quality"] for p in recent) / len(recent)

        early_avg_time = sum(p["execution_time"] for p in early) / len(early)
        recent_avg_time = sum(p["execution_time"] for p in recent) / len(recent)

        return {
            "success_rate_improvement": recent_success_rate - early_success_rate,
            "quality_improvement": recent_avg_quality - early_avg_quality,
            "speed_improvement": early_avg_time - recent_avg_time,  # Lower is better
            "total_experiences": len(self.performance_history),
            "strategies_in_library": len(self.strategy_library.strategies)
        }
```

### The Evaluator: Judging Outcomes

```python
class OutcomeEvaluator:
    """Evaluates action outcomes to drive learning."""

    def __init__(self, llm):
        self.llm = llm

    async def evaluate(
        self,
        task: str,
        action: Dict,
        result: Any,
        strategy: Strategy
    ) -> Dict:
        """Comprehensively evaluate an action's outcome."""

        prompt = f"""
        Task: {task}
        Action Taken: {action['action']}
        Strategy Used: {strategy.name}
        Result: {result}

        Evaluate this outcome:

        1. SUCCESS (true/false): Did the action accomplish the task?

        2. QUALITY (0.0-1.0): How well was it accomplished?
           - 0.0-0.3: Poor - major issues or incomplete
           - 0.4-0.6: Acceptable - task done but room for improvement
           - 0.7-0.9: Good - task done well
           - 0.9-1.0: Excellent - optimal execution

        3. WHAT_WORKED: What aspects of the approach were effective?

        4. WHAT_FAILED: What aspects didn't work well or could be improved?

        5. LESSON: What should be remembered for similar future tasks?

        6. STRATEGY_FIT: Was this strategy appropriate for this task? (good_fit/poor_fit)

        Provide honest, constructive evaluation.
        """

        response = await self.llm.ainvoke(prompt)
        return self._parse_evaluation(response.content)

    async def comparative_evaluate(
        self,
        results: List[Dict]  # Multiple results to compare
    ) -> Dict:
        """Compare multiple approaches to identify best practices."""

        prompt = f"""
        Compare these different approaches to similar tasks:

        {self._format_results(results)}

        Analysis needed:
        1. Which approaches were most effective? Why?
        2. What patterns distinguish successful from failed attempts?
        3. What general principles can we extract?
        4. Any surprising findings?
        """

        response = await self.llm.ainvoke(prompt)
        return self._parse_comparative_analysis(response.content)
```

---

## Memory Architectures for Learning

### First Principles: Types of Memory

```
Memory Types for Learning:
├── Episodic: Specific experiences (what happened when)
├── Semantic: Extracted knowledge (general facts learned)
├── Procedural: How-to knowledge (strategies and skills)
└── Working: Current context (active task state)
```

### Integrated Memory System

```python
class IntegratedLearningMemory:
    """
    Memory system that supports all types of learning.

    Emergence: Complex learning emerges from the interaction
    of different memory types.
    """

    def __init__(self, llm):
        self.llm = llm

        # Episodic: Specific experiences
        self.episodic = VectorExperienceMemory("episodic")

        # Semantic: Extracted knowledge and facts
        self.semantic = KnowledgeBase("semantic")

        # Procedural: Strategies and skills
        self.procedural = StrategyLibrary(llm)

        # Working: Current context
        self.working = WorkingMemory(max_items=10)

    async def process_experience(self, experience: Experience):
        """Process new experience across all memory types."""

        # 1. Store in episodic memory
        self.episodic.store(experience)

        # 2. Extract semantic knowledge
        knowledge = await self.extract_knowledge(experience)
        if knowledge:
            self.semantic.add(knowledge)

        # 3. Update procedural knowledge (strategies)
        if experience.strategy_used:
            strategy = self.procedural.strategies.get(experience.strategy_used)
            if strategy:
                strategy.update_from_outcome(
                    experience.success,
                    experience.quality_score
                )

        # 4. Update working memory with recent context
        self.working.add({
            "type": "recent_experience",
            "summary": experience.lesson_learned,
            "success": experience.success
        })

    async def extract_knowledge(self, experience: Experience) -> Optional[Dict]:
        """Extract generalizable knowledge from specific experience."""

        prompt = f"""
        Experience:
        - Situation: {experience.situation}
        - Action: {experience.action_taken}
        - Outcome: {experience.outcome}
        - Success: {experience.success}

        Is there generalizable knowledge from this experience?
        Something that would apply to other situations, not just this specific one?

        If yes, extract it as a general principle.
        If no (it's too specific), return "NO_GENERALIZABLE_KNOWLEDGE"
        """

        response = await self.llm.ainvoke(prompt)

        if "NO_GENERALIZABLE_KNOWLEDGE" in response.content:
            return None

        return {
            "principle": response.content,
            "source_experience": experience.situation,
            "confidence": 0.5  # Low initial confidence
        }

    async def get_relevant_context(self, situation: str) -> Dict:
        """Gather relevant information from all memory types."""

        # Episodic: Similar past experiences
        similar_experiences = self.episodic.retrieve_similar(situation, k=3)

        # Semantic: Relevant knowledge
        relevant_knowledge = self.semantic.query(situation, k=3)

        # Procedural: Applicable strategies
        applicable_strategies = await self.procedural.get_applicable_strategies(situation)

        # Working: Current context
        current_context = self.working.get_all()

        return {
            "past_experiences": similar_experiences,
            "relevant_knowledge": relevant_knowledge,
            "applicable_strategies": applicable_strategies,
            "current_context": current_context
        }


class KnowledgeBase:
    """Semantic memory - extracted, generalizable knowledge."""

    def __init__(self, name: str):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def add(self, knowledge: Dict):
        """Add a piece of knowledge."""
        embedding = self.encoder.encode(knowledge["principle"]).tolist()

        self.collection.add(
            ids=[f"k_{datetime.now().timestamp()}"],
            embeddings=[embedding],
            documents=[knowledge["principle"]],
            metadatas=[{
                "source": knowledge["source_experience"],
                "confidence": knowledge["confidence"]
            }]
        )

    def query(self, situation: str, k: int = 3) -> List[str]:
        """Find relevant knowledge."""
        embedding = self.encoder.encode(situation).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )

        return results["documents"][0] if results["documents"] else []


class WorkingMemory:
    """Short-term memory for current task context."""

    def __init__(self, max_items: int = 10):
        self.max_items = max_items
        self.items: List[Dict] = []

    def add(self, item: Dict):
        """Add item, removing oldest if at capacity."""
        self.items.append({
            **item,
            "timestamp": datetime.now()
        })

        if len(self.items) > self.max_items:
            self.items.pop(0)

    def get_all(self) -> List[Dict]:
        return self.items

    def clear(self):
        self.items = []
```

---

## Complete Example: Self-Improving Code Assistant

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# --- Tools ---

@tool
def run_code(code: str) -> str:
    """Execute Python code and return output."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals.get('result', 'Code executed successfully'))
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_documentation(query: str) -> str:
    """Search programming documentation."""
    return f"Documentation results for: {query}"

@tool
def analyze_error(error_message: str) -> str:
    """Analyze an error message and suggest fixes."""
    return f"Analysis of: {error_message}"

# --- Self-Improving Code Assistant ---

class SelfImprovingCodeAssistant:
    """
    A code assistant that learns from every interaction.

    Emergence in action:
    - Remembers which approaches work for which problems
    - Develops strategies for common error patterns
    - Gets faster and more accurate over time
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.tools = [run_code, search_documentation, analyze_error]

        # Learning components
        self.memory = IntegratedLearningMemory(self.llm)
        self.strategy_library = StrategyLibrary(self.llm)

        # Initialize with base strategies
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Seed with initial strategies that will be refined through experience."""

        self.strategy_library.add_strategy(Strategy(
            name="direct_implementation",
            description="Directly implement the requested code",
            applicable_when="Request is clear and straightforward",
            approach="""
            1. Understand the requirement
            2. Write the code directly
            3. Test with a simple example
            4. Return the result
            """
        ))

        self.strategy_library.add_strategy(Strategy(
            name="research_first",
            description="Research before implementing",
            applicable_when="Unfamiliar library or concept involved",
            approach="""
            1. Search documentation for relevant APIs
            2. Find examples of similar implementations
            3. Adapt examples to the specific need
            4. Test and refine
            """
        ))

        self.strategy_library.add_strategy(Strategy(
            name="iterative_debugging",
            description="Build incrementally with testing",
            applicable_when="Complex logic or multiple components",
            approach="""
            1. Break into small, testable pieces
            2. Implement and test each piece
            3. Combine pieces gradually
            4. Test the integrated solution
            """
        ))

        self.strategy_library.add_strategy(Strategy(
            name="error_analysis",
            description="Analyze and fix errors systematically",
            applicable_when="Code produces errors",
            approach="""
            1. Capture the full error message
            2. Analyze the error type and location
            3. Search for similar issues and solutions
            4. Apply fix and verify
            """
        ))

    async def help_with_code(self, request: str) -> Dict:
        """
        Help with a coding request while learning from the interaction.
        """

        # 1. Get relevant context from memory
        context = await self.memory.get_relevant_context(request)

        # 2. Select strategy based on request and experience
        strategy = await self.strategy_library.select_strategy(request)

        print(f"Using strategy: {strategy.name}")
        print(f"Based on {len(context['past_experiences'])} similar experiences")

        # 3. Execute with the chosen strategy
        result = await self.execute_with_strategy(request, strategy, context)

        # 4. Evaluate the outcome
        evaluation = await self.evaluate_result(request, result)

        # 5. Create and process experience
        experience = Experience(
            situation=request,
            action_taken=f"Strategy: {strategy.name}",
            outcome=str(result.get("output", result)),
            success=evaluation["success"],
            quality_score=evaluation["quality"],
            strategy_used=strategy.name,
            lesson_learned=evaluation.get("lesson")
        )

        await self.memory.process_experience(experience)

        # 6. Return result with learning metadata
        return {
            "result": result,
            "strategy_used": strategy.name,
            "success": evaluation["success"],
            "experiences_consulted": len(context["past_experiences"]),
            "knowledge_applied": len(context["relevant_knowledge"])
        }

    async def execute_with_strategy(
        self,
        request: str,
        strategy: Strategy,
        context: Dict
    ) -> Dict:
        """Execute the request using the selected strategy."""

        # Format context for the LLM
        experience_hints = "\n".join([
            f"- Similar task: {e.situation[:50]}... -> {'Success' if e.success else 'Failed'}: {e.lesson_learned or 'No lesson recorded'}"
            for e in context["past_experiences"]
        ])

        knowledge_hints = "\n".join([
            f"- {k}" for k in context["relevant_knowledge"]
        ])

        prompt = f"""
        Coding Request: {request}

        Strategy: {strategy.name}
        Approach: {strategy.approach}

        Lessons from similar past tasks:
        {experience_hints or "No similar experiences yet"}

        Relevant knowledge:
        {knowledge_hints or "No relevant knowledge yet"}

        Follow the strategy approach, applying lessons learned.
        Use available tools: {[t.name for t in self.tools]}

        Provide the solution.
        """

        # Execute (simplified - in production would use tool calling)
        response = await self.llm.ainvoke(prompt)

        return {
            "output": response.content,
            "strategy": strategy.name
        }

    async def evaluate_result(self, request: str, result: Dict) -> Dict:
        """Evaluate the quality of the result."""

        prompt = f"""
        Request: {request}
        Solution provided: {result['output'][:500]}...

        Evaluate:
        1. Does this correctly address the request? (success: true/false)
        2. Quality score (0.0-1.0)
        3. What lesson should we remember?

        Be honest - if there are issues, note them.
        """

        response = await self.llm.ainvoke(prompt)
        return self._parse_evaluation(response.content)

    def get_learning_report(self) -> Dict:
        """Report on what the agent has learned."""

        strategy_performance = {}
        for name, strategy in self.strategy_library.strategies.items():
            strategy_performance[name] = {
                "times_used": strategy.times_used,
                "success_rate": f"{strategy.success_rate:.1%}",
                "avg_quality": f"{strategy.avg_quality:.2f}"
            }

        return {
            "strategies": strategy_performance,
            "total_experiences": self.memory.episodic.collection.count(),
            "knowledge_items": self.memory.semantic.collection.count()
        }

    def _parse_evaluation(self, response: str) -> Dict:
        """Parse evaluation response."""
        # Simplified parsing - in production, use structured output
        success = "success: true" in response.lower() or "correctly" in response.lower()
        quality = 0.7 if success else 0.3  # Simplified

        return {
            "success": success,
            "quality": quality,
            "lesson": response.split("lesson")[-1][:200] if "lesson" in response.lower() else None
        }


# --- Usage Example ---

async def demonstrate_learning():
    """Demonstrate the self-improving agent."""

    assistant = SelfImprovingCodeAssistant()

    # Simulate a series of interactions
    requests = [
        "Write a function to calculate factorial",
        "Fix this error: TypeError: 'NoneType' object is not subscriptable",
        "Write a function to reverse a string",
        "Implement binary search",
        "Debug this: IndexError in my loop",
        "Write a function to check if a string is palindrome",
        "Help me understand this error: KeyError 'name'",
        "Create a simple class for a bank account",
    ]

    for request in requests:
        print(f"\n{'='*60}")
        print(f"Request: {request}")
        print('='*60)

        result = await assistant.help_with_code(request)

        print(f"Strategy used: {result['strategy_used']}")
        print(f"Success: {result['success']}")
        print(f"Experiences consulted: {result['experiences_consulted']}")

    # Show learning report
    print("\n" + "="*60)
    print("LEARNING REPORT")
    print("="*60)

    report = assistant.get_learning_report()
    print("\nStrategy Performance:")
    for name, stats in report["strategies"].items():
        print(f"  {name}:")
        print(f"    Used: {stats['times_used']} times")
        print(f"    Success rate: {stats['success_rate']}")
        print(f"    Avg quality: {stats['avg_quality']}")


if __name__ == "__main__":
    asyncio.run(demonstrate_learning())
```

---

## Exercises

### Exercise 1: Experience Replay
Implement a system that periodically "replays" past experiences to extract new insights, similar to how humans learn by reflecting on past events.

### Exercise 2: Strategy Evolution
Create a mechanism for strategies to "evolve" - combining successful elements from multiple strategies to create new ones.

### Exercise 3: Forgetting Mechanism
Implement intelligent forgetting - the ability to deprecate outdated knowledge when the environment changes.

### Exercise 4: Learning Visualization
Build a dashboard that visualizes the agent's learning over time - showing improvement curves, strategy effectiveness, and knowledge growth.

---

## Key Takeaways

1. **Learning emerges from simple components** - Act, Observe, Evaluate, Update. No single component learns; learning is an emergent property.

2. **Experience memory enables pattern recognition** - By storing and retrieving past experiences, agents can recognize when they've seen something similar.

3. **Strategies abstract from specific actions** - Reusable approaches are more valuable than one-time solutions.

4. **Evaluation drives improvement** - Without honest assessment, there's no signal for what to change.

5. **Different memory types serve different purposes** - Episodic, semantic, procedural, and working memory each contribute to learning.

---

## Next Steps

In [Module 13.3: Agent Orchestration & Supervision](03_orchestration_and_supervision.md), we'll explore how to coordinate multiple autonomous, self-improving agents while maintaining appropriate human oversight.

---

*"It is not the strongest of the species that survives, nor the most intelligent that survives. It is the one that is most adaptable to change."* — Often attributed to Darwin

This applies perfectly to agents: the most effective agents are those that adapt and improve.
