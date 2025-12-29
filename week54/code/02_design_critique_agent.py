"""
Design Critique Agent
=====================

This demonstrates how to build an agent workflow that mirrors
a design critique session.

Just as a design critique has structure (present → feedback → discuss → next steps),
this agent follows a structured workflow using LangGraph.

Think of this as designing a multi-screen user flow—but in code.
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STATE DEFINITION
# =============================================================================
# In UX, we track which screen the user is on.
# In agents, we track the "state" of the conversation.

class CritiqueState(TypedDict):
    """
    The state of our critique session.

    Just like tracking form progress:
    - What have they told us?
    - What have we produced?
    - What stage are we in?
    """
    # User inputs
    design_description: str
    design_goals: Optional[str]
    design_constraints: Optional[str]

    # Agent outputs
    initial_reaction: Optional[str]
    detailed_feedback: Optional[str]
    improvement_suggestions: Optional[str]

    # Current stage (like current screen in a flow)
    current_stage: Literal[
        "gathering",      # Getting context (onboarding)
        "reacting",       # Initial impressions (processing)
        "analyzing",      # Deep feedback (main content)
        "suggesting",     # Improvements (results)
        "wrapping_up"     # Summary (confirmation)
    ]

    # Error handling
    needs_clarification: bool
    clarification_question: Optional[str]


# =============================================================================
# NODE FUNCTIONS (Each "Screen" in the Flow)
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def gather_context(state: CritiqueState) -> CritiqueState:
    """
    NODE 1: Gather Context

    Like the first screen of an onboarding flow.
    We need to understand what we're critiquing.
    """
    print("\n📋 [Gathering Context...]")

    # Check if we have enough information
    if not state.get("design_description"):
        return {
            **state,
            "current_stage": "gathering",
            "needs_clarification": True,
            "clarification_question": "What are you working on? Describe the design you'd like feedback on."
        }

    # We have a description, move forward
    return {
        **state,
        "current_stage": "reacting",
        "needs_clarification": False
    }


def initial_reaction(state: CritiqueState) -> CritiqueState:
    """
    NODE 2: Initial Reaction

    Like a loading state with preview.
    Quick, gut-level response to build rapport.
    """
    print("\n👀 [Forming Initial Impressions...]")

    messages = [
        SystemMessage(content="""
You are a senior designer giving feedback in a critique session.
Give a brief, warm initial reaction to this design.
Keep it to 2-3 sentences. Be encouraging but honest.
Focus on your immediate impression of the overall direction.
"""),
        HumanMessage(content=f"""
Design being critiqued:
{state['design_description']}

Goals (if provided): {state.get('design_goals', 'Not specified')}
""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "initial_reaction": response.content,
        "current_stage": "analyzing"
    }


def detailed_analysis(state: CritiqueState) -> CritiqueState:
    """
    NODE 3: Detailed Analysis

    The main content screen. Deep, structured feedback.
    Uses the critique framework from Module 2.
    """
    print("\n🔍 [Analyzing in Detail...]")

    messages = [
        SystemMessage(content="""
You are a senior designer giving detailed feedback.

Structure your critique as:

## What's Working
(2-3 specific things that are effective and why)

## Areas to Explore
(2-3 specific areas that could be stronger)

## Key Question
(One thought-provoking question to push their thinking)

Be specific. Reference design principles. Be constructive.
"""),
        HumanMessage(content=f"""
Design: {state['design_description']}

Goals: {state.get('design_goals', 'Not specified')}
Constraints: {state.get('design_constraints', 'Not specified')}

Initial reaction was: {state['initial_reaction']}

Now provide detailed analysis.
""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "detailed_feedback": response.content,
        "current_stage": "suggesting"
    }


def suggest_improvements(state: CritiqueState) -> CritiqueState:
    """
    NODE 4: Suggest Improvements

    Like the results/recommendations screen.
    Actionable next steps they can take.
    """
    print("\n💡 [Generating Suggestions...]")

    messages = [
        SystemMessage(content="""
Based on the feedback given, suggest 2-3 specific, actionable improvements.

For each suggestion:
1. What to do (specific action)
2. Why it helps (connect to a principle or goal)
3. How to try it (quick way to prototype/test)

Keep suggestions focused and achievable.
"""),
        HumanMessage(content=f"""
Design: {state['design_description']}
Goals: {state.get('design_goals', 'Not specified')}

Feedback given:
{state['detailed_feedback']}

What should they try?
""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "improvement_suggestions": response.content,
        "current_stage": "wrapping_up"
    }


def wrap_up(state: CritiqueState) -> CritiqueState:
    """
    NODE 5: Wrap Up

    Like the confirmation/summary screen.
    Tie it together and offer next steps.
    """
    print("\n✅ [Wrapping Up...]")

    # This node just marks completion—the summary is printed in main
    return state


# =============================================================================
# ROUTING LOGIC (Conditional Edges)
# =============================================================================

def should_continue_gathering(state: CritiqueState) -> str:
    """
    Routing function: Do we have enough context?

    Like form validation—can we proceed, or do we need more input?
    """
    if state.get("needs_clarification"):
        return "needs_input"  # Stay on this screen
    return "has_context"      # Move to next screen


def route_after_analysis(state: CritiqueState) -> str:
    """
    Always proceed to suggestions after analysis.

    (In a more complex agent, we might branch based on severity of issues)
    """
    return "continue"


# =============================================================================
# BUILD THE WORKFLOW (The Flow Diagram as Code)
# =============================================================================

def build_critique_workflow():
    """
    Build the critique workflow graph.

    This is your user flow diagram translated to code:

    [Gather] → [React] → [Analyze] → [Suggest] → [Wrap Up]
         ↑                                              |
         └────── (if needs more info) ─────────────────┘
    """

    # Create the graph
    workflow = StateGraph(CritiqueState)

    # Add nodes (screens in the flow)
    workflow.add_node("gather_context", gather_context)
    workflow.add_node("initial_reaction", initial_reaction)
    workflow.add_node("detailed_analysis", detailed_analysis)
    workflow.add_node("suggest_improvements", suggest_improvements)
    workflow.add_node("wrap_up", wrap_up)

    # Set the entry point
    workflow.set_entry_point("gather_context")

    # Add edges (transitions between screens)
    workflow.add_conditional_edges(
        "gather_context",
        should_continue_gathering,
        {
            "needs_input": END,          # Exit to get more info (handled in main)
            "has_context": "initial_reaction"  # Proceed
        }
    )

    workflow.add_edge("initial_reaction", "detailed_analysis")
    workflow.add_edge("detailed_analysis", "suggest_improvements")
    workflow.add_edge("suggest_improvements", "wrap_up")
    workflow.add_edge("wrap_up", END)

    return workflow.compile()


# =============================================================================
# RUN THE CRITIQUE SESSION
# =============================================================================

def run_critique_session():
    """
    Run an interactive critique session.

    This demonstrates:
    - State management (tracking progress through the flow)
    - Feedback at each stage (like loading states and progress indicators)
    - Graceful handling of incomplete information
    """

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Design Critique Agent                                       ║
    ║                                                              ║
    ║  This agent conducts a structured design critique,           ║
    ║  following the same flow as a real critique session:         ║
    ║                                                              ║
    ║  1. Gather Context → 2. Initial Reaction →                   ║
    ║  3. Detailed Analysis → 4. Suggestions → 5. Wrap Up          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Build the workflow
    app = build_critique_workflow()

    # Gather initial input (progressive disclosure—one thing at a time)
    print("\n" + "="*60)
    print("  Let's start the critique session")
    print("="*60)

    print("\n📝 First, describe the design you're working on:")
    design_description = input("   > ").strip()

    print("\n🎯 What are the goals for this design? (optional, press Enter to skip)")
    design_goals = input("   > ").strip() or None

    print("\n🔒 Any constraints I should know about? (optional, press Enter to skip)")
    design_constraints = input("   > ").strip() or None

    # Create initial state
    initial_state: CritiqueState = {
        "design_description": design_description,
        "design_goals": design_goals,
        "design_constraints": design_constraints,
        "initial_reaction": None,
        "detailed_feedback": None,
        "improvement_suggestions": None,
        "current_stage": "gathering",
        "needs_clarification": False,
        "clarification_question": None
    }

    # Run the workflow
    print("\n" + "="*60)
    print("  Running Critique Session")
    print("="*60)

    final_state = app.invoke(initial_state)

    # Display results (the "confirmation screen")
    print("\n" + "="*60)
    print("  📋 CRITIQUE SUMMARY")
    print("="*60)

    print("\n## First Impression")
    print(final_state.get("initial_reaction", "N/A"))

    print("\n## Detailed Feedback")
    print(final_state.get("detailed_feedback", "N/A"))

    print("\n## Suggested Improvements")
    print(final_state.get("improvement_suggestions", "N/A"))

    print("\n" + "="*60)
    print("  Critique session complete!")
    print("="*60 + "\n")


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    run_critique_session()
