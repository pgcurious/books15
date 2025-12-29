"""
Multi-Agent Design Team
=======================

This demonstrates a multi-agent system modeled after a real design team.

Just as a design team has specialists (researcher, UX designer, visual designer,
copywriter) who collaborate on a project, this system has specialized agents
that work together.

This is design systems thinking applied to AI: composable agents with clear
responsibilities, consistent interfaces, and coordinated handoffs.
"""

from typing import TypedDict, Literal, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# DESIGN TOKENS (Shared Configuration)
# =============================================================================

class TeamTokens:
    """
    Shared configuration for all agents on the team.
    Like design tokens, these ensure consistency across the system.
    """

    # Team identity
    TEAM_NAME = "Creative AI Team"
    PROJECT_STYLE = "modern, clean, user-focused"

    # Shared voice
    FORMALITY = 0.6  # 0=casual, 1=formal
    DETAIL_LEVEL = 0.7  # 0=brief, 1=comprehensive

    # Collaboration rules
    MAX_ITERATIONS = 3
    REQUIRE_CONSENSUS = True

    # Standard phrases (ensuring consistent voice)
    HANDOFF_PHRASE = "Passing this to {next_agent} for {reason}."
    BUILDING_ON = "Building on {prev_agent}'s work..."
    COMPLETE_PHRASE = "Here's the final deliverable:"


# =============================================================================
# SHARED STATE (The Project Brief)
# =============================================================================

class ProjectState(TypedDict):
    """
    Shared project state—like a shared Figma file or design brief.
    All agents can read from and write to this.
    """
    # The original brief
    project_brief: str
    target_audience: Optional[str]
    project_constraints: Optional[str]

    # Agent contributions (each adds their part)
    research_findings: Optional[str]
    ux_recommendations: Optional[str]
    visual_direction: Optional[str]
    copy_suggestions: Optional[str]

    # Coordination
    current_agent: str
    completed_agents: List[str]
    final_synthesis: Optional[str]


# =============================================================================
# PROMPT FRAGMENTS (Atomic Design for Prompts)
# =============================================================================

# These are reusable prompt atoms—like components in a design system

COLLABORATION_GUIDELINES = """
You are part of a collaborative design team. When working:
- Build on the work of other team members
- Don't repeat what they've already covered
- Add your unique expertise
- Keep your contribution focused and actionable
"""

HANDOFF_FORMAT = """
Structure your output for the next team member:
1. Key findings/recommendations (bullet points)
2. Things to consider (relevant to their specialty)
3. Open questions (if any)
"""


# =============================================================================
# AGENT DEFINITIONS (The Team Members)
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def researcher_agent(state: ProjectState) -> ProjectState:
    """
    THE RESEARCHER
    ==============
    Like a UX researcher who gathers context before design begins.

    Responsibilities:
    - Understand the problem space
    - Identify user needs and pain points
    - Surface relevant patterns and precedents
    """
    print("\n🔍 [Researcher] Analyzing the problem space...")

    messages = [
        SystemMessage(content=f"""
You are the Research Specialist on {TeamTokens.TEAM_NAME}.

{COLLABORATION_GUIDELINES}

Your expertise:
- User research and need identification
- Competitive analysis and pattern recognition
- Problem framing and opportunity identification

Project style: {TeamTokens.PROJECT_STYLE}

For this project, provide:
1. **Problem Analysis**: What's the core challenge?
2. **User Considerations**: Who needs this and why?
3. **Precedents**: What patterns or examples are relevant?
4. **Key Insight**: One crucial finding for the team

{HANDOFF_FORMAT}
"""),
        HumanMessage(content=f"""
Project Brief: {state['project_brief']}
Target Audience: {state.get('target_audience', 'Not specified')}
Constraints: {state.get('project_constraints', 'Not specified')}

Conduct initial research for this project.
""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "research_findings": response.content,
        "current_agent": "ux_designer",
        "completed_agents": state.get("completed_agents", []) + ["researcher"]
    }


def ux_designer_agent(state: ProjectState) -> ProjectState:
    """
    THE UX DESIGNER
    ===============
    Translates research into structure and flow.

    Responsibilities:
    - Define information architecture
    - Design user flows and interactions
    - Ensure usability and accessibility
    """
    print("\n📐 [UX Designer] Designing the experience structure...")

    messages = [
        SystemMessage(content=f"""
You are the UX Designer on {TeamTokens.TEAM_NAME}.

{COLLABORATION_GUIDELINES}

{TeamTokens.BUILDING_ON.format(prev_agent="the Researcher")}

Your expertise:
- Information architecture and navigation
- User flows and interaction patterns
- Usability and accessibility
- Progressive disclosure and error handling

Project style: {TeamTokens.PROJECT_STYLE}

For this project, provide:
1. **Structure**: How should this be organized?
2. **Key Flows**: What are the critical user paths?
3. **Interactions**: What patterns would work well?
4. **Accessibility**: Important considerations

{HANDOFF_FORMAT}
"""),
        HumanMessage(content=f"""
Project Brief: {state['project_brief']}

Research findings from our researcher:
{state.get('research_findings', 'No research yet')}

Design the UX structure for this project.
""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "ux_recommendations": response.content,
        "current_agent": "visual_designer",
        "completed_agents": state.get("completed_agents", []) + ["ux_designer"]
    }


def visual_designer_agent(state: ProjectState) -> ProjectState:
    """
    THE VISUAL DESIGNER
    ==================
    Defines the aesthetic direction and visual language.

    Responsibilities:
    - Establish visual hierarchy and emphasis
    - Define color, typography, and spatial systems
    - Create mood and emotional resonance
    """
    print("\n🎨 [Visual Designer] Developing visual direction...")

    messages = [
        SystemMessage(content=f"""
You are the Visual Designer on {TeamTokens.TEAM_NAME}.

{COLLABORATION_GUIDELINES}

{TeamTokens.BUILDING_ON.format(prev_agent="the UX Designer")}

Your expertise:
- Visual hierarchy and emphasis
- Color theory and palette design
- Typography selection and systems
- Layout, spacing, and composition
- Motion and micro-interaction aesthetics

Project style: {TeamTokens.PROJECT_STYLE}

For this project, provide:
1. **Visual Hierarchy**: What needs emphasis?
2. **Color Direction**: What palette and mood?
3. **Typography**: What type system would work?
4. **Spatial System**: Layout and rhythm recommendations

{HANDOFF_FORMAT}
"""),
        HumanMessage(content=f"""
Project Brief: {state['project_brief']}

Research findings:
{state.get('research_findings', 'No research yet')}

UX recommendations:
{state.get('ux_recommendations', 'No UX yet')}

Develop the visual direction for this project.
""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "visual_direction": response.content,
        "current_agent": "copywriter",
        "completed_agents": state.get("completed_agents", []) + ["visual_designer"]
    }


def copywriter_agent(state: ProjectState) -> ProjectState:
    """
    THE COPYWRITER
    ==============
    Crafts the language and messaging.

    Responsibilities:
    - Define voice and tone
    - Write key messages and microcopy
    - Ensure clarity and consistency
    """
    print("\n✍️  [Copywriter] Crafting the messaging...")

    messages = [
        SystemMessage(content=f"""
You are the Copywriter on {TeamTokens.TEAM_NAME}.

{COLLABORATION_GUIDELINES}

{TeamTokens.BUILDING_ON.format(prev_agent="the Visual Designer")}

Your expertise:
- Voice and tone development
- Headline and message hierarchy
- Microcopy (buttons, labels, feedback)
- Clarity and conciseness
- Emotional resonance through words

Project style: {TeamTokens.PROJECT_STYLE}

For this project, provide:
1. **Voice**: How should this sound?
2. **Key Messages**: Core headlines or statements
3. **Microcopy Examples**: Buttons, labels, feedback messages
4. **Tone Variations**: How does tone shift in different contexts?

{HANDOFF_FORMAT}
"""),
        HumanMessage(content=f"""
Project Brief: {state['project_brief']}
Target Audience: {state.get('target_audience', 'Not specified')}

Research findings:
{state.get('research_findings', 'No research yet')}

UX recommendations:
{state.get('ux_recommendations', 'No UX yet')}

Visual direction:
{state.get('visual_direction', 'No visual yet')}

Develop the messaging for this project.
""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "copy_suggestions": response.content,
        "current_agent": "synthesizer",
        "completed_agents": state.get("completed_agents", []) + ["copywriter"]
    }


def synthesizer_agent(state: ProjectState) -> ProjectState:
    """
    THE SYNTHESIZER (Creative Director)
    ====================================
    Brings all contributions together into a cohesive vision.

    Like a creative director reviewing the team's work
    and synthesizing it into a unified direction.
    """
    print("\n🎯 [Creative Director] Synthesizing the vision...")

    messages = [
        SystemMessage(content=f"""
You are the Creative Director synthesizing {TeamTokens.TEAM_NAME}'s work.

Your job is to:
1. Identify the strongest ideas from each team member
2. Resolve any conflicts or inconsistencies
3. Create a unified creative direction
4. Provide a clear, actionable summary

{TeamTokens.COMPLETE_PHRASE}

Structure your synthesis as:
## Creative Direction Summary
(2-3 sentences capturing the unified vision)

## Key Decisions
(The most important choices, based on team input)

## Next Steps
(3-5 specific actions to move forward)
"""),
        HumanMessage(content=f"""
Project Brief: {state['project_brief']}

TEAM CONTRIBUTIONS:

**Research Findings:**
{state.get('research_findings', 'None')}

**UX Recommendations:**
{state.get('ux_recommendations', 'None')}

**Visual Direction:**
{state.get('visual_direction', 'None')}

**Copy Suggestions:**
{state.get('copy_suggestions', 'None')}

Synthesize these into a unified creative direction.
""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "final_synthesis": response.content,
        "current_agent": "complete",
        "completed_agents": state.get("completed_agents", []) + ["synthesizer"]
    }


# =============================================================================
# BUILD THE TEAM WORKFLOW
# =============================================================================

def build_design_team():
    """
    Build the multi-agent team workflow.

    This is the pipeline pattern from Module 3:
    Research → UX → Visual → Copy → Synthesis

    Each agent builds on the previous one's work,
    just like a real design process.
    """

    workflow = StateGraph(ProjectState)

    # Add team members (nodes)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("ux_designer", ux_designer_agent)
    workflow.add_node("visual_designer", visual_designer_agent)
    workflow.add_node("copywriter", copywriter_agent)
    workflow.add_node("synthesizer", synthesizer_agent)

    # Define the collaboration flow (edges)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "ux_designer")
    workflow.add_edge("ux_designer", "visual_designer")
    workflow.add_edge("visual_designer", "copywriter")
    workflow.add_edge("copywriter", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# =============================================================================
# RUN A TEAM PROJECT
# =============================================================================

def run_team_project():
    """
    Run a full team design session.

    Watch as each team member contributes their expertise,
    building on each other's work toward a unified vision.
    """

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Multi-Agent Design Team                                     ║
    ║                                                              ║
    ║  This system simulates a design team working together:       ║
    ║                                                              ║
    ║  🔍 Researcher → 📐 UX Designer → 🎨 Visual Designer →       ║
    ║  ✍️  Copywriter → 🎯 Creative Director                       ║
    ║                                                              ║
    ║  Each agent builds on the previous one's work,               ║
    ║  just like a real collaborative design process.              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Gather the project brief
    print("\n" + "="*60)
    print("  PROJECT BRIEF")
    print("="*60)

    print("\n📋 What's the project? (e.g., 'A mobile app for tracking habits')")
    project_brief = input("   > ").strip()

    print("\n👥 Who's the target audience? (optional)")
    target_audience = input("   > ").strip() or None

    print("\n🔒 Any constraints? (optional)")
    constraints = input("   > ").strip() or None

    # Create initial state
    initial_state: ProjectState = {
        "project_brief": project_brief,
        "target_audience": target_audience,
        "project_constraints": constraints,
        "research_findings": None,
        "ux_recommendations": None,
        "visual_direction": None,
        "copy_suggestions": None,
        "current_agent": "researcher",
        "completed_agents": [],
        "final_synthesis": None
    }

    # Run the team workflow
    print("\n" + "="*60)
    print("  TEAM IN ACTION")
    print("="*60)

    team = build_design_team()
    final_state = team.invoke(initial_state)

    # Display final synthesis
    print("\n" + "="*60)
    print("  📋 FINAL CREATIVE DIRECTION")
    print("="*60)
    print(f"\n{final_state.get('final_synthesis', 'No synthesis generated')}")

    # Show team contributions summary
    print("\n" + "="*60)
    print("  TEAM CONTRIBUTIONS")
    print("="*60)

    print("\n🔍 RESEARCHER:")
    print("-" * 40)
    print(final_state.get('research_findings', 'None')[:500] + "..." if len(final_state.get('research_findings', '')) > 500 else final_state.get('research_findings', 'None'))

    print("\n📐 UX DESIGNER:")
    print("-" * 40)
    print(final_state.get('ux_recommendations', 'None')[:500] + "..." if len(final_state.get('ux_recommendations', '')) > 500 else final_state.get('ux_recommendations', 'None'))

    print("\n🎨 VISUAL DESIGNER:")
    print("-" * 40)
    print(final_state.get('visual_direction', 'None')[:500] + "..." if len(final_state.get('visual_direction', '')) > 500 else final_state.get('visual_direction', 'None'))

    print("\n✍️  COPYWRITER:")
    print("-" * 40)
    print(final_state.get('copy_suggestions', 'None')[:500] + "..." if len(final_state.get('copy_suggestions', '')) > 500 else final_state.get('copy_suggestions', 'None'))

    print("\n" + "="*60)
    print("  Project complete!")
    print("="*60 + "\n")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    run_team_project()
