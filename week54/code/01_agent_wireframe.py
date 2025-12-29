"""
Agent Wireframe: The Conversational UX Agent
=============================================

This code demonstrates how to build an agent using design thinking.
Just as you'd wireframe a UI before building it, we wireframe our
agent's structure before implementing the details.

Think of this as a low-fidelity prototype—boxes and arrows translated to code.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# =============================================================================
# DESIGN TOKENS
# =============================================================================
# Just like CSS variables in a design system, these values propagate everywhere.
# Change them here, and the entire agent's behavior updates.

class DesignTokens:
    """Central configuration—your agent's design tokens."""

    # Identity
    AGENT_NAME = "Design Helper"
    COMPANY = "Creative Studio"

    # Personality (0-1 scale)
    WARMTH = 0.8          # How friendly vs. formal
    VERBOSITY = 0.5       # How detailed vs. concise
    PROACTIVITY = 0.6     # How much it offers vs. waits to be asked

    # Behavior
    MAX_SUGGESTIONS = 3   # Don't overwhelm with options
    ALWAYS_ASK_FOLLOWUP = True

    # Voice
    GREETING = f"Hey! I'm {AGENT_NAME}. What are you working on?"
    THINKING_PHRASE = "Let me think about that..."
    SUCCESS_PHRASE = "Here's what I've got:"
    CLARIFY_PHRASE = "Quick question:"


# =============================================================================
# PROMPT FRAGMENTS (Atoms)
# =============================================================================
# Reusable pieces of prompts, like components in a design system.

TONE_WARM = """
Be warm and encouraging. Use conversational language.
Celebrate good ideas. Be supportive of creative exploration.
"""

TONE_PROFESSIONAL = """
Be clear and professional. Use precise language.
Focus on actionable feedback. Respect the user's time.
"""

DESIGN_EXPERTISE = """
You have deep knowledge of:
- Visual design principles (hierarchy, balance, contrast, rhythm)
- UX patterns (progressive disclosure, error prevention, feedback)
- Design systems (components, tokens, documentation)
- Accessibility (WCAG, inclusive design)
- Typography, color theory, and layout
"""

CRITIQUE_FRAMEWORK = """
When giving feedback, use this structure:
1. Start with what's working (build confidence)
2. Identify the core issue (be specific, not vague)
3. Suggest an improvement (be actionable)
4. Explain the "why" (connect to principles)
"""


# =============================================================================
# THE AGENT
# =============================================================================
# Here's where the wireframe becomes real. Notice how the prompt structure
# mirrors the layout of a well-designed interface.

def create_design_assistant():
    """
    Create an agent that helps with design decisions.

    The prompt is structured like a page layout:
    - Identity (headline)
    - Expertise (subhead)
    - Tone (style)
    - Framework (structure)
    """

    # Choose tone based on warmth token
    tone = TONE_WARM if DesignTokens.WARMTH > 0.5 else TONE_PROFESSIONAL

    # Build the system prompt like you'd build a layout
    system_prompt = f"""
# Identity
You are {DesignTokens.AGENT_NAME}, a design assistant for {DesignTokens.COMPANY}.

# Expertise
{DESIGN_EXPERTISE}

# Voice & Tone
{tone}

# When Giving Critiques
{CRITIQUE_FRAMEWORK}

# Behavior
- Limit suggestions to {DesignTokens.MAX_SUGGESTIONS} options
- {"Always offer a follow-up question" if DesignTokens.ALWAYS_ASK_FOLLOWUP else "Wait for user to ask more"}
- When thinking, say: "{DesignTokens.THINKING_PHRASE}"
- When sharing results, say: "{DesignTokens.SUCCESS_PHRASE}"
- When you need clarity, say: "{DesignTokens.CLARIFY_PHRASE}"
"""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Create the model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Chain them together (this is our "flow")
    chain = prompt | llm | StrOutputParser()

    return chain


# =============================================================================
# INTERACTION FLOW
# =============================================================================
# This demonstrates the micro-interactions from Module 2.

def run_conversation():
    """
    Run a simple conversation demonstrating feedback design.

    Notice how each interaction has:
    - Acknowledgment (we heard you)
    - Processing (we're working on it)
    - Response (here's the result)
    - Follow-up (what's next)
    """

    assistant = create_design_assistant()

    print("\n" + "="*60)
    print(f"  {DesignTokens.AGENT_NAME}")
    print("="*60)
    print(f"\n{DesignTokens.GREETING}\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print(f"\n{DesignTokens.AGENT_NAME}: Great chatting! Good luck with your design work.\n")
            break

        if not user_input:
            continue

        # Show processing feedback (micro-interaction!)
        print(f"\n{DesignTokens.AGENT_NAME}: {DesignTokens.THINKING_PHRASE}")

        # Get response
        response = assistant.invoke({"input": user_input})

        # Display response with proper formatting
        print(f"\n{DesignTokens.AGENT_NAME}: {response}\n")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Agent Wireframe: Design Helper                              ║
    ║                                                              ║
    ║  This demonstrates:                                          ║
    ║  • Design tokens for configuration                           ║
    ║  • Prompt fragments as reusable atoms                        ║
    ║  • Structured prompts as "layouts"                           ║
    ║  • Micro-interaction patterns (feedback, acknowledgment)     ║
    ║                                                              ║
    ║  Try asking about:                                           ║
    ║  • Color palette choices                                     ║
    ║  • Layout decisions                                          ║
    ║  • Typography selection                                      ║
    ║  • UX flow improvements                                      ║
    ║                                                              ║
    ║  Type 'quit' to exit                                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    run_conversation()
