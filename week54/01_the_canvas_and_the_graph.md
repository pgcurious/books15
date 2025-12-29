# Module 1: The Canvas and The Graph

## Visual Thinking Meets Agent Architecture

*Time: 90 minutes*

---

## The Secret No One Told You

Every agent architecture diagram you'll ever see is just a user flow with different labels.

This isn't a metaphor. It's not an analogy to make you feel comfortable. It's a statement of fact that the AI engineering world hasn't fully grasped yet.

When engineers draw agent architectures, they use boxes and arrows. When you draw user flows, you use boxes and arrows. When engineers describe state transitions in a workflow, they're describing the same thing you describe when mapping a user journey. When they talk about "nodes" and "edges" in a graph, they're talking about steps and connections—the vocabulary of your daily work.

The tools are different. The jargon is different. The underlying structure is identical.

This module is about making that identity explicit, so you can bring your visual thinking—your greatest superpower—directly into agent design.

---

## Part 1: The Graph as Canvas

### What Is a Graph?

In computer science, a "graph" is a mathematical structure with two components:
- **Nodes** (vertices): Points that represent states, steps, or entities
- **Edges**: Connections between nodes that represent relationships or transitions

That's it. Nodes and edges. Points and lines.

Now consider what you already know:

| Design Artifact | Nodes Are... | Edges Are... |
|----------------|--------------|--------------|
| User Flow | Screens/States | User Actions |
| Site Map | Pages | Navigation Links |
| Journey Map | Touchpoints | User Progression |
| Service Blueprint | Activities | Flows |
| Component Hierarchy | Components | Composition |
| State Diagram | States | Transitions |

You've been drawing graphs your entire career. You just called them something else.

### LangGraph: Your New Canvas

LangGraph is the tool we use to build agent workflows. Its name tells you everything: it's a graph-based approach to constructing agent behavior.

Here's how a simple user flow maps to a LangGraph workflow:

**User Flow: Password Reset**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Enter     │────▶│   Verify    │────▶│   Reset     │────▶│   Success   │
│   Email     │     │   Identity  │     │   Password  │     │   Message   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
        │                  │
        ▼                  ▼
┌─────────────┐     ┌─────────────┐
│   Invalid   │     │   Verify    │
│   Email     │     │   Failed    │
│   Error     │     │   Error     │
└─────────────┘     └─────────────┘
```

**LangGraph Workflow: Information Gathering Agent**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Receive   │────▶│   Classify  │────▶│   Research  │────▶│   Return    │
│   Query     │     │   Intent    │     │   Answer    │     │   Response  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
        │                  │
        ▼                  ▼
┌─────────────┐     ┌─────────────┐
│   Unclear   │     │   Cannot    │
│   Query     │     │   Research  │
│   Clarify   │     │   Fallback  │
└─────────────┘     └─────────────┘
```

The structure is identical. The thinking is identical. The labels changed.

---

## Part 2: Visual Hierarchy in Agent Design

### What Gets Attention?

In visual design, hierarchy determines what the eye sees first. Through size, contrast, position, and whitespace, you guide attention through a composition. Nothing is accidental—every element has a purpose and a priority.

Agent workflows have the same requirement. Every agent faces a stream of information and must decide: What matters? What comes first? What can be ignored?

**Visual Design Hierarchy:**
```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│     ████████████████████████████████████                          │
│           MAIN HEADLINE (highest priority)                         │
│                                                                    │
│     ─────────────────────────────────────                         │
│           Supporting text (secondary)                              │
│                                                                    │
│     ○ Detail item         ○ Detail item         ○ Detail item     │
│           (tertiary)            (tertiary)            (tertiary)  │
│                                                                    │
│                                              [ BUTTON ]            │
│                                               (action)             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Agent Information Hierarchy:**
```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│     ████████████████████████████████████                          │
│           SYSTEM PROMPT (core identity/behavior)                   │
│                                                                    │
│     ─────────────────────────────────────                         │
│           Recent context (current conversation)                    │
│                                                                    │
│     ○ Retrieved fact     ○ Tool result      ○ Previous message    │
│           (supporting)        (supporting)        (supporting)    │
│                                                                    │
│                                              [ NEXT ACTION ]       │
│                                               (decision)           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

The system prompt is your headline—it sets the context for everything else. Recent context is your subheadline. Supporting details are your body copy. The decision is your call to action.

### Designing the Prompt as Layout

Consider a system prompt as a page layout. The organization matters as much as the content:

**Poor "Layout" (Cluttered, No Hierarchy):**
```
You are a helpful assistant. Always be polite. You can search the web.
You know about cooking and technology. Never give medical advice. If
someone asks about weather, use the weather tool. Be concise but thorough.
Try to include examples when helpful. You work for Acme Corp.
```

**Good "Layout" (Clear Hierarchy):**
```
# Identity
You are a culinary assistant for Acme Corp.

# Primary Behavior
- Answer cooking questions with expertise and warmth
- Use the web search tool for current information (prices, availability)
- Provide practical, actionable advice

# Constraints
- Never provide medical or nutritional health advice
- Keep responses concise; elaborate only when asked

# Voice
- Warm and encouraging, like a patient cooking teacher
- Use specific examples from common home kitchen scenarios
```

The second prompt has visual hierarchy *even in text form*. Headers create sections. Bullets create scannable lists. White space separates concerns. This is design thinking applied to a non-visual medium.

---

## Part 3: Composition and Agent Architecture

### Gestalt Principles in Agent Design

Gestalt psychology gave us principles for how humans perceive wholes from parts. These same principles illuminate how we should compose agent architectures.

#### Principle 1: Proximity

**In Design:** Elements placed close together are perceived as related.

```
┌──────────────────────────────────────────────┐
│                                              │
│   [First Name]  [Last Name]    [Email]       │
│        └──────────┬──────────┘     │         │
│         perceived as "Name"   separate       │
│                                              │
└──────────────────────────────────────────────┘
```

**In Agents:** Related functions should be grouped together. An agent that handles both user authentication and document retrieval is poorly composed—these are unrelated concerns that should live in separate agents or clearly separated modules.

```
GOOD: Agent Proximity

┌─────────────────────┐      ┌─────────────────────┐
│  Research Agent     │      │  Writing Agent      │
│  ─────────────────  │      │  ─────────────────  │
│  • Web Search       │      │  • Draft Content    │
│  • Fact Verification│──────│  • Edit Content     │
│  • Source Citation  │      │  • Format Output    │
└─────────────────────┘      └─────────────────────┘
     related tools               related tools

BAD: Mixed Proximity

┌─────────────────────────────────────────────┐
│  General Agent                              │
│  ─────────────────────────────────────────  │
│  • Web Search    • Draft Content            │
│  • Send Email    • Fact Check               │
│  • Book Calendar • Format Output            │
│  • Play Music    • Edit Content             │
└─────────────────────────────────────────────┘
     unrelated tools mixed together
```

#### Principle 2: Similarity

**In Design:** Elements that look similar are perceived as having similar function.

**In Agents:** Tools that serve similar purposes should have similar interfaces. If your web search tool returns `{"result": "..."}` but your database query returns `{"data": "..."}`, you've created unnecessary cognitive load. Consistency in tool design is like consistency in button styling.

```python
# GOOD: Similar interfaces for similar functions
def search_web(query: str) -> dict:
    """Returns {"content": str, "source": str, "confidence": float}"""

def search_database(query: str) -> dict:
    """Returns {"content": str, "source": str, "confidence": float}"""

def search_documents(query: str) -> dict:
    """Returns {"content": str, "source": str, "confidence": float}"""

# BAD: Inconsistent interfaces
def search_web(query: str) -> str:  # returns plain string

def query_db(q: str, limit: int) -> list:  # different params, returns list

def doc_search(text: str) -> dict:  # different key name, returns {"results": [...]}
```

#### Principle 3: Closure

**In Design:** We perceive complete shapes even when parts are missing.

```
   ●  ●  ●             We see a triangle,
  ●        ●           not six separate dots
 ●          ●
●            ●
```

**In Agents:** Users (and agent orchestrators) mentally "complete" agent capabilities. If your agent handles 80% of a use case, users will expect the other 20%. Design with closure in mind—either complete the shape or clearly indicate where the edges are.

```
COMPLETE CLOSURE:

┌─────────────────────────────────────────┐
│  Email Agent                            │
│  ─────────────────────────────────────  │
│  • Read emails                          │
│  • Send emails                          │
│  • Search emails                        │
│  • Delete emails                        │  ← Complete set
│  • Archive emails                       │
│  • Label/categorize emails              │
└─────────────────────────────────────────┘

INCOMPLETE (but clear about boundaries):

┌─────────────────────────────────────────┐
│  Email Reader Agent                     │
│  ─────────────────────────────────────  │
│  • Read emails                          │
│  • Search emails                        │
│  ─────────────────────────────────────  │
│  ⚠ READ-ONLY: Cannot send, delete,     │
│    or modify emails                     │  ← Clear limitation
└─────────────────────────────────────────┘
```

#### Principle 4: Figure-Ground

**In Design:** We distinguish objects (figures) from backgrounds. What you emphasize is as important as what you de-emphasize.

**In Agents:** The agent's primary function should be immediately clear. Support systems (logging, error handling, memory) should recede into the background. An agent's "figure" is its core capability; everything else is "ground."

```
┌─────────────────────────────────────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░  ┌────────────────────────────────────────────────┐  ░░░░ │
│ ░░░  │                                                │  ░░░░ │
│ ░░░  │         CUSTOMER SUPPORT AGENT                 │  ░░░░ │
│ ░░░  │      (Answer questions, resolve issues)        │  ░░░░ │
│ ░░░  │                                                │  ░░░░ │  FIGURE
│ ░░░  │                                                │  ░░░░ │
│ ░░░  └────────────────────────────────────────────────┘  ░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░  memory   │   logging   │   error handling   │   auth   ░░ │  GROUND
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Wireframing Agents

### The Agent Wireframe

Before you design high-fidelity mockups, you create wireframes. They're intentionally low-fidelity—boxes and lines, no color, no polish. They let you focus on structure before aesthetics.

Agent development needs the same discipline. Before you write code, you should wireframe your agent.

**Agent Wireframe Template:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ AGENT NAME: _____________________                                    │
│                                                                     │
│ PRIMARY FUNCTION:                                                   │
│ __________________________________________________________________ │
│                                                                     │
│ TRIGGERS (What starts this agent?):                                 │
│ ○ ______________________                                            │
│ ○ ______________________                                            │
│                                                                     │
│ TOOLS (What can it do?):                                           │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│ │                 │ │                 │ │                 │        │
│ │                 │ │                 │ │                 │        │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘        │
│                                                                     │
│ OUTPUTS (What does it produce?):                                   │
│ ○ ______________________                                            │
│ ○ ______________________                                            │
│                                                                     │
│ CONNECTS TO:                                                        │
│ ──────▶ [Other Agent/System] ○ ______________________              │
│ ◀────── [Other Agent/System] ○ ______________________              │
│                                                                     │
│ ERROR STATES:                                                       │
│ ○ If ______________, then ______________                            │
│ ○ If ______________, then ______________                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Example: Research Agent Wireframe:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ AGENT NAME: Research Assistant                                       │
│                                                                     │
│ PRIMARY FUNCTION:                                                   │
│ Gather and synthesize information from multiple sources             │
│                                                                     │
│ TRIGGERS:                                                           │
│ ○ User asks a factual question                                      │
│ ○ Another agent requests information                                │
│                                                                     │
│ TOOLS:                                                              │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│ │   Web Search    │ │ Document Store  │ │ Fact Verifier   │        │
│ │   (DuckDuckGo)  │ │   (Vector DB)   │ │  (Cross-ref)    │        │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘        │
│                                                                     │
│ OUTPUTS:                                                            │
│ ○ Synthesized answer with citations                                 │
│ ○ Confidence score (high/medium/low)                                │
│                                                                     │
│ CONNECTS TO:                                                        │
│ ──────▶ [Writing Agent] passes researched facts                    │
│ ◀────── [Supervisor] receives research requests                    │
│                                                                     │
│ ERROR STATES:                                                       │
│ ○ If no results found, return "insufficient data" + suggestions    │
│ ○ If conflicting sources, flag uncertainty + show all views        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### From Wireframe to Flow

Once you've wireframed individual agents, you can compose them into flows—just like you'd arrange screens into a user journey.

**Multi-Agent Workflow Wireframe:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          USER QUERY                                       │
│                              │                                            │
│                              ▼                                            │
│                    ┌─────────────────┐                                   │
│                    │   SUPERVISOR    │                                   │
│                    │    (Router)     │                                   │
│                    └────────┬────────┘                                   │
│               ┌─────────────┼─────────────┐                              │
│               ▼             ▼             ▼                              │
│     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                     │
│     │  RESEARCH   │ │   WRITING   │ │   CODING    │                     │
│     │    AGENT    │ │    AGENT    │ │    AGENT    │                     │
│     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                     │
│            │               │               │                              │
│            └───────────────┼───────────────┘                              │
│                            ▼                                              │
│                    ┌─────────────────┐                                   │
│                    │   SYNTHESIZER   │                                   │
│                    │     AGENT       │                                   │
│                    └────────┬────────┘                                   │
│                             ▼                                             │
│                    ┌─────────────────┐                                   │
│                    │  FINAL OUTPUT   │                                   │
│                    └─────────────────┘                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

This is a user flow. The labels say "agent" instead of "screen," but the structure is identical to what you'd create for any multi-step user journey.

---

## Part 5: Practical Exercise — Sketching an Agent

### The Brief

Design an agent that helps users plan dinner parties. The agent should:
- Suggest menus based on guest preferences and dietary restrictions
- Create shopping lists
- Provide cooking timelines
- Offer wine pairing suggestions

### Step 1: Wireframe the Agent

Before any code, sketch:

```
DINNER PARTY PLANNING AGENT — WIREFRAME

┌─────────────────────────────────────────────────────────────────────┐
│ PRIMARY FUNCTION:                                                   │
│ Help users plan complete dinner parties from menu to timeline       │
│                                                                     │
│ USER INPUTS:                                                        │
│ ○ Number of guests                                                  │
│ ○ Dietary restrictions (vegetarian, gluten-free, allergies)        │
│ ○ Cuisine preferences                                               │
│ ○ Skill level (beginner, intermediate, advanced)                    │
│ ○ Available time for preparation                                    │
│                                                                     │
│ TOOLS:                                                              │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │
│ │ Recipe     │ │ Shopping   │ │ Timeline   │ │ Wine       │        │
│ │ Database   │ │ List Gen   │ │ Calculator │ │ Pairing DB │        │
│ └────────────┘ └────────────┘ └────────────┘ └────────────┘        │
│                                                                     │
│ OUTPUTS:                                                            │
│ ○ Complete menu with recipes                                        │
│ ○ Consolidated shopping list                                        │
│ ○ Day-of cooking timeline                                           │
│ ○ Wine recommendations                                              │
│                                                                     │
│ ERROR STATES:                                                       │
│ ○ Conflicting restrictions → suggest compromises or separate dishes │
│ ○ Too ambitious for time → suggest simpler alternatives            │
│ ○ Unknown ingredients → offer substitutions                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 2: Map the User Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  [START]                                                                   │
│     │                                                                      │
│     ▼                                                                      │
│  ┌──────────────────┐                                                     │
│  │ Gather Party     │ "Tell me about your dinner party..."               │
│  │ Details          │──────────────────────────────────────────┐          │
│  └──────────────────┘                                          │          │
│     │                                                          │          │
│     │ details complete                    missing info         │          │
│     ▼                                          │               │          │
│  ┌──────────────────┐                         │               │          │
│  │ Analyze          │◀────────────────────────┘               │          │
│  │ Constraints      │                      ask follow-up       │          │
│  └──────────────────┘                                          │          │
│     │                                                          │          │
│     ▼                                                          │          │
│  ┌──────────────────┐     conflicts?     ┌──────────────────┐ │          │
│  │ Generate Menu    │───────────────────▶│ Resolve          │ │          │
│  │ Options          │◀───────────────────│ Conflicts        │ │          │
│  └──────────────────┘    resolved        └──────────────────┘ │          │
│     │                                                          │          │
│     │ user selects menu                                        │          │
│     ▼                                                          │          │
│  ┌──────────────────┐                                          │          │
│  │ Create Shopping  │                                          │          │
│  │ List             │                                          │          │
│  └──────────────────┘                                          │          │
│     │                                                          │          │
│     ▼                                                          │          │
│  ┌──────────────────┐                                          │          │
│  │ Generate         │                                          │          │
│  │ Timeline         │                                          │          │
│  └──────────────────┘                                          │          │
│     │                                                          │          │
│     ▼                                                          │          │
│  ┌──────────────────┐                                          │          │
│  │ Suggest Wine     │                                          │          │
│  │ Pairings         │                                          │          │
│  └──────────────────┘                                          │          │
│     │                                                          │          │
│     ▼                                                          │          │
│  [COMPLETE PLAN DELIVERED]                                                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Step 3: Translate to Code Structure

Now the code almost writes itself. The visual structure becomes the code structure:

```python
from langgraph.graph import StateGraph, END

# The states from your flow diagram become nodes
workflow = StateGraph(PartyPlannerState)

# Add nodes (boxes in your flow)
workflow.add_node("gather_details", gather_party_details)
workflow.add_node("analyze_constraints", analyze_constraints)
workflow.add_node("generate_menu", generate_menu_options)
workflow.add_node("resolve_conflicts", resolve_dietary_conflicts)
workflow.add_node("create_shopping_list", create_shopping_list)
workflow.add_node("generate_timeline", generate_cooking_timeline)
workflow.add_node("suggest_wine", suggest_wine_pairings)

# Add edges (arrows in your flow)
workflow.add_edge("gather_details", "analyze_constraints")
workflow.add_conditional_edges(
    "analyze_constraints",
    check_info_complete,
    {
        "complete": "generate_menu",
        "incomplete": "gather_details"  # Loop back for more info
    }
)
workflow.add_conditional_edges(
    "generate_menu",
    check_conflicts,
    {
        "conflicts": "resolve_conflicts",
        "clear": "create_shopping_list"
    }
)
workflow.add_edge("resolve_conflicts", "generate_menu")  # Try again
workflow.add_edge("create_shopping_list", "generate_timeline")
workflow.add_edge("generate_timeline", "suggest_wine")
workflow.add_edge("suggest_wine", END)
```

Notice how the code mirrors the visual structure exactly. Every box becomes a node. Every arrow becomes an edge. Conditional branches become `add_conditional_edges`.

---

## Part 6: The Designer's Debugging Eye

### Visual Debugging

When an agent misbehaves, engineers often dive into logs and traces. You have another tool: your trained eye for visual problems.

**Common Visual Antipatterns in Agent Architecture:**

#### Antipattern 1: The Spaghetti Graph
```
BAD:
                 ┌───────────────────┐
        ┌───────▶│       A           │◀────────┐
        │        └─────────┬─────────┘         │
        │                  │                    │
        │        ┌─────────▼─────────┐         │
        │   ┌───▶│       B           │◀───┐    │
        │   │    └─────────┬─────────┘    │    │
        │   │              │              │    │
        │   │    ┌─────────▼─────────┐    │    │
        └───┼───▶│       C           │────┼────┘
            │    └─────────┬─────────┘    │
            │              │              │
            └──────────────┴──────────────┘
```

When you see a graph that looks like this, your design instincts should scream "refactor!" Just as you'd simplify a cluttered interface, you need to simplify a cluttered agent graph.

**Solution:** Identify the actual flow, remove unnecessary connections, introduce intermediate states.

#### Antipattern 2: The Dead End

```
BAD:
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   Start  │────▶│  Process │────▶│   End    │
    └──────────┘     └──────────┘     └──────────┘
                           │
                           ▼
                     ┌──────────┐
                     │  Error   │     ← Where does this go?
                     │  State   │       No exit path!
                     └──────────┘
```

Every user flow needs to handle error states with clear exit paths. Agent workflows are no different.

**Solution:** Every node needs an edge out. Error states should recover or escalate, never dead-end.

#### Antipattern 3: The God Node

```
BAD:
                     ┌────────────────────────────────┐
    ───────────────▶ │                                │
    ───────────────▶ │                                │
    ───────────────▶ │     MAIN PROCESSOR             │ ──────────────▶
    ───────────────▶ │     (does everything)          │ ──────────────▶
    ───────────────▶ │                                │ ──────────────▶
                     └────────────────────────────────┘
```

Just as a single screen shouldn't do everything, a single node shouldn't do everything. This is the equivalent of a design with no clear information architecture.

**Solution:** Break the god node into focused, single-purpose nodes with clear responsibilities.

---

## Synthesis: You've Been Doing This All Along

The core insight of this module is that agent architecture is visual design applied to intelligence. The skills you've developed—seeing structure, creating hierarchy, composing elements, designing flows—are exactly what's needed to build effective AI agents.

The difference is medium, not mindset.

When you look at an agent architecture diagram, don't see something foreign. See a user flow with different labels. See a design system with different components. See an interface with different interactions.

Then apply everything you know.

---

## Key Takeaways

1. **Agent graphs are user flows** — Nodes are states, edges are transitions. You've drawn thousands of these.

2. **Visual hierarchy applies to prompts** — Structure your system prompts like you'd structure a page layout. Hierarchy, whitespace, and progressive disclosure all apply.

3. **Gestalt principles illuminate composition** — Proximity, similarity, closure, and figure-ground all apply to how you compose agents and their tools.

4. **Wireframe before you code** — Just as you wouldn't jump to high-fidelity mockups, don't jump to code. Sketch your agent's structure first.

5. **Your debugging eye works here** — Visual antipatterns (spaghetti, dead ends, god nodes) are as recognizable in agent graphs as in UI designs.

---

## Practice Exercise

Take a complex user flow you've designed in the past—an onboarding sequence, a checkout process, a multi-step form. Translate it into an agent wireframe using the template provided. Identify:

- What would be nodes (states)?
- What would be edges (transitions)?
- What are the error states and how would they resolve?
- What tools would the agent need at each step?

You'll find that the translation is surprisingly natural.

---

## Next Module

In Module 2, we'll move from architecture to behavior—translating interaction design principles into agent workflows. You'll learn how the micro-interactions you design every day map directly to agent behaviors.

---

*"The details are not the details. They make the design."*
— Charles Eames

*In agent design, as in all design, the details make the intelligence.*
