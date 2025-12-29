# Module 3: Design Systems for Agent Systems

## Composability, Scale, and Coherence

*Time: 90 minutes*

---

## The Unseen Architecture

You've built design systems. You know the painstaking work of creating something that enables a hundred designers to produce coherent output. The component libraries. The design tokens. The documentation. The governance.

You also know the alternative: chaos. Fifty shades of blue. Inconsistent spacing. Buttons that behave differently on every screen. Products that feel like they were designed by a hundred people who never talked to each other—because they were.

AI agents are headed toward the same chaos.

Organizations are building agents in silos. One team creates a customer service agent. Another builds a sales assistant. A third develops an internal knowledge bot. Each has different personalities, different error messages, different ways of asking for information. Users encounter these agents and feel the same jarring inconsistency they feel using a product without a design system.

This module teaches you to apply design system thinking to AI agents—creating coherent, composable, scalable agent ecosystems.

---

## Part 1: Atomic Design for Agents

### Brad Frost's Hierarchy, Translated

Brad Frost's Atomic Design methodology describes how interfaces are composed:

```
ATOMIC DESIGN HIERARCHY

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ATOMS          The basic building blocks                          │
│  ○              (buttons, inputs, labels)                          │
│                                                                     │
│  MOLECULES      Simple combinations of atoms                       │
│  ○─○            (search form, card header)                         │
│                                                                     │
│  ORGANISMS      Complex combinations of molecules                  │
│  ┌───┐          (navigation, hero section, footer)                 │
│  │○─○│                                                             │
│  └───┘                                                              │
│                                                                     │
│  TEMPLATES      Page layouts without real content                  │
│  ┌─────────┐    (blog post template, dashboard template)           │
│  │ ┌───┐   │                                                       │
│  │ │○─○│   │                                                       │
│  │ └───┘   │                                                       │
│  └─────────┘                                                        │
│                                                                     │
│  PAGES          Templates filled with real content                 │
│  ┌─────────┐    (specific instances of templates)                  │
│  │ ARTICLE │                                                       │
│  └─────────┘                                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

This hierarchy applies directly to agent systems:

```
ATOMIC AGENT DESIGN

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ATOMS          Core capabilities                                  │
│  ○              (tools, prompts, memory stores)                    │
│                                                                     │
│  MOLECULES      Combined capabilities                              │
│  ○─○            (search-then-summarize, validate-then-store)       │
│                                                                     │
│  ORGANISMS      Single-purpose agents                              │
│  ┌───┐          (research agent, writing agent, coding agent)      │
│  │○─○│                                                             │
│  └───┘                                                              │
│                                                                     │
│  TEMPLATES      Agent workflow patterns                            │
│  ┌─────────┐    (supervisor pattern, pipeline pattern)             │
│  │ ┌───┐   │                                                       │
│  │ │○─○│   │                                                       │
│  │ └───┘   │                                                       │
│  └─────────┘                                                        │
│                                                                     │
│  PAGES          Complete agent systems                             │
│  ┌─────────┐    (customer support system, content workflow)        │
│  │ SYSTEM  │                                                       │
│  └─────────┘                                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Level 1: Atoms — Tools and Prompts

Atoms are the irreducible building blocks. In agent systems, these are:

**Tools** — Single-purpose functions the agent can call:
```python
# ATOM: A single tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Returns search results

def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email."""
    # Returns success/failure

def get_weather(location: str) -> dict:
    """Get current weather for a location."""
    # Returns weather data
```

**Prompt Fragments** — Reusable pieces of system prompts:
```python
# ATOM: Reusable prompt fragments

TONE_PROFESSIONAL = """
Communicate in a professional, clear manner.
Use complete sentences. Avoid slang or overly casual language.
"""

TONE_FRIENDLY = """
Be warm and approachable. Use conversational language.
Feel free to use light humor when appropriate.
"""

ERROR_HANDLING_STANDARD = """
When you encounter errors:
1. Acknowledge the issue directly
2. Explain what happened in plain language
3. Suggest an alternative or next step
Never blame the user or use technical jargon.
"""

SAFETY_STANDARD = """
Never provide:
- Medical diagnoses or treatment recommendations
- Legal advice for specific situations
- Financial advice for specific investments
Always recommend consulting qualified professionals.
"""
```

**Memory Stores** — Data structures for persistence:
```python
# ATOM: Memory components

class ConversationMemory:
    """Stores recent conversation history."""

class UserPreferences:
    """Stores learned user preferences."""

class FactStore:
    """Stores verified facts for reference."""
```

### Level 2: Molecules — Combined Capabilities

Molecules combine atoms into useful combinations:

```python
# MOLECULE: Search-then-summarize

class ResearchCapability:
    """Combines search, evaluation, and summarization."""

    def __init__(self):
        self.search = search_web          # ATOM
        self.evaluate = evaluate_source   # ATOM
        self.summarize = summarize_text   # ATOM

    def research(self, query: str) -> dict:
        results = self.search(query)
        evaluated = [self.evaluate(r) for r in results]
        reliable = [r for r in evaluated if r.reliability > 0.7]
        summary = self.summarize(reliable)
        return {
            "summary": summary,
            "sources": reliable
        }
```

```python
# MOLECULE: Validate-then-store

class DataIntakeCapability:
    """Combines validation, transformation, and storage."""

    def __init__(self):
        self.validate = validate_input    # ATOM
        self.transform = normalize_data   # ATOM
        self.store = save_to_database     # ATOM

    def intake(self, data: dict) -> dict:
        validation = self.validate(data)
        if not validation.is_valid:
            return {"error": validation.errors}
        normalized = self.transform(data)
        record_id = self.store(normalized)
        return {"success": True, "id": record_id}
```

### Level 3: Organisms — Single-Purpose Agents

Organisms are complete, single-purpose agents composed of molecules:

```python
# ORGANISM: Research Agent

class ResearchAgent:
    """A complete agent specialized in research tasks."""

    def __init__(self):
        # MOLECULES
        self.research = ResearchCapability()
        self.citation = CitationCapability()

        # ATOMS (prompt configuration)
        self.system_prompt = f"""
        You are a research specialist.
        {TONE_PROFESSIONAL}
        {SAFETY_STANDARD}

        Your job is to find accurate information and cite sources.
        Always indicate confidence levels in your findings.
        """

    def run(self, query: str) -> str:
        findings = self.research.research(query)
        formatted = self.citation.format(findings)
        return formatted
```

```python
# ORGANISM: Writing Agent

class WritingAgent:
    """A complete agent specialized in content creation."""

    def __init__(self):
        # MOLECULES
        self.drafting = DraftingCapability()
        self.editing = EditingCapability()

        # ATOMS (prompt configuration)
        self.system_prompt = f"""
        You are a writing specialist.
        {TONE_PROFESSIONAL}

        Your job is to create clear, engaging content.
        Adapt your style to the requested format and audience.
        """
```

### Level 4: Templates — Workflow Patterns

Templates are reusable patterns for combining organisms:

```python
# TEMPLATE: Supervisor Pattern

class SupervisorWorkflow:
    """
    A template where a supervisor agent routes tasks
    to specialized worker agents.
    """

    def __init__(self, workers: list[Agent]):
        self.supervisor = SupervisorAgent()
        self.workers = {w.name: w for w in workers}

    def run(self, task: str) -> str:
        # Supervisor decides which worker(s) to use
        plan = self.supervisor.plan(task)

        results = []
        for step in plan:
            worker = self.workers[step.worker]
            result = worker.run(step.subtask)
            results.append(result)

        # Supervisor synthesizes results
        return self.supervisor.synthesize(results)
```

```python
# TEMPLATE: Pipeline Pattern

class PipelineWorkflow:
    """
    A template where agents process sequentially,
    each building on the previous output.
    """

    def __init__(self, stages: list[Agent]):
        self.stages = stages

    def run(self, input: str) -> str:
        result = input
        for stage in self.stages:
            result = stage.run(result)
        return result
```

### Level 5: Pages — Complete Systems

Pages are complete agent systems for specific use cases:

```python
# PAGE: Content Creation System

class ContentCreationSystem:
    """
    Complete system for creating marketing content.
    Uses the supervisor pattern with specialized agents.
    """

    def __init__(self):
        # ORGANISMS
        research_agent = ResearchAgent()
        writing_agent = WritingAgent()
        editing_agent = EditingAgent()
        seo_agent = SEOAgent()

        # TEMPLATE
        self.workflow = SupervisorWorkflow([
            research_agent,
            writing_agent,
            editing_agent,
            seo_agent
        ])

    def create_content(self, brief: str) -> dict:
        return self.workflow.run(brief)
```

---

## Part 2: Design Tokens for Agents

### What Are Design Tokens?

Design tokens are the abstract values that propagate through a design system:

```css
/* DESIGN TOKENS (Traditional) */
--color-primary: #0066CC;
--color-secondary: #00AA44;
--spacing-unit: 8px;
--font-size-body: 16px;
--border-radius-default: 4px;
--transition-duration: 200ms;
```

These tokens create consistency. Change `--color-primary` once, and every button, link, and highlight updates. Tokens are the single source of truth.

### Agent Configuration Tokens

Agent systems need the same abstraction layer:

```python
# AGENT TOKENS

class AgentTokens:
    """
    Central configuration that propagates through all agents.
    Change once here, applies everywhere.
    """

    # Identity
    COMPANY_NAME = "Acme Corp"
    PRODUCT_NAME = "Acme Assistant"

    # Personality
    TONE = "professional"  # or "casual", "formal"
    WARMTH = 0.7  # 0 = cold, 1 = warm
    VERBOSITY = 0.5  # 0 = terse, 1 = verbose

    # Behavior
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30
    CONFIDENCE_THRESHOLD = 0.7

    # Safety
    ALLOW_EXTERNAL_LINKS = True
    ALLOW_FILE_UPLOADS = False
    REQUIRE_CONFIRMATION_FOR = ["delete", "send", "purchase"]

    # Limits
    MAX_TOKENS_PER_RESPONSE = 500
    MAX_TOOLS_PER_TURN = 5
    CONVERSATION_MEMORY_TURNS = 20

    # Contact points
    ESCALATION_CONTACT = "support@acme.com"
    FEEDBACK_URL = "https://acme.com/feedback"
```

These tokens create consistency across all agents:

```python
# Using tokens in agent configuration

def build_system_prompt(agent_type: str) -> str:
    """Generate a system prompt using tokens."""

    base = f"""
    You are {AgentTokens.PRODUCT_NAME}, an assistant for {AgentTokens.COMPANY_NAME}.
    """

    if AgentTokens.TONE == "professional":
        base += TONE_PROFESSIONAL
    elif AgentTokens.TONE == "casual":
        base += TONE_CASUAL

    base += f"""
    If you can't help, direct users to {AgentTokens.ESCALATION_CONTACT}.
    """

    return base
```

### Token Categories

Design systems organize tokens by category. Agent systems should too:

```
AGENT TOKEN CATEGORIES

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  IDENTITY TOKENS                                                   │
│  Names, branding, legal requirements                               │
│  ─────────────────────────────────────────────────────────────────│
│  • COMPANY_NAME          • PRODUCT_NAME                            │
│  • LEGAL_DISCLAIMER      • COPYRIGHT_NOTICE                        │
│                                                                     │
│  PERSONALITY TOKENS                                                │
│  Voice, tone, communication style                                  │
│  ─────────────────────────────────────────────────────────────────│
│  • TONE                  • FORMALITY                               │
│  • WARMTH                • HUMOR_LEVEL                             │
│  • VERBOSITY             • EMOJI_USAGE                             │
│                                                                     │
│  BEHAVIOR TOKENS                                                   │
│  How agents act and respond                                        │
│  ─────────────────────────────────────────────────────────────────│
│  • CONFIDENCE_THRESHOLD  • MAX_RETRIES                             │
│  • TIMEOUT_SECONDS       • PROACTIVITY_LEVEL                       │
│                                                                     │
│  SAFETY TOKENS                                                     │
│  Guardrails and restrictions                                       │
│  ─────────────────────────────────────────────────────────────────│
│  • BLOCKED_TOPICS        • REQUIRE_CONFIRMATION                    │
│  • ALLOWED_ACTIONS       • FORBIDDEN_ACTIONS                       │
│                                                                     │
│  LIMIT TOKENS                                                      │
│  Resource constraints                                              │
│  ─────────────────────────────────────────────────────────────────│
│  • MAX_TOKENS            • RATE_LIMITS                             │
│  • MEMORY_SIZE           • TOOL_LIMITS                             │
│                                                                     │
│  ESCALATION TOKENS                                                 │
│  Handoff and fallback configuration                                │
│  ─────────────────────────────────────────────────────────────────│
│  • HUMAN_HANDOFF_CONTACT • FALLBACK_BEHAVIOR                       │
│  • ESCALATION_TRIGGERS   • FEEDBACK_CHANNELS                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: The Agent Style Guide

### Why Agents Need Style Guides

Every design system has a style guide—documentation of voice, tone, and usage patterns. Agent systems need the same.

Without it, you get:
- Customer service agent: "Hi! How can I help you today? 😊"
- Sales agent: "Greetings. I am prepared to assist with your inquiry."
- IT support agent: "yo whats broken"

Three agents, three personalities, one confused user.

### Agent Style Guide Template

```markdown
# [Company] Agent Style Guide

## Voice Principles

Our agents embody these consistent traits:

1. **Helpful without being pushy**
   - Offer assistance, don't insist
   - Good: "I can help with that if you'd like."
   - Bad: "Let me help you with that!"

2. **Knowledgeable without being arrogant**
   - Share expertise with humility
   - Good: "Based on our data, this usually works best..."
   - Bad: "The correct answer is..."

3. **Warm without being unprofessional**
   - Be human, but stay appropriate
   - Good: "Great question!"
   - Bad: "OMG that's such a good question lol"

## Tone Adaptation

### By Situation

| Situation | Tone | Example |
|-----------|------|---------|
| Welcome | Warm, inviting | "Hi! I'm here to help." |
| Success | Affirming | "Done! Your order is confirmed." |
| Error | Helpful, calm | "That didn't work, but here's what we can try..." |
| Waiting | Informative | "Searching now—this takes about 10 seconds..." |
| Confusion | Patient | "No problem—let me clarify..." |
| Escalation | Professional | "I'm connecting you with a specialist who can help." |

### By User State

| User Seems... | Adjust To... |
|---------------|--------------|
| Frustrated | Calmer, more empathetic |
| Confused | Simpler language, more examples |
| Expert | More technical, less explanation |
| In a hurry | More concise, action-focused |

## Standard Phrases

### Opening
- "Hi! I'm [Agent]. How can I help?"
- "Welcome back, [Name]. What can I do for you?"

### Acknowledgment
- "Got it."
- "I understand."
- "Makes sense."

### Working
- "Let me look into that..."
- "One moment while I check..."
- "Working on it..."

### Success
- "Done!"
- "All set."
- "Here's what I found:"

### Can't Help
- "I'm not able to do that, but here's what I can do..."
- "That's outside what I'm trained for. Let me connect you with someone who can help."

### Closing
- "Is there anything else I can help with?"
- "Glad I could help!"

## Forbidden Phrases

Never use:
- "I cannot..."
- "That is not possible..."
- "You must..."
- "As an AI..."
- "I don't have feelings but..."
- Technical error codes without explanation
- Blaming the user for errors

## Formatting Standards

### Lists
- Use bullet points for 3+ items
- Use numbered lists only for sequential steps

### Length
- Keep responses under 150 words unless user asks for more
- Break long responses into sections with headers

### Code/Technical
- Always format code in code blocks
- Explain technical terms on first use
```

---

## Part 4: Multi-Agent Team Design

### Design as Organizational Design

Building a multi-agent system is essentially organizational design. You're creating a team—with roles, responsibilities, handoffs, and communication channels.

Your experience thinking about how design teams work gives you insight into how agent teams should work.

### Team Patterns

**Pattern 1: The Supervisor Model**

Like a design director who assigns tasks and reviews work:

```
                    ┌─────────────────┐
                    │   SUPERVISOR    │
                    │   (routes &     │
                    │   synthesizes)  │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   RESEARCH   │ │   WRITING    │ │   EDITING    │
    │   AGENT      │ │   AGENT      │ │   AGENT      │
    └──────────────┘ └──────────────┘ └──────────────┘
```

**When to use:** Complex tasks requiring multiple specializations, when output quality must be consistent.

**Design parallel:** Like having a creative director who assigns tasks to specialists and ensures the final output is cohesive.

---

**Pattern 2: The Pipeline Model**

Like a design sprint with sequential phases:

```
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │   RESEARCH   │──▶│   IDEATION   │──▶│   DRAFTING   │──▶│   POLISH     │
    │   AGENT      │   │   AGENT      │   │   AGENT      │   │   AGENT      │
    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

**When to use:** When tasks naturally flow in stages, when each stage's output is the next stage's input.

**Design parallel:** Like a design process where research informs ideation, which informs wireframes, which inform final designs.

---

**Pattern 3: The Peer Network**

Like a collaborative design team where anyone can consult anyone:

```
                  ┌──────────────┐
                  │   RESEARCH   │
                  │   AGENT      │
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────────┐           ┌──────────────┐
    │   WRITING    │◄─────────▶│   DESIGN     │
    │   AGENT      │           │   AGENT      │
    └──────┬───────┘           └──────┬───────┘
           │                          │
           └──────────┬───────────────┘
                      │
                      ▼
               ┌──────────────┐
               │   REVIEW     │
               │   AGENT      │
               └──────────────┘
```

**When to use:** When specializations need to collaborate dynamically, when the problem isn't well-defined upfront.

**Design parallel:** Like a cross-functional team where designers, writers, and developers collaborate fluidly rather than in strict handoffs.

---

**Pattern 4: The Expert Panel**

Like a design critique with multiple reviewers:

```
                    ┌─────────────────┐
                    │   INPUT/WORK    │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   EXPERT 1   │ │   EXPERT 2   │ │   EXPERT 3   │
    │   (UX VIEW)  │ │  (TECH VIEW) │ │ (BIZ VIEW)   │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           └────────────────┼────────────────┘
                            ▼
                    ┌─────────────────┐
                    │   SYNTHESIZER   │
                    │   (combines     │
                    │    viewpoints)  │
                    └─────────────────┘
```

**When to use:** When you need multiple perspectives on a single artifact, when quality requires diverse viewpoints.

**Design parallel:** Like a design critique where experts from different disciplines review and provide feedback.

---

### Designing Agent Handoffs

Just as you design handoffs between designers and developers, you need to design handoffs between agents.

**Handoff Design Template:**

```
AGENT HANDOFF: [Source Agent] → [Target Agent]

TRIGGER:
What causes the handoff?
• Completion of source agent's task
• Explicit routing decision
• Error requiring escalation

PAYLOAD:
What information transfers?
• Summary of work completed
• Relevant context/data
• User preferences learned
• Errors encountered

FORMAT:
How is information structured?
• Standardized handoff schema
• Natural language summary
• Structured data + narrative

ACKNOWLEDGMENT:
How does target confirm receipt?
• Explicit acknowledgment message
• Continuation of task
• Request for clarification if needed

FAILURE MODE:
What if handoff fails?
• Retry with exponential backoff
• Fall back to supervisor
• Escalate to human
```

**Example: Research → Writing Handoff**

```
AGENT HANDOFF: Research Agent → Writing Agent

TRIGGER:
Research complete, facts gathered and verified

PAYLOAD:
{
  "topic": "Climate change effects on coastal cities",
  "key_facts": [...],
  "sources": [...],
  "confidence": 0.85,
  "user_context": {
    "expertise_level": "general audience",
    "requested_length": "500 words"
  }
}

FORMAT:
Structured JSON with narrative summary:
"I've gathered 12 key facts about climate change effects on
coastal cities, with 8 high-quality sources. The user wants
a general-audience piece around 500 words."

ACKNOWLEDGMENT:
Writing Agent confirms:
"Got the research. I'll draft a 500-word piece for a general
audience. Starting now..."

FAILURE MODE:
If Writing Agent unavailable:
1. Queue the handoff for retry (3 attempts)
2. Notify supervisor of delay
3. If persistent failure, escalate to human editor
```

---

## Part 5: Governance and Maintenance

### The Living System

Design systems aren't static. They evolve. They require governance—processes for adding components, deprecating old ones, and maintaining quality.

Agent systems are the same.

### Agent System Governance

```
GOVERNANCE STRUCTURE

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  COMPONENT REGISTRY                                                │
│  Central catalog of all available agent components                 │
│  ─────────────────────────────────────────────────────────────────│
│  • All atoms (tools, prompts, memory stores)                       │
│  • All molecules (combined capabilities)                           │
│  • All organisms (single-purpose agents)                           │
│  • Version history and compatibility                               │
│                                                                     │
│  ADDITION PROCESS                                                  │
│  How new components get approved                                   │
│  ─────────────────────────────────────────────────────────────────│
│  1. Proposal with use case justification                           │
│  2. Review for overlap with existing components                    │
│  3. Testing against quality standards                              │
│  4. Documentation requirement                                      │
│  5. Gradual rollout with monitoring                                │
│                                                                     │
│  DEPRECATION PROCESS                                               │
│  How old components get retired                                    │
│  ─────────────────────────────────────────────────────────────────│
│  1. Identify replacement or reason for removal                     │
│  2. Flag as deprecated (still works, but warned)                   │
│  3. Migration period with support                                  │
│  4. Removal after migration complete                               │
│                                                                     │
│  QUALITY STANDARDS                                                 │
│  Requirements for all components                                   │
│  ─────────────────────────────────────────────────────────────────│
│  • Consistent naming conventions                                   │
│  • Standard input/output schemas                                   │
│  • Error handling patterns                                         │
│  • Documentation format                                            │
│  • Test coverage requirements                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Documentation Standards

Every agent component needs documentation, just like every design system component:

```markdown
# Component: Research Tool

## Purpose
Searches the web for information and returns structured results.

## Interface
```python
def search_web(query: str, max_results: int = 5) -> dict:
    """
    Args:
        query: Natural language search query
        max_results: Maximum number of results (1-20)

    Returns:
        {
            "results": [
                {
                    "title": str,
                    "url": str,
                    "snippet": str,
                    "confidence": float
                }
            ],
            "query_interpreted": str
        }
    """
```

## Usage Examples

Basic search:
```python
results = search_web("latest AI developments")
```

Limited results:
```python
results = search_web("climate change effects", max_results=3)
```

## Dependencies
- Requires `SEARCH_API_KEY` environment variable
- Rate limited to 100 queries/hour

## Related Components
- `evaluate_source`: Assess reliability of search results
- `summarize_text`: Condense search findings

## Version History
- v1.0.0: Initial release
- v1.1.0: Added confidence scores
- v1.2.0: Added query interpretation in response
```

---

## Part 6: Practical Application — Designing an Agent Design System

### The Brief

Design the foundation of an agent design system for a customer service department. They need agents for:
- General inquiries
- Order status and tracking
- Returns and refunds
- Technical support
- Billing questions

Currently, each is being built independently, creating inconsistency.

### Agent Design System Blueprint

```
CUSTOMER SERVICE AGENT DESIGN SYSTEM

═══════════════════════════════════════════════════════════════════

1. TOKENS
─────────────────────────────────────────────────────────────────

IDENTITY:
  COMPANY_NAME: "TechCorp"
  TEAM_NAME: "Support"

PERSONALITY:
  TONE: "helpful-professional"
  WARMTH: 0.7
  PATIENCE: 0.9  # High patience for support contexts

BEHAVIOR:
  MAX_TURNS_BEFORE_ESCALATION: 5
  CONFIDENCE_THRESHOLD: 0.75
  ALWAYS_OFFER_HUMAN: true

SAFETY:
  CAN_ACCESS_ORDER_DATA: true
  CAN_PROCESS_REFUNDS: true (with confirmation)
  CAN_ACCESS_PAYMENT_INFO: false (escalate to human)
  BLOCKED_TOPICS: ["legal advice", "competitor comparisons"]

ESCALATION:
  HUMAN_QUEUE: "support@techcorp.com"
  ESCALATION_TRIGGERS: [
    "customer explicitly requests human",
    "confidence below 0.5",
    "customer expresses strong frustration",
    "issue involves payment security"
  ]

═══════════════════════════════════════════════════════════════════

2. ATOMS (Shared Tools)
─────────────────────────────────────────────────────────────────

CUSTOMER TOOLS:
  • lookup_customer(email or phone)
  • get_customer_history(customer_id)
  • update_customer_notes(customer_id, note)

ORDER TOOLS:
  • get_order_status(order_id)
  • get_tracking_info(order_id)
  • list_recent_orders(customer_id)

PRODUCT TOOLS:
  • search_products(query)
  • get_product_details(product_id)
  • check_inventory(product_id)

ACTION TOOLS:
  • initiate_return(order_id, reason)
  • process_refund(order_id, amount)
  • create_support_ticket(details)
  • escalate_to_human(reason)

PROMPT FRAGMENTS:
  • GREETING_STANDARD
  • EMPATHY_PHRASES
  • CONFIRMATION_PATTERNS
  • CLOSURE_STANDARD

═══════════════════════════════════════════════════════════════════

3. MOLECULES (Combined Capabilities)
─────────────────────────────────────────────────────────────────

ORDER_STATUS_FLOW:
  = lookup_customer → list_recent_orders → get_order_status
    → get_tracking_info → format_status_response

RETURN_INITIATION_FLOW:
  = lookup_customer → verify_order → check_return_eligibility
    → initiate_return → confirm_with_customer

ISSUE_DIAGNOSIS_FLOW:
  = gather_symptoms → search_knowledge_base → match_known_issues
    → suggest_resolution

═══════════════════════════════════════════════════════════════════

4. ORGANISMS (Specialized Agents)
─────────────────────────────────────────────────────────────────

GENERAL_INQUIRY_AGENT:
  Purpose: Handle broad questions, route to specialists
  Capabilities: All lookup tools, escalation
  Personality: Welcoming, triage-focused

ORDER_STATUS_AGENT:
  Purpose: Track orders, provide shipping updates
  Capabilities: Order tools, tracking tools
  Personality: Efficient, detail-oriented

RETURNS_AGENT:
  Purpose: Process returns and refunds
  Capabilities: Order tools, return tools, refund tools
  Personality: Empathetic, solution-focused

TECH_SUPPORT_AGENT:
  Purpose: Troubleshoot product issues
  Capabilities: Product tools, diagnostics, knowledge base
  Personality: Patient, methodical

BILLING_AGENT:
  Purpose: Answer billing questions
  Capabilities: Customer tools, order history
  Personality: Precise, reassuring
  Notes: CANNOT access full payment info—escalates

═══════════════════════════════════════════════════════════════════

5. TEMPLATE (Orchestration Pattern)
─────────────────────────────────────────────────────────────────

SUPERVISOR PATTERN:

         ┌──────────────────────────────────────────┐
         │           ROUTER AGENT                   │
         │   (classifies intent, routes to          │
         │    appropriate specialist)               │
         └──────────────────┬───────────────────────┘
    ┌─────────────┬─────────┼─────────┬─────────────┐
    ▼             ▼         ▼         ▼             ▼
┌────────┐  ┌─────────┐  ┌───────┐  ┌──────┐  ┌─────────┐
│General │  │ Order   │  │Returns│  │ Tech │  │ Billing │
│Inquiry │  │ Status  │  │       │  │      │  │         │
└────────┘  └─────────┘  └───────┘  └──────┘  └─────────┘

═══════════════════════════════════════════════════════════════════

6. STYLE GUIDE (Excerpts)
─────────────────────────────────────────────────────────────────

OPENING:
  Standard: "Hi! I'm here to help. What can I do for you?"
  Returning: "Welcome back, [Name]. How can I help today?"

EMPATHY PHRASES:
  "I understand how frustrating that is."
  "I'm sorry you're dealing with this."
  "That shouldn't have happened—let's fix it."

CONFIRMATION:
  Before actions: "Just to confirm, you'd like me to [action].
                   Is that right?"
  After actions:  "Done! I've [completed action]. You should
                   receive [expected outcome]."

HANDOFF TO HUMAN:
  "I want to make sure you get the best help for this.
   Let me connect you with a specialist who can [specific help].
   They'll be with you shortly."

CLOSING:
  "Is there anything else I can help with?"
  "Thanks for reaching out. Have a great day!"

FORBIDDEN:
  • "I cannot help with that" (use: "I can't do that directly,
     but here's what I can do...")
  • "That's not my department" (use: "Let me connect you with
     someone who specializes in that")
  • Technical jargon without explanation
  • Blaming the customer

═══════════════════════════════════════════════════════════════════
```

---

## Synthesis: You Build Systems Already

Design systems are about creating coherence at scale. You've been doing this work—establishing patterns, enforcing consistency, enabling others to create within constraints.

Agent systems need the same discipline:
- **Atoms** that are consistent and well-defined
- **Molecules** that combine capabilities reliably
- **Organisms** that have clear purposes and boundaries
- **Templates** that encode proven patterns
- **Tokens** that enable system-wide updates
- **Style guides** that ensure coherent personality
- **Governance** that maintains quality over time

You already know how to think at this level. Now you can apply it to intelligence itself.

---

## Key Takeaways

1. **Atomic design applies to agents** — Build from atoms (tools, prompts) through molecules (combined capabilities) to organisms (complete agents) to systems.

2. **Design tokens create consistency** — Abstract configuration into tokens that propagate through all agents.

3. **Style guides are essential** — Document voice, tone, and usage patterns to ensure coherent personality.

4. **Team patterns are organizational design** — Multi-agent systems are teams that need clear roles, handoffs, and communication.

5. **Governance enables evolution** — Like design systems, agent systems need processes for growth and maintenance.

---

## Practice Exercise

Take an existing product or service with multiple customer touchpoints (support, sales, onboarding). Design an agent design system for it:

1. Define the tokens (what values should be consistent?)
2. Identify the atoms (what tools and prompts are shared?)
3. Design the organisms (what specialized agents are needed?)
4. Choose the template (how should they coordinate?)
5. Draft the style guide (how should they all "sound"?)

You'll find that your design system expertise translates directly to this challenge.

---

## The Designer's Path Forward

You've now seen how your skills translate to AI agent design:
- Visual thinking → Agent architecture (Module 1)
- Interaction design → Agent behavior (Module 2)
- Design systems → Agent ecosystems (Module 3)

The field is young. The patterns are still being established. The people building these systems desperately need the human-centered, systems-oriented perspective that designers bring.

This isn't about designers becoming engineers. It's about designers bringing their unique superpowers to a new medium.

The future of AI needs you. Not to make it look good—but to make it work well.

---

*"Good design is actually a lot harder to notice than poor design, in part because good designs fit our needs so well that the design is invisible."*
— Don Norman

*The best AI agent is one that users don't think about—because it just works.*
