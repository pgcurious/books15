# Module 2: From User Flows to Agent Workflows

## Interaction Design for Intelligent Systems

*Time: 90 minutes*

---

## The Interaction Designer's Superpower

You obsess over moments. The precise timing of a loading animation. The exact wording of a button label. The feedback when a form validates. The recovery when something goes wrong. These "micro-interactions" are invisible when they work and infuriating when they don't.

This obsession—this attention to the moment-by-moment experience—is exactly what AI agents need and exactly what most AI engineers neglect.

An agent is a series of interactions. Every response is a micro-interaction. Every question is a form field. Every error is a state to handle. Every transition is a motion to design.

This module translates your interaction design expertise into the language of intelligent agents.

---

## Part 1: The Conversation as Interface

### Every Conversation Is an Interface

Consider a typical interface interaction:

```
USER INTERFACE INTERACTION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  What's your email?                                             │
│  ┌─────────────────────────────────────────────────┐           │
│  │ user@example.com                                │  ← INPUT  │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ✓ Email format looks good                          ← FEEDBACK │
│                                                                 │
│                                        [ Continue → ]  ← ACTION│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Now consider an agent conversation:

```
AGENT CONVERSATION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  AGENT: What's your email so I can send the report?   ← PROMPT │
│                                                                 │
│  USER: user@example.com                                ← INPUT  │
│                                                                 │
│  AGENT: Got it! I'll send the report to                        │
│         user@example.com. Give me a moment...         ← FEEDBACK│
│                                                                 │
│  [Agent sends email]                                   ← ACTION │
│                                                                 │
│  AGENT: Done! I've sent the report. Check your inbox. ← CONFIRM│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The structure is identical:
- **Prompt/Label**: What information is needed
- **Input**: User provides data
- **Validation/Feedback**: System acknowledges and confirms
- **Action**: System acts on the data
- **Confirmation**: System confirms completion

The medium changed from pixels to words. The interaction pattern is the same.

### Designing the Conversational Form

Every complex task your agent handles is essentially a multi-step form. Consider how form design principles apply:

| Form Design Principle | Agent Conversation Equivalent |
|-----------------------|------------------------------|
| Group related fields | Group related questions in the same turn |
| One field at a time (for complex forms) | One question at a time (for complex tasks) |
| Smart defaults | Intelligent assumptions based on context |
| Inline validation | Immediate acknowledgment of inputs |
| Clear error messages | Helpful clarification requests |
| Progress indication | Status updates during long tasks |
| Summary before submit | Confirmation before irreversible actions |

**Bad Agent Form Design:**
```
AGENT: Please provide your name, email address, preferred meeting
time (in your timezone), the topic you'd like to discuss, any
materials you'd like me to review beforehand, and whether you'd
prefer video on or off.

USER: Uh... [closes chat]
```

**Good Agent Form Design:**
```
AGENT: Let's schedule a meeting. What's your name?

USER: Sarah Chen

AGENT: Nice to meet you, Sarah. What day and time works best
for you? (I'll adjust for your timezone.)

USER: Next Tuesday at 2pm EST

AGENT: Tuesday, December 17th at 2pm EST - got it. What would
you like to discuss?

USER: Quarterly review of the marketing campaign

AGENT: Perfect. Before we finalize - would you like me to review
any materials beforehand, or should we keep it conversational?

USER: Just conversational

AGENT: Great! Here's what I have:
  • Meeting with Sarah Chen
  • Tuesday, Dec 17, 2pm EST
  • Topic: Q4 Marketing Campaign Review
  • Format: Conversational (no prep materials)

Does this look right? I'll send you a calendar invite once you confirm.
```

The second version applies progressive disclosure, one question at a time, inline confirmation, and summary before action—all interaction design fundamentals.

---

## Part 2: Micro-Interactions in Agent Design

### The Anatomy of an Interaction

In his seminal book on micro-interactions, Dan Saffer identified four parts:

1. **Trigger**: What initiates the interaction
2. **Rules**: What happens during the interaction
3. **Feedback**: How the user knows what's happening
4. **Loops & Modes**: Long-term behavior and variations

These map perfectly to agent design:

```
MICRO-INTERACTION ANATOMY

UI VERSION:                          AGENT VERSION:
─────────────────────────────────────────────────────────────────

TRIGGER                              TRIGGER
  User clicks "Submit"                 User says "Book a flight"
         │                                    │
         ▼                                    ▼
RULES                                RULES
  • Validate all fields                • Parse destination, dates
  • Check against database             • Check availability
  • Create record                      • Compare prices
  • Send confirmation                  • Reserve flight
         │                                    │
         ▼                                    ▼
FEEDBACK                             FEEDBACK
  • Button becomes loading             • "Searching for flights..."
  • Spinner appears                    • "Found 12 options..."
  • Success checkmark                  • "I've booked your flight"
  • Toast notification                 • Summary of booking
         │                                    │
         ▼                                    ▼
LOOPS & MODES                        LOOPS & MODES
  • Retry on failure                   • Retry on API failure
  • Different flows for                • Different flows for
    different user states                different user preferences
```

### Designing Agent Feedback

The feedback layer is where most agents fail. Engineers build technically correct agents that give no indication of what's happening. Users are left staring at a blank screen, unsure if the agent is working, stuck, or broken.

**Feedback Design Principles for Agents:**

#### Principle 1: Acknowledge Immediately

```
BAD:
USER: Search for flights from NYC to Paris next month
[...10 seconds of silence...]
AGENT: Here are 15 flight options...

GOOD:
USER: Search for flights from NYC to Paris next month
AGENT: Looking for NYC to Paris flights in January...
[...search happens...]
AGENT: Found 15 options. Here are the top 5 by price:
```

The acknowledgment ("Looking for...") tells the user three things:
1. The agent heard the request
2. The agent understood the request correctly
3. The agent is working on it

#### Principle 2: Indicate Progress for Long Tasks

```
BAD:
USER: Analyze our Q4 sales data and create a report
[...45 seconds of nothing...]
AGENT: [giant report appears]

GOOD:
USER: Analyze our Q4 sales data and create a report
AGENT: I'll analyze the Q4 sales data and create a comprehensive
       report. This will take a minute or two.

       ✓ Loading sales data... done
       ✓ Analyzing trends... done
       ◐ Generating visualizations... in progress
       ○ Compiling report... pending

AGENT: [Update] Visualizations ready. Now compiling the final report...

AGENT: Here's your Q4 Sales Report:
       [report content]
```

#### Principle 3: Explain What Went Wrong (And What Comes Next)

```
BAD:
USER: Send this email to john@company.com
AGENT: Error: Could not send email.

GOOD:
USER: Send this email to john@company.com
AGENT: I tried to send the email, but the address john@company.com
       isn't valid—it looks like that domain doesn't exist.

       Did you mean john@companyname.com? Or could you double-check
       the address?
```

The good version:
- Explains what failed (invalid address)
- Explains why (domain doesn't exist)
- Offers a likely correction
- Invites the user to help resolve

This is exactly how you'd design an error state in a form.

---

## Part 3: State Machines — User Flows by Another Name

### What Is a State Machine?

A state machine is a system that can be in exactly one of a finite number of states at any time. It transitions between states based on inputs.

Sound familiar? It's a user flow. It's a journey map. It's a service blueprint.

```
SIMPLE STATE MACHINE (TOGGLE SWITCH):

    ┌─────────┐   click   ┌─────────┐
    │   OFF   │──────────▶│   ON    │
    └─────────┘           └─────────┘
         ▲                      │
         │        click         │
         └──────────────────────┘
```

```
COMPLEX STATE MACHINE (CHECKOUT FLOW):

┌────────────┐      ┌────────────┐      ┌────────────┐      ┌────────────┐
│    CART    │─────▶│  SHIPPING  │─────▶│  PAYMENT   │─────▶│  COMPLETE  │
└────────────┘      └────────────┘      └────────────┘      └────────────┘
      │                   │                   │
      │                   ▼                   ▼
      │             ┌────────────┐      ┌────────────┐
      │             │   ERROR    │      │   FAILED   │
      │             │ (fix addr) │      │ (retry pay)│
      │             └────────────┘      └────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
┌────────────┐      ┌────────────┐      ┌────────────┐
│  ABANDONED │      │   (back)   │      │   (back)   │
└────────────┘      └────────────┘      └────────────┘
```

You design state machines constantly. You just call them "flows."

### Agent States

Every agent can be modeled as a state machine. At any moment, the agent is in exactly one state, and user input (or tool results) cause transitions between states.

**Customer Support Agent State Machine:**

```
                                    ┌─────────────────────────────────────────────┐
                                    ▼                                             │
┌───────────────┐   user      ┌───────────────┐   classified   ┌──────────────┐ │
│   WAITING     │────────────▶│   RECEIVING   │───────────────▶│  PROCESSING  │ │
│  (idle)       │   message   │   (parsing)   │                │  (working)   │ │
└───────────────┘             └───────────────┘                └──────┬───────┘ │
                                      │                               │         │
                                      │ unclear                       │ done    │
                                      ▼                               ▼         │
                              ┌───────────────┐                ┌──────────────┐ │
                              │   CLARIFYING  │                │  RESPONDING  │ │
                              │  (asking ?s)  │                │  (sending)   │ │
                              └───────┬───────┘                └──────┬───────┘ │
                                      │                               │         │
                                      │ got answer                    │ need    │
                                      ▼                               │ more?   │
                              ┌───────────────┐                       │         │
                              │   RECEIVING   │◀──────────────────────┘         │
                              └───────────────┘                                 │
                                                                                │
                              ┌───────────────┐                                 │
                              │   ESCALATING  │◀───── cannot resolve ───────────┘
                              │  (to human)   │
                              └───────────────┘
```

This is a user flow. The states are screens. The transitions are user actions (and agent actions). The error states (CLARIFYING, ESCALATING) handle edge cases.

### Designing State Transitions

Just as you design the transitions between screens, you need to design the transitions between agent states. Questions to consider:

1. **What triggers this transition?** (User input? Tool result? Timeout?)
2. **What feedback accompanies the transition?** (Status message? Acknowledgment?)
3. **Is the transition reversible?** (Can the user go back?)
4. **What happens if the transition fails?** (Error state? Retry? Escalation?)

**Transition Design Template:**

```
TRANSITION: [FROM STATE] → [TO STATE]

Trigger:       What causes this transition?
Conditions:    What must be true for it to happen?
Actions:       What happens during the transition?
Feedback:      What does the user see/hear?
Failure Mode:  What if it can't complete?
Recovery:      How does the user get unstuck?
```

**Example: PROCESSING → RESPONDING**

```
TRANSITION: PROCESSING → RESPONDING

Trigger:       Agent finishes using tools and has an answer
Conditions:    Tool calls succeeded, answer confidence is high
Actions:       Format response, check for safety issues, prepare to send
Feedback:      "I found what you need..." (or similar acknowledgment)
Failure Mode:  Answer confidence too low or safety check fails
Recovery:      → CLARIFYING (ask for more info) or
               → ESCALATING (transfer to human)
```

---

## Part 4: Progressive Disclosure in Agent Responses

### The Principle

Progressive disclosure is the practice of revealing information gradually—showing what's needed now and providing paths to dig deeper. It's how you manage complexity in interfaces without overwhelming users.

Agents need the same discipline.

### Information Architecture of Agent Responses

**Bad: All Information at Once**
```
AGENT: Here's the weather for New York City. Currently it's 42°F
(5.6°C) with humidity at 65% and winds from the northwest at
12 mph with gusts up to 18 mph. Barometric pressure is 30.12
inHg and rising. Visibility is 10 miles. UV index is 2. Today's
high will be 48°F and low 35°F. Tomorrow will be cloudy with a
high of 45°F and a 40% chance of rain starting at 3pm with
expected accumulation of 0.2 inches. Thursday will be...
```

**Good: Progressive Disclosure**
```
AGENT: It's 42°F and cloudy in NYC right now.

Tomorrow looks like rain in the afternoon—want me to give you
the full week forecast?
```

The second version:
- Leads with the essential information
- Mentions the most relevant upcoming change
- Offers a path to more detail without forcing it

### Levels of Disclosure

Design your agent responses with explicit disclosure levels:

```
RESPONSE STRUCTURE:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  LEVEL 1: HEADLINE (always shown)                                  │
│  The single most important piece of information                    │
│  ─────────────────────────────────────────────────────────────────│
│                                                                     │
│  LEVEL 2: SUMMARY (shown when relevant)                            │
│  2-3 key supporting facts or next steps                            │
│  ─────────────────────────────────────────────────────────────────│
│                                                                     │
│  LEVEL 3: DETAILS (shown on request)                               │
│  Comprehensive information for those who need it                   │
│  ─────────────────────────────────────────────────────────────────│
│                                                                     │
│  LEVEL 4: REFERENCE (linked/offered)                               │
│  Source data, technical details, raw information                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Example: Research Query**

```
LEVEL 1 (Always):
"Climate change is raising global temperatures by about 0.2°C per decade."

LEVEL 2 (When relevant):
"This is causing more extreme weather events and rising sea levels.
The Paris Agreement aims to limit warming to 1.5°C."

LEVEL 3 (On request):
[Detailed explanation of mechanisms, regional variations, mitigation
strategies, economic impacts...]

LEVEL 4 (Linked):
"Sources: IPCC 2023 Report, NASA Climate Data, Nature Climate Change"
```

### Designing Progressive Disclosure Triggers

How does the agent know when to go deeper? Design explicit triggers:

```
TRIGGER PATTERNS FOR DISCLOSURE:

EXPLICIT REQUEST:
  User: "Tell me more about..."
  User: "What are the details on..."
  User: "Can you elaborate..."
  → Go to next disclosure level

IMPLICIT SIGNAL:
  User: "Wait, what about [specific aspect]?"
  → User needs more detail on that specific thing

CONTEXT-BASED:
  User is a domain expert (detected from language)
  → Start at higher disclosure level

TASK-BASED:
  User is making a decision
  → Provide comparison-level detail

  User is just curious
  → Stay at summary level unless asked
```

---

## Part 5: Error States and Recovery Paths

### Designing for Failure

You know that error states are not edge cases—they're primary design concerns. A form without error handling isn't done. A flow without recovery paths is broken.

Agent design is the same. Errors are not exceptions; they're expected states that require thoughtful design.

### Categories of Agent Errors

```
ERROR TAXONOMY:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  UNDERSTANDING ERRORS (Agent doesn't get it)                       │
│  • Ambiguous request                                               │
│  • Missing information                                             │
│  • Out of scope                                                    │
│  ─────────────────────────────────────────────────────────────────│
│                                                                     │
│  CAPABILITY ERRORS (Agent can't do it)                             │
│  • Tool not available                                              │
│  • Permission denied                                               │
│  • Resource not found                                              │
│  ─────────────────────────────────────────────────────────────────│
│                                                                     │
│  EXECUTION ERRORS (Something broke)                                │
│  • API failure                                                     │
│  • Timeout                                                         │
│  • Rate limit                                                      │
│  ─────────────────────────────────────────────────────────────────│
│                                                                     │
│  CONFIDENCE ERRORS (Agent isn't sure)                              │
│  • Low confidence in answer                                        │
│  • Conflicting information                                         │
│  • Incomplete data                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Error Message Design

Apply the same principles you use for UI error messages:

**Bad Error Messages:**
```
• "Error"
• "Something went wrong"
• "I couldn't do that"
• "Request failed"
```

**Good Error Messages:**
```
• "I didn't find a restaurant called 'Olive Garden' in Tokyo.
   There is one in Yokohama—did you mean that?"

• "I can search for flights, but I can't book them directly.
   Want me to find options for you to book on the airline's site?"

• "The weather service is temporarily unavailable. I can try
   again in a few seconds, or you can check weather.com directly."

• "I found conflicting information about that. Source A says X,
   but Source B says Y. Which would you like me to use, or should
   I dig deeper?"
```

Good error messages:
1. **Acknowledge** the failure
2. **Explain** why it happened (in human terms)
3. **Suggest** what to do next
4. **Offer** alternatives when possible

### Designing Recovery Flows

Every error needs a recovery path. Design them explicitly:

```
ERROR RECOVERY PATTERNS:

CLARIFICATION LOOP:
┌───────────────┐     unclear    ┌───────────────┐     clarified
│   PROCESSING  │───────────────▶│   CLARIFYING  │──────────────┐
└───────────────┘                └───────────────┘              │
        ▲                                                        │
        └────────────────────────────────────────────────────────┘


RETRY WITH BACKOFF:
┌───────────────┐     timeout    ┌───────────────┐     success
│   EXECUTING   │───────────────▶│   WAITING     │──────────────┐
└───────────────┘                └──────┬────────┘              │
        ▲                               │                       │
        │         retry (3x max)        │                       ▼
        └───────────────────────────────┘              ┌───────────────┐
                        │                              │   COMPLETE    │
                        │ max retries                  └───────────────┘
                        ▼
              ┌───────────────┐
              │   FALLBACK    │
              │   (alt path)  │
              └───────────────┘


GRACEFUL DEGRADATION:
┌───────────────┐     partial    ┌───────────────┐
│   FULL        │───────────────▶│   DEGRADED    │
│   SERVICE     │                │   SERVICE     │
└───────────────┘                └───────────────┘
        │                               │
        │ failure                       │ works with limits
        ▼                               ▼
┌───────────────┐                ┌───────────────┐
│   FALLBACK    │                │   COMPLETE    │
│   (manual)    │                │   (partial)   │
└───────────────┘                └───────────────┘
```

---

## Part 6: Designing Agent Personality Through Interaction

### Voice and Tone in Agent Design

You know that a brand's voice remains consistent while its tone adapts to context. A playful brand is still playful when delivering bad news—just more gently.

Agents need the same design attention.

**Agent Voice (Consistent):**
- Personality traits (helpful, witty, formal, casual)
- Values (accuracy, speed, thoroughness)
- Communication style (concise, verbose, emoji-using)

**Agent Tone (Context-Dependent):**
- Success: Celebratory but not excessive
- Error: Helpful and non-blaming
- Waiting: Informative and patient
- Complexity: Clear and structured

### Personality Design Template

```
AGENT PERSONALITY DESIGN

Name: ____________________

VOICE (constant across all interactions):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│ Personality:   □ Formal    □ Professional   □ Casual   □ Playful│
│                                                                 │
│ Intelligence:  □ Expert    □ Knowledgeable  □ Learning together │
│                                                                 │
│ Warmth:        □ Warm      □ Neutral        □ All business      │
│                                                                 │
│ Pacing:        □ Fast/Direct   □ Thorough   □ Conversational   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

TONE VARIATIONS:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│ When succeeding:    ___________________________________________ │
│                                                                 │
│ When failing:       ___________________________________________ │
│                                                                 │
│ When uncertain:     ___________________________________________ │
│                                                                 │
│ When asking:        ___________________________________________ │
│                                                                 │
│ When waiting:       ___________________________________________ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

PHRASES TO USE:                   PHRASES TO AVOID:
• _______________________         • _______________________
• _______________________         • _______________________
• _______________________         • _______________________
```

**Example: Culinary Assistant Agent**

```
AGENT PERSONALITY DESIGN

Name: Chef Claude

VOICE:
  Personality:   Professional with warmth
  Intelligence:  Expert (culinary school graduate)
  Warmth:        Warm (encouraging teacher)
  Pacing:        Conversational but efficient

TONE VARIATIONS:
  When succeeding:    "Excellent choice! Here's what you'll need..."
  When failing:       "Hmm, I'm not finding that. Let me try another approach..."
  When uncertain:     "I'd usually suggest X, but there are other options..."
  When asking:        "Quick question—how many people are you cooking for?"
  When waiting:       "Just looking through my recipes... one moment."

PHRASES TO USE:                   PHRASES TO AVOID:
• "Great choice"                  • "ERROR"
• "Let me find..."               • "I cannot..."
• "Here's a tip:"                • "You should..."
• "What sounds good?"            • "You must..."
```

---

## Part 7: Practical Application — Designing an Interaction Flow

### The Brief

Design the complete interaction flow for an agent that helps users write professional emails. Consider:
- How does it gather information?
- How does it provide feedback?
- How does it handle errors?
- What's its personality?

### Complete Interaction Design

```
EMAIL WRITING ASSISTANT — INTERACTION DESIGN

═══════════════════════════════════════════════════════════════════

TRIGGER: User says "Help me write an email" (or similar)

═══════════════════════════════════════════════════════════════════

STATE 1: GATHERING CONTEXT
─────────────────────────────────────────────────────────────────

[Initial prompt — progressive disclosure]

AGENT: I'll help you write an email. First, what's the main
       purpose? For example:
       • Asking for something
       • Following up
       • Sharing information
       • Apologizing/explaining

USER: I need to ask my boss for time off

AGENT: Got it—a time-off request. A few quick questions:

       Who's your boss? (Just first name is fine)

USER: Sarah

AGENT: And what dates are you hoping to take off?

USER: December 23-27

AGENT: Last one: Any specific reason you'd like to mention,
       or should I keep it general?

USER: My family is visiting from overseas

[Context gathered — transition to next state]

═══════════════════════════════════════════════════════════════════

STATE 2: DRAFTING
─────────────────────────────────────────────────────────────────

[Acknowledge, show progress, deliver draft]

AGENT: Perfect. Let me draft this for you...

       Here's a professional time-off request:

       ───────────────────────────────────────────────────
       Subject: Time Off Request: December 23-27

       Hi Sarah,

       I'd like to request time off from December 23-27.
       My family is visiting from overseas, and I'd love
       to spend the holidays with them.

       I'll make sure all my responsibilities are covered
       before I leave. Let me know if you need any
       additional information.

       Thanks for considering!
       [Your name]
       ───────────────────────────────────────────────────

       Want me to adjust the tone, add anything, or make
       it shorter/longer?

═══════════════════════════════════════════════════════════════════

STATE 3: REFINING (loop until satisfied)
─────────────────────────────────────────────────────────────────

[Handle revision requests with feedback]

USER: Can you make it a bit more formal?

AGENT: Sure, here's a more formal version:

       ───────────────────────────────────────────────────
       Subject: Time Off Request: December 23-27

       Dear Sarah,

       I am writing to formally request time off from
       December 23rd through December 27th. My family
       will be visiting from overseas, and I would
       appreciate the opportunity to spend the holiday
       period with them.

       I will ensure all current projects are on track
       and that coverage is arranged before my departure.
       Please let me know if you require any additional
       information or if there are any concerns.

       Thank you for your consideration.

       Best regards,
       [Your name]
       ───────────────────────────────────────────────────

       How's that?

USER: Perfect!

═══════════════════════════════════════════════════════════════════

STATE 4: COMPLETION
─────────────────────────────────────────────────────────────────

[Confirm and offer next steps]

AGENT: Great! You can copy this into your email client.

       Would you like me to:
       • Save this as a template for future requests?
       • Help you draft a follow-up if you don't hear back?

       Good luck with the request!

═══════════════════════════════════════════════════════════════════

ERROR STATES:
─────────────────────────────────────────────────────────────────

UNCLEAR PURPOSE:
USER: "Write an email"
AGENT: "Sure! What's the email about? I can help with anything
        from work requests to personal messages."

MISSING KEY INFO:
USER: "Ask for time off"
AGENT: "I'll help with that! I just need to know:
        • Who it's going to (your manager's name?)
        • What dates you need off"

CONFLICTING REQUEST:
USER: "Make it more formal but also more casual"
AGENT: "Those are opposites! Let me show you both versions
        and you can pick which direction feels right."

═══════════════════════════════════════════════════════════════════

PERSONALITY:
─────────────────────────────────────────────────────────────────

Voice:     Professional, efficient, helpful
Warmth:    Friendly but not overly casual
Pacing:    Conversational, one step at a time

Phrases:   "I'll help you with that..."
           "Here's what I have..."
           "Want me to adjust..."
           "How's that?"

Avoids:    "I cannot..."
           "You must provide..."
           Technical jargon
           Over-explaining

═══════════════════════════════════════════════════════════════════
```

---

## Synthesis: Interaction Is Your Medium Now

You've spent your career designing interactions—the moment-to-moment experience of using something. You understand that every micro-moment matters. You know that feedback is not optional. You believe that error states deserve as much attention as happy paths.

These skills translate directly to agent design. An agent is not a piece of software that happens to have conversations. An agent is an interaction system, and you are an interaction designer.

The medium changed from pixels to words, from screens to turns, from clicks to utterances. The craft is the same.

---

## Key Takeaways

1. **Conversations are interfaces** — Every agent interaction has the same structure as a form or flow: prompts, inputs, feedback, actions, confirmations.

2. **Micro-interactions apply to words** — Triggers, rules, feedback, and loops work identically in agent design.

3. **State machines are user flows** — You've been designing state machines for years. Now you know what to call them.

4. **Progressive disclosure is essential** — Don't dump information. Lead with headlines, offer details on request.

5. **Error states are primary concerns** — Design for failure with the same care you give to success paths.

6. **Personality is design work** — Voice and tone in agents require the same attention as brand voice in products.

---

## Practice Exercise

Take a complex form you've designed—an application, an onboarding flow, a checkout process. Redesign it as a conversational agent interaction:

1. Map each form field to a conversational prompt
2. Design the feedback for each input
3. Create error messages for common failure modes
4. Plan the progressive disclosure levels
5. Define the agent's personality for this context

You'll find that the conversational version often reveals problems in your original form design—and vice versa.

---

## Next Module

In Module 3, we'll zoom out from individual interactions to systems of agents. You'll learn how your experience building design systems—component libraries, design tokens, style guides—applies to building ecosystems of collaborative AI agents.

---

*"Obvious always wins."*
— Luke Wroblewski

*In agent design, as in interface design, clarity is kindness.*
