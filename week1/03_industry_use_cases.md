# Module 1.3: Industry Use Cases

## What You'll Learn
- How agentic AI patterns apply across industries
- Real-world architectures and implementations
- Common patterns and anti-patterns
- How to identify opportunities for agentic AI in any domain

---

## The Universal Agent Pattern

Before diving into specific industries, let's establish the universal pattern that underlies all agentic AI applications:

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIVERSAL AGENT PATTERN                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   INPUT          REASONING         ACTION          OUTPUT       │
│     │                │                │               │         │
│     ▼                ▼                ▼               ▼         │
│  ┌──────┐       ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Task │──────►│ Analyze  │───►│ Execute  │───►│ Deliver  │   │
│  │      │       │ & Plan   │    │ Tools    │    │ Result   │   │
│  └──────┘       └──────────┘    └──────────┘    └──────────┘   │
│     ▲                │                │               │         │
│     │                ▼                ▼               │         │
│     │           ┌────────────────────────┐           │         │
│     └───────────│       MEMORY           │───────────┘         │
│                 └────────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

This pattern repeats across every industry. What changes are:
- **The domain-specific tools**
- **The domain knowledge in prompts/memory**
- **The types of tasks and outputs**

---

## Industry 1: Customer Service

### The Problem (First Principles)

**First Principle #1:** Customers want fast, accurate answers 24/7.
**First Principle #2:** Human agents are expensive and can't scale infinitely.
**First Principle #3:** Most customer queries follow predictable patterns.

**Conclusion:** An agent that can handle common queries, access customer data, and escalate complex issues should handle 70-80% of volume.

### The Analogy: Tiered Support

Think of it like a hospital emergency room:
- **Triage Nurse (First-line Agent)**: Quick assessment, handles simple cases
- **Specialist (Second-line Agent)**: Handles complex domain-specific issues
- **Doctor (Human Agent)**: Called in for critical or unusual cases

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                CUSTOMER SERVICE AGENT ARCHITECTURE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Customer Query                                                 │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         INTENT CLASSIFIER                │                   │
│   │    (Route to appropriate handler)        │                   │
│   └─────────────────────────────────────────┘                   │
│        │           │           │           │                     │
│        ▼           ▼           ▼           ▼                     │
│   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐               │
│   │ FAQ    │  │ Order  │  │ Tech   │  │ Billing│               │
│   │ Agent  │  │ Agent  │  │ Support│  │ Agent  │               │
│   └────────┘  └────────┘  └────────┘  └────────┘               │
│        │           │           │           │                     │
│        └───────────┴───────────┴───────────┘                     │
│                         │                                        │
│                         ▼                                        │
│            ┌────────────────────────┐                           │
│            │    ESCALATION CHECK    │                           │
│            │  (Confidence < 80%?)   │                           │
│            └────────────────────────┘                           │
│                    │         │                                   │
│                    ▼         ▼                                   │
│              [Respond]  [Human Handoff]                         │
│                                                                  │
│   TOOLS:                                                         │
│   • Customer DB lookup    • Knowledge base search               │
│   • Order management      • Ticket creation                     │
│   • Refund processing     • Human escalation                    │
│                                                                  │
│   MEMORY:                                                        │
│   • Conversation history  • Customer preferences                │
│   • Past interactions     • Resolution patterns                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Emergence in Customer Service

Simple rules + context = sophisticated service:

1. **Empathy emerges** from training on support conversations
2. **De-escalation emerges** from exposure to resolution patterns
3. **Personalization emerges** from memory of past interactions
4. **Domain expertise emerges** from knowledge base retrieval

### Code Example: Customer Service Agent

```python
# customer_service_agent.py

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from datetime import datetime

# Domain-specific tools
@tool
def lookup_customer(customer_id: str) -> str:
    """Look up customer information by ID."""
    # Simulated customer database
    customers = {
        "C001": {"name": "Alice Johnson", "tier": "Gold", "since": "2020"},
        "C002": {"name": "Bob Smith", "tier": "Silver", "since": "2022"},
    }
    if customer_id in customers:
        c = customers[customer_id]
        return f"Customer: {c['name']}, Tier: {c['tier']}, Member since: {c['since']}"
    return "Customer not found"

@tool
def lookup_order(order_id: str) -> str:
    """Look up order status and details."""
    orders = {
        "ORD-123": {"status": "Shipped", "eta": "Dec 28", "items": "Widget Pro"},
        "ORD-456": {"status": "Processing", "eta": "Dec 30", "items": "Gadget Plus"},
    }
    if order_id in orders:
        o = orders[order_id]
        return f"Order {order_id}: {o['status']}, ETA: {o['eta']}, Items: {o['items']}"
    return "Order not found"

@tool
def search_knowledge_base(query: str) -> str:
    """Search the support knowledge base for relevant articles."""
    kb = {
        "return": "Return Policy: 30-day returns for unused items. Start at /returns",
        "shipping": "Free shipping over $50. Standard delivery: 5-7 days.",
        "warranty": "All products include 1-year warranty. Gold members get 2 years.",
        "payment": "We accept Visa, MC, Amex, and PayPal. Payment issues: call billing.",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return "No specific article found. Suggest contacting support@company.com"

@tool
def create_support_ticket(
    customer_id: str,
    issue_type: str,
    description: str,
    priority: str = "normal"
) -> str:
    """Create a support ticket for issues requiring human follow-up."""
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"Ticket {ticket_id} created. Priority: {priority}. A specialist will respond within 24 hours."

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent."""
    return f"Escalating to human agent. Reason: {reason}. Estimated wait time: 3 minutes."

# System prompt encoding domain knowledge
CUSTOMER_SERVICE_PROMPT = """You are a helpful customer service agent for TechCo.

Your capabilities:
- Look up customer information and order status
- Answer questions using the knowledge base
- Create support tickets for complex issues
- Escalate to humans when needed

Guidelines:
1. Always greet customers warmly
2. Verify customer identity when accessing personal information
3. Be concise but thorough
4. If unsure, search the knowledge base before responding
5. Escalate if: customer is angry, issue is complex, or you're unsure
6. Always end with "Is there anything else I can help with?"

Customer Context:
{customer_context}

Conversation History:
{history}
"""

class CustomerServiceAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = [
            lookup_customer,
            lookup_order,
            search_knowledge_base,
            create_support_ticket,
            escalate_to_human
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.history = []
        self.customer_context = "Unknown customer"

    def chat(self, user_message: str) -> str:
        # Build prompt with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", CUSTOMER_SERVICE_PROMPT),
            ("human", "{input}")
        ])

        chain = prompt | self.llm_with_tools

        response = chain.invoke({
            "customer_context": self.customer_context,
            "history": "\n".join(self.history[-10:]),
            "input": user_message
        })

        # Handle tool calls...
        self.history.append(f"Customer: {user_message}")
        self.history.append(f"Agent: {response.content}")

        return response.content
```

### Business Impact

| Metric | Before Agent | After Agent |
|--------|-------------|-------------|
| First Response Time | 4 hours | Instant |
| Resolution Rate | 65% | 78% |
| Cost per Ticket | $15 | $3 |
| Customer Satisfaction | 72% | 85% |
| Human Agent Capacity | 100% utilized | Focus on complex cases |

---

## Industry 2: Software Development

### The Problem (First Principles)

**First Principle #1:** Code follows patterns—patterns can be learned.
**First Principle #2:** 60-70% of developer time is reading/understanding code.
**First Principle #3:** Debugging is systematic hypothesis testing.

**Conclusion:** An agent that can read code, understand context, and follow systematic processes can accelerate development significantly.

### The Analogy: Pair Programming Partner

An AI coding agent is like having a brilliant pair programmer who:
- Has read every programming book and Stack Overflow answer
- Never gets tired or frustrated
- Can hold vast amounts of context in mind
- Is always available to discuss or help

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              SOFTWARE DEVELOPMENT AGENT ARCHITECTURE             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Developer Request: "Add user authentication to the app"       │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         CODEBASE ANALYZER                │                   │
│   │    • Read existing code structure       │                   │
│   │    • Identify relevant files            │                   │
│   │    • Understand patterns used           │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │           PLANNER                        │                   │
│   │    • Break down into subtasks           │                   │
│   │    • Identify dependencies              │                   │
│   │    • Estimate complexity                │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │           IMPLEMENTER                    │                   │
│   │    • Write code following patterns      │                   │
│   │    • Generate tests                     │                   │
│   │    • Create documentation               │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │           VALIDATOR                      │                   │
│   │    • Run tests                          │                   │
│   │    • Check for errors                   │                   │
│   │    • Review for security issues         │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                  │
│   TOOLS:                                                         │
│   • File read/write     • Code search                           │
│   • Run terminal        • Git operations                        │
│   • Test runner         • Documentation gen                     │
│                                                                  │
│   MEMORY:                                                        │
│   • Codebase index     • Coding standards                       │
│   • Past changes       • Bug patterns                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Emergence in Coding

Simple capabilities combine into sophisticated behavior:

1. **Pattern recognition** emerges from seeing millions of code examples
2. **Debugging intuition** emerges from associating symptoms with causes
3. **Architectural understanding** emerges from reading diverse codebases
4. **Style adaptation** emerges from context in the codebase

---

## Industry 3: Healthcare (Clinical Decision Support)

### The Problem (First Principles)

**First Principle #1:** Doctors can't remember everything—medical knowledge doubles every 73 days.
**First Principle #2:** Diagnostic errors cause 40,000-80,000 deaths annually in the US.
**First Principle #3:** Most diagnostic information is in unstructured notes and documents.

**Conclusion:** An agent that can synthesize patient data, recall relevant medical knowledge, and suggest differential diagnoses can save lives.

### The Analogy: The Perfect Medical Librarian

Imagine a medical librarian who:
- Has read every medical journal and textbook
- Knows your patient's complete history
- Can instantly recall relevant case studies
- Suggests what a doctor might have missed

**Critical:** The agent **supports** decisions—it doesn't make them. The human physician remains responsible.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│            CLINICAL DECISION SUPPORT ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Physician Query: "55yo male, chest pain, diabetic, smoker"    │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         PATIENT DATA AGGREGATOR          │                   │
│   │    • EHR data extraction                │                   │
│   │    • Lab results synthesis              │                   │
│   │    • Medication history                 │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         DIFFERENTIAL GENERATOR           │                   │
│   │    • Consider symptom patterns          │                   │
│   │    • Weight by risk factors             │                   │
│   │    • Check against medical knowledge    │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         EVIDENCE RETRIEVER               │                   │
│   │    • Find relevant studies              │                   │
│   │    • Cite clinical guidelines           │                   │
│   │    • Note similar cases                 │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         RECOMMENDATION PRESENTER         │                   │
│   │    • Ranked differential diagnosis      │                   │
│   │    • Suggested tests                    │                   │
│   │    • Red flags highlighted              │                   │
│   │    ⚠️ ALWAYS: "Consult physician"        │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                  │
│   TOOLS:                                                         │
│   • EHR query         • PubMed search                           │
│   • Drug interaction  • Lab reference                           │
│   • Guidelines DB     • Risk calculators                        │
│                                                                  │
│   MEMORY:                                                        │
│   • Patient history    • Clinical guidelines                    │
│   • Local patterns     • Physician preferences                  │
│                                                                  │
│   ⚠️ GUARDRAILS:                                                 │
│   • Never diagnose—only suggest differentials                   │
│   • Always cite sources                                         │
│   • Flag uncertainty explicitly                                 │
│   • Require physician confirmation                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Emergence in Healthcare AI

1. **Clinical intuition** emerges from exposure to millions of cases
2. **Rare disease recognition** emerges from pattern matching across literature
3. **Interaction detection** emerges from comprehensive drug databases
4. **Personalized recommendations** emerge from patient-specific context

---

## Industry 4: Finance (Investment Research)

### The Problem (First Principles)

**First Principle #1:** Markets reflect aggregate information—alpha comes from information advantage.
**First Principle #2:** Financial data is vast—earnings calls, filings, news, social sentiment.
**First Principle #3:** Humans can only process a fraction of available information.

**Conclusion:** An agent that can synthesize vast amounts of financial data, identify patterns, and surface insights can provide information advantage.

### The Analogy: The Analyst Team

One AI agent can do the work of an entire analyst team:
- **Junior Analyst**: Gathers and organizes data
- **Senior Analyst**: Identifies trends and anomalies
- **Sector Expert**: Provides domain context
- **Strategist**: Synthesizes into actionable insights

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│             INVESTMENT RESEARCH AGENT ARCHITECTURE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Analyst Query: "Analyze AAPL earnings and competitive position"│
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         DATA AGGREGATION                 │                   │
│   │    • SEC filings (10-K, 10-Q, 8-K)      │                   │
│   │    • Earnings transcripts               │                   │
│   │    • News articles                      │                   │
│   │    • Social sentiment                   │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         FINANCIAL ANALYSIS               │                   │
│   │    • Revenue/margin trends              │                   │
│   │    • Segment performance                │                   │
│   │    • Guidance vs. actuals               │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         COMPETITIVE ANALYSIS             │                   │
│   │    • Market share trends                │                   │
│   │    • Competitor earnings comparison     │                   │
│   │    • Industry dynamics                  │                   │
│   └─────────────────────────────────────────┘                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────┐                   │
│   │         SYNTHESIS & INSIGHTS             │                   │
│   │    • Key takeaways                      │                   │
│   │    • Risk factors                       │                   │
│   │    • Catalyst calendar                  │                   │
│   │    ⚠️ "Not investment advice"            │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                  │
│   TOOLS:                                                         │
│   • SEC EDGAR API      • Financial data APIs                    │
│   • News aggregators   • Sentiment analysis                     │
│   • Excel/modeling     • Chart generation                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Industry 5: Legal (Contract Analysis)

### First Principles

**First Principle #1:** Contracts are structured documents with standard clauses.
**First Principle #2:** Due diligence requires reviewing thousands of pages.
**First Principle #3:** Missing a clause can cost millions.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              CONTRACT ANALYSIS AGENT ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   DOCUMENT INTAKE                                                │
│   └── Parse contracts (PDF, Word, scans)                        │
│        │                                                         │
│        ▼                                                         │
│   CLAUSE EXTRACTION                                              │
│   └── Identify standard clauses (indemnity, IP, termination)    │
│        │                                                         │
│        ▼                                                         │
│   RISK ANALYSIS                                                  │
│   └── Compare against templates, flag deviations                │
│        │                                                         │
│        ▼                                                         │
│   SUMMARY GENERATION                                             │
│   └── Executive summary with key terms and risks                │
│                                                                  │
│   TOOLS:                                                         │
│   • Document parser    • Clause database                        │
│   • Risk scoring       • Comparison engine                      │
│   • Citation finder    • Redline generator                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cross-Industry Patterns

### Pattern 1: The Triage Pattern
```
Input → Classify → Route to Specialist → Execute → Verify → Output
```
**Used in:** Customer service, healthcare, IT support

### Pattern 2: The Research Synthesizer
```
Query → Gather Sources → Extract Key Points → Synthesize → Present
```
**Used in:** Finance, legal, academic research

### Pattern 3: The Workflow Automator
```
Trigger → Validate → Execute Steps → Handle Exceptions → Report
```
**Used in:** DevOps, business process automation

### Pattern 4: The Human-in-the-Loop
```
Analyze → Propose → Wait for Approval → Execute → Confirm
```
**Used in:** High-stakes decisions (medical, financial, legal)

---

## Identifying Opportunities

### The Three Questions

When evaluating if agentic AI fits a use case, ask:

1. **Is there a pattern?**
   - Are tasks repetitive with variations?
   - Can the decision process be articulated?

2. **Is there data?**
   - Are there knowledge bases to query?
   - Is historical data available for context?

3. **Is there value?**
   - What's the cost of the current process?
   - What's the impact of errors?
   - What's the value of speed/scale?

### Anti-Patterns: When NOT to Use Agents

| Situation | Why Agents Don't Fit |
|-----------|---------------------|
| Purely creative tasks | Lack of objective success criteria |
| One-off tasks | Setup cost exceeds benefit |
| Physical-world actions | Agents can't manipulate physical reality |
| Low-volume, high-stakes | Human judgment irreplaceable |
| Rapidly changing domains | Training/knowledge becomes stale |

---

## Exercises

1. **Map Your Industry**: Choose an industry and identify three processes that could benefit from agentic AI

2. **Design an Agent**: Pick one process and design the:
   - Tools it would need
   - Memory it would require
   - Guardrails for safety

3. **Identify Emergence**: What sophisticated behaviors would emerge from simple tools + domain knowledge?

---

## Key Takeaways

1. **The pattern is universal**: Perceive → Reason → Act → Learn
2. **Domain specificity comes from**: Tools + Prompts + Memory content
3. **Value comes from**: Scale, speed, consistency, coverage
4. **Guardrails are essential**: Especially in high-stakes domains
5. **Human-in-the-loop**: Most production systems keep humans involved

---

## What's Next

You now have the conceptual foundation. In **Week 2**, we'll get hands-on:
- Build your first agent with LangChain
- Deploy it as a working application
- Learn debugging and optimization

[→ Week 2: Build Your First Agent](../week2/README.md)
