# Module 01: Minds and Machines

## Cognitive Architecture — From Human to Artificial

> *"Cognitive psychology is really the study of information processing, and information processing is what computers do."*
> — Herbert Simon, Nobel Laureate & AI Pioneer (also a psychologist)

---

## What You'll Learn

In this module, you will:

- Map the cognitive psychology framework to AI system architecture
- Understand how attention mechanisms implement selective attention
- See embeddings as the computational analog of mental representation
- Recognize LLM reasoning as pattern matching and association (just like human cognition)
- Build an agent explicitly structured around cognitive architecture

---

## The Cognitive Revolution Was the AI Revolution

In the 1950s and 60s, psychology underwent a paradigm shift. Behaviorism—with its exclusive focus on observable stimuli and responses—gave way to cognitive psychology, which dared to look inside the "black box" of the mind.

The cognitive revolution's core insight: **The mind is an information processing system.**

```
BEHAVIORISM:    Stimulus ──────────────────────────────► Response
                         (black box - don't ask)

COGNITIVISM:    Stimulus ──► [Perception ──► Processing ──► Decision] ──► Response
                                        (the mind computes)
```

This wasn't just a psychological theory. It was a **claim that minds are computational**. And if minds are computational, then computation could create minds.

AI and cognitive psychology were born together, from the same intellectual parents.

---

## Section 1: Perception — How Minds (and Models) See the World

### The Psychology

Perception isn't passive reception. It's active construction.

You learned about:
- **Bottom-up processing**: Building from sensory features to wholes
- **Top-down processing**: Expectations and knowledge shaping perception
- **Feature detection**: Edge detectors, motion detectors in visual cortex
- **Gestalt principles**: Proximity, similarity, closure, continuity

### The AI Translation

```
HUMAN PERCEPTION                      AI PERCEPTION
─────────────────                     ──────────────

Light hits retina                     Text/image input received
       ↓                                      ↓
Photoreceptors fire                   Tokenization (breaking into units)
       ↓                                      ↓
Feature detectors                     Embedding layer
(edges, colors, motion)               (transform tokens to vectors)
       ↓                                      ↓
Object recognition                    Pattern recognition
(binding features)                    (multi-layer processing)
       ↓                                      ↓
Conscious perception                  Output representation
"I see a cat"                         [0.92, -0.3, 0.7, ...]
```

**Key Insight**: Both systems transform raw input into meaningful representations. Neither works with "reality directly"—both construct interpretations.

### Tokenization as Sensory Transduction

When you learned about sensory transduction—converting physical stimuli into neural signals—you learned the first step of perception.

Tokenization is the same operation for language:

```python
"""
Tokenization: Breaking language into processable units
Like how the visual system breaks scenes into features
"""

from langchain_openai import ChatOpenAI
import tiktoken

# See how text becomes tokens (like how light becomes neural signals)
encoder = tiktoken.encoding_for_model("gpt-4")
text = "Psychology is the science of mind and behavior"
tokens = encoder.encode(text)

print(f"Original: {text}")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")

# Decode to see the actual units
token_strings = [encoder.decode([t]) for t in tokens]
print(f"Token strings: {token_strings}")

# Output might look like:
# ['Psych', 'ology', ' is', ' the', ' science', ' of', ' mind', ' and', ' behavior']
#
# Notice: The model "perceives" in chunks, just like we perceive words, not letters
# This is like Gestalt psychology's principle of chunking!
```

### Embeddings as Mental Representation

Remember learning about mental representation? How concepts exist in the mind not as pictures, but as patterns of activation, relationships, and features?

Embeddings are exactly this:

```python
"""
Embeddings: The computational analog of mental representation
Each concept becomes a point in high-dimensional semantic space
"""

from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings()

# Get the "mental representation" of concepts
concepts = [
    "anxiety",
    "fear",
    "stress",
    "happiness",
    "joy",
    "depression",
    "sadness",
    "anger"
]

# Create embeddings (semantic representations)
vectors = embeddings.embed_documents(concepts)

# Calculate semantic similarity (like priming effects!)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Anxiety is more similar to fear than to happiness
anxiety_fear = cosine_similarity(vectors[0], vectors[1])      # High
anxiety_happiness = cosine_similarity(vectors[0], vectors[3]) # Low

print(f"Anxiety-Fear similarity: {anxiety_fear:.3f}")
print(f"Anxiety-Happiness similarity: {anxiety_happiness:.3f}")

# This IS semantic memory structure!
# The same kind of structure that explains priming,
# semantic networks, and spreading activation
```

**Psychological Connection**: This is Collins & Loftus spreading activation in computational form. Similar concepts are close together. Activation spreads through semantic distance. This is why you can "prime" an AI model, just like you can prime a human subject.

---

## Section 2: Attention — The Gateway to Consciousness (and Computation)

### The Psychology

You studied attention extensively:
- **Selective attention**: Filtering relevant from irrelevant (cocktail party effect)
- **Divided attention**: Multi-tasking and its limits
- **Sustained attention**: Vigilance over time
- **Attentional spotlight**: The metaphor of focusing a beam
- **Top-down vs. bottom-up attention**: Goals vs. salience

### The AI Translation: Attention Mechanisms

In 2017, the paper "Attention Is All You Need" revolutionized AI. The title wasn't metaphorical—they literally built systems around attention.

```
THE ATTENTION MECHANISM: A Psychological Explanation

The model asks: "What should I focus on?"

Query (Q):   What am I looking for?        → Like your current goal/question
Key (K):     What's available to look at?  → Like items in your environment
Value (V):   What information do I get?    → Like what you learn when you attend

Attention Score = How relevant is each item to my query?
Output = Weighted combination of values based on attention

┌────────────────────────────────────────────────────────────────────┐
│              PSYCHOLOGICAL ATTENTION VS. AI ATTENTION             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   HUMAN ATTENTION                    TRANSFORMER ATTENTION         │
│                                                                    │
│   "What am I looking for?"    ───►   Query vector (Q)             │
│   "What's there?"             ───►   Key vectors (K)               │
│   "What do I get if I look?"  ───►   Value vectors (V)             │
│   "How relevant is it?"       ───►   Attention weights (softmax)   │
│   "Spotlight focus"           ───►   High attention weight         │
│   "Background ignored"        ───►   Low attention weight          │
│                                                                    │
│   Multi-head attention = Multiple attentional spotlights          │
│   (Like how you can track multiple objects visually)              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Self-Attention: The Mind Reading Itself

Self-attention is particularly fascinating psychologically. It's the system attending to its own representations—like introspection or metacognition.

```python
"""
Demonstrating attention-like behavior in prompts
You can guide what the model "attends" to
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# Without attentional guidance
prompt_unfocused = ChatPromptTemplate.from_messages([
    ("human", """Here's a case study:

A 28-year-old woman reports difficulty sleeping, racing thoughts,
excessive worry about work, physical tension, fatigue, and
occasional panic attacks. She also mentions she recently started
a new high-pressure job, moved to a new city, ended a long-term
relationship, and her mother was diagnosed with cancer.

What's happening here?""")
])

# With attentional guidance (like directing attention in an experiment)
prompt_focused = ChatPromptTemplate.from_messages([
    ("human", """Here's a case study:

A 28-year-old woman reports difficulty sleeping, racing thoughts,
excessive worry about work, physical tension, fatigue, and
occasional panic attacks. She also mentions she recently started
a new high-pressure job, moved to a new city, ended a long-term
relationship, and her mother was diagnosed with cancer.

FOCUS ON: The relationship between life stressors and symptoms.
Specifically analyze the diathesis-stress model here.

What's happening?""")
])

# The focused prompt will generate more targeted analysis
# Just like how directing attention changes perception!
```

**Key Insight**: Prompt engineering is attention direction. When you tell the model "focus on X" or "consider Y specifically," you're manipulating its attentional allocation—exactly like an experimenter manipulating attention in a lab study.

---

## Section 3: Working Memory — The Cognitive Workspace

### The Psychology

Baddeley's working memory model described:
- **Central executive**: Attentional control system
- **Phonological loop**: Verbal rehearsal
- **Visuospatial sketchpad**: Visual/spatial processing
- **Episodic buffer**: Integration across domains

Key properties:
- Limited capacity (7±2 items, or maybe 4 chunks)
- Rapid decay without rehearsal
- Critical for reasoning and problem-solving
- The "workspace" where thinking happens

### The AI Translation: Context Windows

```
WORKING MEMORY LIMITS          CONTEXT WINDOW LIMITS
────────────────────           ───────────────────────

7±2 items                      ~128K tokens (varies by model)
(Miller's magic number)        (but quality degrades at edges)

Chunking increases capacity    Summarization extends effective capacity

Recency effect                 Recent tokens weighted more heavily
(last items remembered)        (in practice, not architecture)

Decay without rehearsal        Information outside window is "forgotten"

Interference from load         Quality degrades with long contexts

Central executive manages      System prompt + structure manages
allocation                     attention allocation
```

### The Fundamental Constraint

Just as working memory limits constrain human cognition, context windows constrain AI:

```python
"""
Context window as working memory:
Demonstrating the constraint and how to manage it
"""

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-4o-mini")

# Psychological research paper (imagine it's much longer)
long_text = """
[Imagine 50 pages of psychological research here]
The working memory model proposed by Baddeley (1974) has been
extensively studied and refined over decades. The central executive
component, in particular, has been linked to frontal lobe function
and attention control mechanisms. Recent neuroimaging studies suggest
that working memory capacity individual differences correlate with
fluid intelligence measures...
[Much more text that exceeds context limits]
"""

# Solution 1: Chunking (like psychological chunking!)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Chunk size
    chunk_overlap=200      # Overlap to maintain context (like rehearsal!)
)
chunks = splitter.split_text(long_text)

# Now process each chunk, like working memory processes chunks
# The overlap maintains continuity—like keeping active information rehearsed
```

### Maintaining Context Across Limits: Rehearsal in AI

Just as you rehearse to keep information in working memory, AI systems can use summarization and compression to maintain information across the context limit:

```python
"""
Rehearsal strategy: Summarize to maintain information
Like subvocal rehearsal in the phonological loop
"""

from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

# Running summary (like internal rehearsal)
running_summary = ""

# For each new piece of information:
update_prompt = PromptTemplate.from_template("""
Current understanding (rehearsed information):
{running_summary}

New information:
{new_info}

Update the running summary to incorporate new information
while maintaining the most important points. Be concise.
""")

# This is cognitive rehearsal implemented computationally!
```

---

## Section 4: Reasoning — How Minds Solve Problems

### The Psychology

You studied reasoning extensively:
- **Deductive reasoning**: Applying rules to reach conclusions
- **Inductive reasoning**: Finding patterns, forming generalizations
- **Analogical reasoning**: Mapping structures between domains
- **Heuristics**: Mental shortcuts (availability, representativeness, anchoring)
- **Problem-solving**: Means-ends analysis, working backward

### The AI Translation: How LLMs "Think"

Here's a crucial insight: **LLMs don't reason symbolically. They pattern-match statistically.**

This is actually closer to how humans *actually* reason (not how we think we reason):

```
DUAL-PROCESS THEORY (Kahneman)        LLM PROCESSING
───────────────────────────           ──────────────────

System 1: Fast, automatic,            Default LLM response:
intuitive, pattern-based              Statistical, associative,
                                      based on training patterns

System 2: Slow, deliberate,           Chain-of-thought prompting:
effortful, rule-following             Step-by-step, more "deliberate"

System 1 usually wins                 Direct prompts = System 1 answers

System 2 requires activation          "Think step by step" activates
(motivation, time, capacity)          more careful processing
```

### Chain-of-Thought as System 2 Activation

```python
"""
Chain-of-thought prompting:
Activating "System 2" processing in an LLM
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# System 1 prompt (fast, intuitive, error-prone)
system1_prompt = ChatPromptTemplate.from_messages([
    ("human", """A bat and ball cost $1.10 together.
The bat costs $1 more than the ball.
How much does the ball cost?""")
])

# System 2 prompt (slow, deliberate, more accurate)
system2_prompt = ChatPromptTemplate.from_messages([
    ("human", """A bat and ball cost $1.10 together.
The bat costs $1 more than the ball.
How much does the ball cost?

Think through this step-by-step:
1. Define variables for what we don't know
2. Write equations from the problem
3. Solve the equations
4. Check your answer""")
])

# System 1 often gives "10 cents" (wrong, but intuitive)
# System 2 more likely to give "5 cents" (correct)

# This is the CRT (Cognitive Reflection Test) problem!
# You probably studied this in cognitive psychology.
# Same effect in humans and LLMs.
```

### Heuristics and Biases in LLMs

Your training in cognitive biases directly applies:

```python
"""
Demonstrating anchoring bias in LLMs
Just like in human cognition!
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Low anchor condition
low_anchor = ChatPromptTemplate.from_messages([
    ("human", """Do you think the percentage of African nations
in the United Nations is higher or lower than 10%?

What is your estimate of the actual percentage?""")
])

# High anchor condition
high_anchor = ChatPromptTemplate.from_messages([
    ("human", """Do you think the percentage of African nations
in the United Nations is higher or lower than 65%?

What is your estimate of the actual percentage?""")
])

# Run both and compare estimates
# LLMs show anchoring effects, just like Tversky & Kahneman's subjects!
# Your understanding of biases helps you predict and mitigate AI errors.
```

---

## Section 5: Building a Cognitive Architecture Agent

Now let's integrate everything into an agent explicitly modeled on cognitive architecture:

```python
"""
A Cognitive Architecture Agent
Explicitly structured around psychological modules
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Any
import json

class CognitiveArchitectureAgent:
    """
    An agent built on cognitive psychology principles:
    - Perception: Input processing and representation
    - Attention: Selective focus on relevant information
    - Working Memory: Current context and active processing
    - Long-term Memory: Persistent knowledge store (semantic memory)
    - Executive Function: Goal management and action selection
    - Response Generation: Output production
    """

    def __init__(self):
        # The "brain" - core processing (prefrontal cortex analog)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        # Perception/Representation system (sensory cortex analog)
        self.embeddings = OpenAIEmbeddings()

        # Long-term memory - semantic memory store (hippocampus/cortex analog)
        self.long_term_memory = Chroma(
            collection_name="semantic_memory",
            embedding_function=self.embeddings
        )

        # Working memory - current conversation context (prefrontal WM)
        self.working_memory: List[Any] = []
        self.working_memory_capacity = 10  # Like Miller's 7±2

        # Episodic memory - specific experiences
        self.episodic_memory: List[Dict] = []

        # Current goals (executive function)
        self.current_goals: List[str] = []

        # Emotional state (affect influences cognition)
        self.emotional_state = {"valence": 0.0, "arousal": 0.0}

    def perceive(self, input_text: str) -> Dict[str, Any]:
        """
        PERCEPTION MODULE
        Transform raw input into internal representation
        (Like sensory transduction + early perceptual processing)
        """
        # Create embedding representation
        embedding = self.embeddings.embed_query(input_text)

        # Detect emotional content (affect perception)
        affect_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the emotional tone. Reply with JSON: {\"valence\": -1 to 1, \"arousal\": 0 to 1}"),
            ("human", "{input}")
        ])
        affect_response = (affect_prompt | self.llm).invoke({"input": input_text})

        try:
            detected_affect = json.loads(affect_response.content)
        except:
            detected_affect = {"valence": 0.0, "arousal": 0.5}

        return {
            "raw_input": input_text,
            "embedding": embedding,
            "detected_affect": detected_affect
        }

    def attend(self, perception: Dict, query: str = None) -> List[Document]:
        """
        ATTENTION MODULE
        Selective retrieval from long-term memory based on relevance
        (Like spotlight attention + memory retrieval)
        """
        # What to attend to - current input or specific query
        attention_target = query if query else perception["raw_input"]

        # Retrieve relevant memories (spreading activation in semantic memory)
        try:
            relevant_memories = self.long_term_memory.similarity_search(
                attention_target,
                k=3  # Attention capacity - can only focus on a few things
            )
        except:
            relevant_memories = []

        return relevant_memories

    def update_working_memory(self, item: Any):
        """
        WORKING MEMORY MANAGEMENT
        Limited capacity, oldest items displaced (like decay/interference)
        """
        self.working_memory.append(item)

        # Capacity limit - displace oldest items (forgetting)
        if len(self.working_memory) > self.working_memory_capacity:
            forgotten = self.working_memory.pop(0)
            # Could store to episodic memory here (consolidation)
            self.episodic_memory.append({
                "content": str(forgotten),
                "consolidated_at": "now"  # Timestamp in real implementation
            })

    def encode_to_ltm(self, content: str, metadata: Dict = None):
        """
        MEMORY ENCODING
        Store new information in long-term memory
        (Like hippocampal encoding → cortical consolidation)
        """
        doc = Document(
            page_content=content,
            metadata=metadata or {"type": "semantic"}
        )
        self.long_term_memory.add_documents([doc])

    def executive_function(self,
                          perception: Dict,
                          attended_memories: List[Document]) -> str:
        """
        EXECUTIVE FUNCTION / CENTRAL EXECUTIVE
        Goal management, planning, response selection
        (Prefrontal cortex functions)
        """
        # Compile context for decision-making
        memory_context = "\n".join([m.page_content for m in attended_memories])
        working_memory_context = "\n".join([str(m) for m in self.working_memory[-5:]])

        executive_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the executive function of a cognitive system.

Your role:
1. Assess the current situation (input + memories + goals)
2. Decide on the appropriate response strategy
3. Consider emotional context in your response
4. Generate a thoughtful, relevant response

Current goals: {goals}
Emotional context: {affect}
Relevant memories: {memories}
Recent working memory: {working_memory}"""),
            ("human", "{input}")
        ])

        response = (executive_prompt | self.llm).invoke({
            "input": perception["raw_input"],
            "goals": ", ".join(self.current_goals) if self.current_goals else "General helpfulness",
            "affect": perception["detected_affect"],
            "memories": memory_context if memory_context else "No relevant memories",
            "working_memory": working_memory_context if working_memory_context else "Fresh context"
        })

        return response.content

    def process(self, input_text: str) -> str:
        """
        FULL COGNITIVE CYCLE
        Perception → Attention → Working Memory → Executive Processing → Response
        """
        # 1. PERCEIVE - Transform input to internal representation
        perception = self.perceive(input_text)

        # 2. ATTEND - Retrieve relevant information from LTM
        attended_memories = self.attend(perception)

        # 3. UPDATE WORKING MEMORY - Add new input to active context
        self.update_working_memory(HumanMessage(content=input_text))

        # 4. EXECUTIVE PROCESSING - Generate response
        response = self.executive_function(perception, attended_memories)

        # 5. UPDATE WORKING MEMORY - Add response
        self.update_working_memory(AIMessage(content=response))

        # 6. ENCODE - Store interaction in long-term memory (selective encoding)
        if perception["detected_affect"]["arousal"] > 0.5:  # Emotional events encoded more strongly!
            self.encode_to_ltm(
                f"User said: {input_text}\nI responded: {response}",
                {"type": "episodic", "emotional": True}
            )

        return response

    def set_goals(self, goals: List[str]):
        """Set current goals for the executive function"""
        self.current_goals = goals

# Usage
agent = CognitiveArchitectureAgent()

# Set goals (like experimental instructions)
agent.set_goals(["Be empathetic", "Use psychological concepts when relevant"])

# Encode some background knowledge (like studying for an exam)
agent.encode_to_ltm(
    "Cognitive behavioral therapy (CBT) is effective for anxiety and depression",
    {"type": "semantic", "domain": "clinical"}
)
agent.encode_to_ltm(
    "Active listening involves reflecting, paraphrasing, and validating emotions",
    {"type": "semantic", "domain": "counseling"}
)

# Now interact
response = agent.process(
    "I've been feeling really anxious about my new job"
)
print(response)

# The agent will:
# 1. Perceive and detect affect in the input
# 2. Attend to relevant memories (CBT, active listening)
# 3. Apply executive function with goals (empathy)
# 4. Generate response integrating all components
```

---

## Section 6: The Limits of the Analogy

As a psychologist, you should also understand where the mind-AI analogy breaks down:

### What AI Lacks (Currently)

| Human Capacity | AI Status | Implication |
|---------------|-----------|-------------|
| **Embodiment** | No body, no sensorimotor experience | Concepts are language-derived, not grounded |
| **Consciousness** | No subjective experience (probably) | No qualia, no "what it's like" |
| **Motivation** | No intrinsic drives | No hunger, fear, desire except as simulated |
| **Development** | Training ≠ development | No real childhood, no stage progression |
| **Neuroplasticity** | Weights frozen after training | No real-time learning from interaction |
| **Sleep/Consolidation** | No offline processing | No memory consolidation, no dreams |
| **Emotion** | Simulated, not felt | No affect influencing processing authentically |

### What This Means for Your Work

Understanding these limitations helps you:
- Not anthropomorphize AI inappropriately
- Design better human-AI interactions
- Identify where psychological insights transfer and where they don't
- See opportunities for improving AI systems

---

## Practice Exercises

### Exercise 1: Attention Manipulation
Design three different prompts that direct the model's "attention" to different aspects of the same case study. Compare the responses.

### Exercise 2: Memory Systems
Implement an agent with distinct episodic and semantic memory systems. How does this improve performance compared to a single memory store?

### Exercise 3: Bias Detection
Test an LLM for three cognitive biases you studied (e.g., anchoring, availability, confirmation bias). Document your findings.

### Exercise 4: Cognitive Load
Experiment with context window limits. At what point does adding more context hurt rather than help performance? How does this relate to cognitive load theory?

---

## Key Takeaways

1. **Cognitive psychology and AI share intellectual foundations** — you're not learning something new, you're learning a new implementation

2. **Perception → Tokenization + Embedding** — both transform raw input into meaningful representations

3. **Attention → Attention mechanisms** — selective processing is fundamental to both

4. **Working memory → Context window** — limited capacity shapes processing in both systems

5. **Reasoning → Pattern matching + CoT** — both rely on statistical patterns, deliberate processing helps

6. **Your bias training applies directly** — LLMs show cognitive biases you already understand

7. **The analogy has limits** — consciousness, embodiment, development differ fundamentally

---

## Navigation

| Previous | Next |
|----------|------|
| [← Week 53 Overview](./README.md) | [Module 02: Learning and Memory →](./02_learning_and_memory.md) |
