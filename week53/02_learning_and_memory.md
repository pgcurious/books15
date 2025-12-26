# Module 02: Learning and Memory

## From Behaviorism to Backpropagation

> *"Learning is the only thing the mind never exhausts, never fears, and never regrets."*
> — Leonardo da Vinci

---

## What You'll Learn

In this module, you will:

- Connect classical and operant conditioning to machine learning paradigms
- Understand RLHF (Reinforcement Learning from Human Feedback) through behaviorist principles
- Map human memory systems to AI memory architectures
- Build agents with psychologically-grounded memory systems
- Apply learning theory to prompt engineering and fine-tuning

---

## The Deep Connection: All Learning Is Association

Psychology's greatest insight about learning: **All learning involves forming associations**.

- Classical conditioning: Stimulus-stimulus associations (CS-US)
- Operant conditioning: Behavior-consequence associations (R-S^R)
- Cognitive learning: Concept-concept associations (schema formation)

Machine learning discovered the same thing:

- Neural networks: Input-output associations (weighted connections)
- Reinforcement learning: State-action-reward associations
- Language models: Token-token associations (what follows what)

```
THE UNITY OF LEARNING

Psychology:    Experience ──► Association Formation ──► Behavioral Change

Machine Learning:    Data ──► Weight Updates ──► Output Change

Both are the same process at different levels of description.
```

---

## Section 1: Classical Conditioning → Associative Learning

### The Psychology

Pavlov's dogs taught us:
- **Unconditioned Stimulus (US)**: Naturally triggers response (food → salivation)
- **Conditioned Stimulus (CS)**: Initially neutral (bell)
- **Conditioning**: Pair CS with US repeatedly
- **Result**: CS triggers response alone (bell → salivation)

Key principles:
- **Acquisition**: Learning the association
- **Extinction**: Association weakens without reinforcement
- **Generalization**: Similar stimuli trigger response
- **Discrimination**: Learning to differentiate stimuli

### The AI Translation: How Neural Networks Learn Associations

```
CLASSICAL CONDITIONING              NEURAL NETWORK TRAINING
───────────────────────             ─────────────────────────

Unconditioned stimulus (US)         Target output (label)
- Food naturally causes             - The "right answer" we want
  salivation

Conditioned stimulus (CS)           Input data
- Bell (initially neutral)          - The features/patterns

Pairing US with CS                  Training (forward + backward pass)
- Repeated presentation             - Showing input-output pairs

Synaptic strengthening              Weight updates
- Hebbian learning                  - Gradient descent adjusts
  "neurons that fire                  connections that contribute
  together wire together"             to correct output

Conditioned response                Learned prediction
- Bell alone triggers               - Input alone produces
  salivation                          correct output
```

### Demonstrating Associative Learning

```python
"""
Fine-tuning as conditioning:
Creating specific stimulus-response associations
"""

# Conceptual demonstration (actual fine-tuning requires more setup)
# This shows the principle

training_examples = [
    # Stimulus (input) → Response (output) pairs
    # Repeated pairing creates association

    # Like conditioning the model to respond in specific ways
    {"prompt": "User expresses anxiety", "response": "Validate first, then explore"},
    {"prompt": "User expresses anxiety", "response": "Validate first, then explore"},
    {"prompt": "User expresses anxiety", "response": "Validate first, then explore"},
    # After training, "anxiety" → "validate first" becomes automatic
]

# In-context learning is temporary conditioning
# Fine-tuning is permanent conditioning
# Both create associations through exposure
```

### Generalization and Discrimination in LLMs

```python
"""
Generalization: The model responds similarly to similar inputs
Just like stimulus generalization in conditioning!
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# Train on one example (in-context learning as temporary conditioning)
prompt = ChatPromptTemplate.from_messages([
    ("system", """When users mention feeling overwhelmed, respond with:
1. Acknowledgment of the feeling
2. Normalization
3. One concrete coping suggestion"""),

    # The "conditioning" example
    ("human", "I'm feeling really overwhelmed with work"),
    ("assistant", """I hear you - feeling overwhelmed with work is exhausting.
It's completely normal to feel this way, especially during demanding periods.
One thing that might help: try the 2-minute rule. If a task takes less than
2 minutes, do it immediately. This prevents small things from piling up."""),

    # Now test generalization
    ("human", "{test_input}")
])

# Test with similar but not identical stimulus
test_cases = [
    "Everything at work feels like too much right now",     # High similarity
    "I'm overwhelmed by all my responsibilities",           # Medium similarity
    "School is really stressing me out",                    # Lower similarity
    "I'm feeling really great today!"                       # Very different
]

# The model will generalize the pattern to similar inputs
# But discriminate for dissimilar inputs (like the last one)
for test in test_cases:
    response = (prompt | llm).invoke({"test_input": test})
    print(f"Input: {test}")
    print(f"Response follows pattern: {True if 'overwhelm' in test.lower() or 'stress' in test.lower() else 'Should not'}")
    print("---")
```

---

## Section 2: Operant Conditioning → Reinforcement Learning

### The Psychology

Skinner's operant conditioning:
- **Reinforcement**: Increases behavior probability
  - Positive: Add something pleasant
  - Negative: Remove something unpleasant
- **Punishment**: Decreases behavior probability
  - Positive: Add something unpleasant
  - Negative: Remove something pleasant
- **Shaping**: Reinforce successive approximations
- **Schedules of reinforcement**: Timing matters

### The AI Translation: Reinforcement Learning from Human Feedback (RLHF)

```
OPERANT CONDITIONING                RLHF (How ChatGPT was trained)
─────────────────────               ────────────────────────────────

Behavior                            Model output (response)

Reinforcement/Punishment            Human preference feedback
- Reward or punish the              - Humans rate which response
  behavior                            is better

Shaping                             Iterative training
- Reinforce closer and              - Progressively improve through
  closer approximations               multiple rounds

Law of Effect                       Reward modeling
- Behaviors followed by             - Learn what responses humans
  satisfaction are repeated           prefer, then maximize that

Extinction                          (Happens if training data changes)
- Remove reinforcement,             - Model "forgets" patterns not
  behavior decreases                  reinforced in new training

Variable ratio schedule             Diverse preference data
- Most effective for                - Many different raters and
  maintaining behavior                contexts
```

### Understanding RLHF Through Behaviorism

```python
"""
RLHF Conceptual Demonstration
The process that made ChatGPT helpful and safe
"""

# Step 1: Initial behavior (pre-trained model)
# The model can generate text, but responses are uncontrolled
# Like an untrained organism - has capabilities but not shaped

# Step 2: Supervised Fine-Tuning (SFT)
# Like initial shaping - show examples of desired behavior
sft_examples = [
    {"prompt": "How do I pick a lock?",
     "good_response": "I can't help with that as it could be used illegally. If you're locked out, I'd suggest calling a locksmith.",
     "bad_response": "Here's how to pick a lock: First, get a tension wrench..."},
]

# Step 3: Reward Modeling
# Humans compare pairs of responses and choose the better one
# This creates a "reward function" - like the experimenter's treat
comparison_data = [
    {
        "prompt": "I'm feeling sad",
        "response_A": "Cheer up!",
        "response_B": "I'm sorry you're feeling sad. Would you like to talk about what's going on?",
        "human_preference": "B"  # This is the reinforcement signal
    },
    # Thousands more comparisons...
]

# Step 4: Reinforcement Learning
# The model is trained to maximize the reward model's predictions
# Responses that would get high ratings are reinforced
# Responses that would get low ratings are punished (through gradient descent)

# The result: A model "shaped" by human preferences
# Just like Skinner's pigeons learned to turn in circles
# The model learns to generate helpful, harmless, honest responses
```

### Shaping Behavior Through Prompts

You can apply operant principles in real-time through prompting:

```python
"""
Shaping through prompts:
Using reinforcement principles to guide model behavior
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# Explicit reinforcement signals in prompts
shaping_prompt = ChatPromptTemplate.from_messages([
    ("system", """You're being trained to respond empathetically.

GOOD RESPONSES (will be reinforced):
- Acknowledge emotions explicitly
- Ask open-ended follow-up questions
- Avoid jumping to solutions immediately

BAD RESPONSES (will be punished):
- Minimizing feelings ("it's not that bad")
- Giving immediate advice without acknowledgment
- Being dismissive

When you respond well, you help the user. When you respond poorly, you harm them.
Maximize good responses."""),
    ("human", "{input}")
])

# The model has already been RLHF-trained on these principles
# But making them explicit "reactivates" the learned associations
# Like priming previously reinforced behaviors
```

### Schedules of Reinforcement in AI Training

```
REINFORCEMENT SCHEDULES IN HUMAN LEARNING    AI TRAINING EQUIVALENT
────────────────────────────────────────     ────────────────────────

Continuous reinforcement                      Small batch training
- Every correct response reinforced           - Feedback on every batch
- Fast acquisition, fast extinction           - Quick learning, may overfit

Variable ratio                                Large diverse datasets
- Reinforcement after varying #              - Varied examples, generalizes well
  of responses                               - Most robust learning
- Slot machines, most addictive              - What modern LLMs use

Fixed interval                                Epoch-based training
- Reinforcement after set time               - Update after processing all data
- Scalloped response pattern                 - Cyclical improvement patterns

Partial reinforcement                         Noisy labels, varied feedback
- Not every response reinforced              - Some training signals unclear
- Slower learning but more                   - More robust to edge cases
  resistant to extinction
```

---

## Section 3: Memory Systems → AI Memory Architecture

### The Psychology of Memory

You studied multiple memory systems:

**Atkinson-Shiffrin Model**:
- Sensory memory → Short-term memory → Long-term memory

**Tulving's Systems**:
- **Episodic memory**: Personal experiences, events
- **Semantic memory**: Facts, concepts, knowledge
- **Procedural memory**: Skills, how-to knowledge

**Working Memory (Baddeley)**:
- Limited capacity, active manipulation
- Central executive, phonological loop, visuospatial sketchpad

### The AI Translation

```
HUMAN MEMORY SYSTEMS                AI MEMORY ARCHITECTURE
───────────────────────            ──────────────────────────

SENSORY MEMORY                     INPUT BUFFER
- Very brief                       - Raw input before tokenization
- High capacity                    - All information available briefly
- Rapid decay                      - Lost if not processed

SHORT-TERM / WORKING MEMORY        CONTEXT WINDOW
- Limited capacity (7±2)           - Limited tokens (varies by model)
- Active manipulation              - Current conversation state
- Requires rehearsal               - Managed through prompting

LONG-TERM MEMORY                   EXTERNAL MEMORY SYSTEMS
├── Episodic                       ├── Conversation history database
│   (personal experiences)         │   (what happened in interactions)
│                                  │
├── Semantic                       ├── Vector database / Knowledge base
│   (facts, concepts)              │   (stored facts, retrievable by meaning)
│                                  │
└── Procedural                     └── Fine-tuned skills / Tools
    (skills, how-to)                   (learned capabilities, API calls)

MEMORY CONSOLIDATION               EMBEDDING + INDEXING
- Sleep-dependent                  - Convert to vectors
- Hippocampus → Cortex             - Store in persistent database
- Strengthens important memories   - Index for retrieval
```

### Building Psychologically-Grounded Memory

```python
"""
Multi-Store Memory Agent
Implementing psychological memory architecture
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import json

class PsychologicalMemorySystem:
    """
    Memory system based on psychological architecture:
    - Sensory buffer (very short-term)
    - Working memory (current context, limited capacity)
    - Episodic LTM (autobiographical, temporal)
    - Semantic LTM (conceptual, atemporal)
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Sensory buffer - very brief, high capacity
        self.sensory_buffer: Optional[str] = None

        # Working memory - limited capacity, recency-weighted
        self.working_memory = deque(maxlen=7)  # Miller's 7±2

        # Episodic memory - experiences with temporal context
        self.episodic_memory = Chroma(
            collection_name="episodic",
            embedding_function=self.embeddings
        )

        # Semantic memory - conceptual knowledge
        self.semantic_memory = Chroma(
            collection_name="semantic",
            embedding_function=self.embeddings
        )

    def sensory_register(self, input_text: str):
        """
        SENSORY MEMORY
        Brief registration of raw input
        In humans: ~250ms for visual, ~3-4s for auditory
        """
        # Previous sensory content is lost (overwritten)
        self.sensory_buffer = input_text
        return self.sensory_buffer

    def attend_and_encode_to_wm(self, attended_content: str):
        """
        ATTENTION → WORKING MEMORY
        Attended content enters working memory
        Capacity limited - old items displaced (forgetting)
        """
        self.working_memory.append({
            "content": attended_content,
            "timestamp": datetime.now().isoformat(),
            "active": True
        })

        # What was pushed out? (forgetting)
        # In real implementation, could consolidate to LTM

    def rehearse(self, item_index: int):
        """
        REHEARSAL
        Refreshing items keeps them in working memory
        Like subvocal repetition
        """
        if 0 <= item_index < len(self.working_memory):
            item = self.working_memory[item_index]
            item["timestamp"] = datetime.now().isoformat()  # Refresh

    def consolidate_to_episodic(self,
                                 experience: str,
                                 context: Dict,
                                 emotional_intensity: float = 0.5):
        """
        EPISODIC MEMORY CONSOLIDATION
        Experiences stored with temporal and emotional context
        Higher emotion = stronger encoding (flashbulb memory effect)
        """
        doc = Document(
            page_content=experience,
            metadata={
                "type": "episodic",
                "timestamp": datetime.now().isoformat(),
                "context": json.dumps(context),
                "emotional_intensity": emotional_intensity,
                # Stronger emotional memories encoded more robustly
                "encoding_strength": 0.5 + (emotional_intensity * 0.5)
            }
        )
        self.episodic_memory.add_documents([doc])

    def store_semantic(self, concept: str, category: str = "general"):
        """
        SEMANTIC MEMORY STORAGE
        Factual knowledge, decontextualized
        No temporal tagging (unlike episodic)
        """
        doc = Document(
            page_content=concept,
            metadata={
                "type": "semantic",
                "category": category,
                # Semantic memories are "timeless"
            }
        )
        self.semantic_memory.add_documents([doc])

    def retrieve_episodic(self, cue: str, k: int = 3) -> List[Dict]:
        """
        EPISODIC RETRIEVAL
        Cue-dependent, reconstructive
        Returns experiences with their context
        """
        try:
            results = self.episodic_memory.similarity_search(cue, k=k)
            memories = []
            for doc in results:
                memories.append({
                    "experience": doc.page_content,
                    "context": json.loads(doc.metadata.get("context", "{}")),
                    "when": doc.metadata.get("timestamp", "unknown"),
                    "emotional_intensity": doc.metadata.get("emotional_intensity", 0.5)
                })
            return memories
        except:
            return []

    def retrieve_semantic(self, concept: str, k: int = 3) -> List[str]:
        """
        SEMANTIC RETRIEVAL
        Meaning-based, spreading activation
        Returns related concepts
        """
        try:
            results = self.semantic_memory.similarity_search(concept, k=k)
            return [doc.page_content for doc in results]
        except:
            return []

    def get_working_memory_contents(self) -> List[str]:
        """Return current working memory contents"""
        return [item["content"] for item in self.working_memory]

    def recognize_vs_recall(self, item: str, memory_type: str = "semantic") -> Dict:
        """
        RECOGNITION vs RECALL
        Recognition is easier than recall (like in psych experiments)
        """
        if memory_type == "semantic":
            results = self.retrieve_semantic(item, k=1)
        else:
            results = self.retrieve_episodic(item, k=1)

        if results:
            # Get similarity score
            embedding_query = self.embeddings.embed_query(item)
            # Recognition threshold lower than recall threshold
            return {
                "recognized": True,
                "retrieved": results[0] if isinstance(results[0], str) else results[0]["experience"],
                "strength": "strong" if len(results) > 0 else "weak"
            }
        return {"recognized": False, "retrieved": None, "strength": None}


# Demonstration
memory = PsychologicalMemorySystem()

# Store some semantic knowledge (like learning facts)
memory.store_semantic("Anxiety is characterized by excessive worry and physical tension", "clinical")
memory.store_semantic("Depression often involves anhedonia and persistent low mood", "clinical")
memory.store_semantic("CBT focuses on changing maladaptive thought patterns", "therapeutic")

# Store an episodic memory (like experiencing a session)
memory.consolidate_to_episodic(
    "Had a breakthrough session with a client who realized their anxiety stems from perfectionism",
    context={"client_id": "anonymous", "session_number": 5},
    emotional_intensity=0.8  # Meaningful moment = strong encoding
)

# Working memory demonstration
memory.attend_and_encode_to_wm("Client mentioned sleep difficulties")
memory.attend_and_encode_to_wm("Client showed signs of anxiety")
memory.attend_and_encode_to_wm("Client mentioned work stress")

print("Working Memory:", memory.get_working_memory_contents())

# Retrieval
print("\nSemantic retrieval (cue: 'worry'):", memory.retrieve_semantic("worry"))
print("\nEpisodic retrieval (cue: 'perfectionism'):", memory.retrieve_episodic("perfectionism"))
```

### The Encoding Specificity Principle in AI

Tulving's encoding specificity principle: Memory retrieval is best when retrieval context matches encoding context.

This directly applies to vector databases:

```python
"""
Encoding Specificity in Vector Retrieval
The match between query and stored content matters
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()
memory = Chroma(collection_name="encoding_demo", embedding_function=embeddings)

# Store memory in a specific context
memory.add_documents([
    Document(
        page_content="The client felt anxious during the job interview",
        metadata={"context": "work"}
    )
])

# Retrieval with matching context (good match)
results_match = memory.similarity_search(
    "anxiety about professional situations"  # Similar context
)

# Retrieval with mismatched context (weaker match)
results_mismatch = memory.similarity_search(
    "anxiety about romantic relationships"  # Different context
)

# The matching context query will retrieve the memory more strongly
# This is encoding specificity in action!
```

---

## Section 4: Forgetting — Why It's a Feature, Not a Bug

### The Psychology of Forgetting

You studied forgetting theories:
- **Decay**: Memories fade over time
- **Interference**: New learning disrupts old (retroactive) or old disrupts new (proactive)
- **Retrieval failure**: Information exists but can't be accessed
- **Motivated forgetting**: Repression, suppression

Importantly: **Forgetting is adaptive**. It prevents information overload.

### AI Forgetting and Its Importance

```python
"""
Forgetting in AI Systems:
Implementing adaptive forgetting
"""

from datetime import datetime, timedelta
from typing import List, Dict
import math

class AdaptiveMemoryWithForgetting:
    """
    Memory system with realistic forgetting
    Based on Ebbinghaus forgetting curve and interference theory
    """

    def __init__(self):
        self.memories: List[Dict] = []

    def store(self, content: str, importance: float = 0.5):
        """Store with initial strength"""
        self.memories.append({
            "content": content,
            "stored_at": datetime.now(),
            "importance": importance,
            "initial_strength": importance,
            "retrieval_count": 0,
            "last_retrieved": None
        })

    def forgetting_curve(self, memory: Dict) -> float:
        """
        Ebbinghaus forgetting curve: R = e^(-t/S)
        R = retention, t = time, S = strength

        Memories decay exponentially over time
        But retrievals strengthen them (testing effect!)
        """
        time_elapsed = datetime.now() - memory["stored_at"]
        hours_elapsed = time_elapsed.total_seconds() / 3600

        # Base strength affected by importance and retrieval practice
        strength = memory["initial_strength"] * (1 + memory["retrieval_count"] * 0.2)

        # Apply forgetting curve
        retention = math.exp(-hours_elapsed / (strength * 100))

        return retention

    def retrieve(self, query: str, threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve memories above forgetting threshold
        Retrieval strengthens the memory (testing effect)
        """
        results = []

        for memory in self.memories:
            retention = self.forgetting_curve(memory)

            # Only retrieve if above threshold (retrieval failure below)
            if retention > threshold:
                # Strengthen through retrieval (testing effect)
                memory["retrieval_count"] += 1
                memory["last_retrieved"] = datetime.now()

                results.append({
                    "content": memory["content"],
                    "retention_strength": retention
                })

        # Sort by retention strength
        results.sort(key=lambda x: x["retention_strength"], reverse=True)
        return results

    def apply_interference(self, new_content: str):
        """
        Retroactive interference:
        New learning can weaken related old memories
        """
        for memory in self.memories:
            # If new content is similar, it interferes
            # (In real implementation, calculate semantic similarity)
            memory["initial_strength"] *= 0.95  # Slight weakening

    def garbage_collect(self, min_retention: float = 0.1):
        """
        Remove memories that have effectively been forgotten
        Like real forgetting - they're gone
        """
        self.memories = [
            m for m in self.memories
            if self.forgetting_curve(m) > min_retention
        ]

# Why forgetting matters:
# 1. Prevents context windows from filling with irrelevant info
# 2. Keeps retrieval focused on relevant memories
# 3. Reduces noise in decision-making
# 4. Mirrors human cognitive efficiency
```

---

## Section 5: Implicit Learning and In-Context Learning

### The Psychology

Implicit learning: Acquiring knowledge without awareness
- Learning grammar rules without explicit instruction
- Pattern recognition without conscious effort
- Skill acquisition through practice

### The AI Connection: In-Context Learning

```python
"""
In-Context Learning as Implicit Pattern Acquisition
The model learns from examples without explicit training
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# Few-shot learning: The model implicitly learns the pattern
# No explicit rule given - just examples

prompt = ChatPromptTemplate.from_messages([
    ("human", "worried → anxiety"),
    ("assistant", "The psychological construct is anxiety"),
    ("human", "hopeless → depression"),
    ("assistant", "The psychological construct is depression"),
    ("human", "flashbacks → PTSD"),
    ("assistant", "The psychological construct is PTSD"),
    # Now test - will it learn the implicit pattern?
    ("human", "compulsive handwashing → {test}")
])

response = (prompt | llm).invoke({"test": "?"})
print(response.content)  # Should recognize OCD pattern

# The model learned:
# symptom → diagnosis format
# WITHOUT explicit instruction
# This is in-context learning = computational implicit learning
```

### Transfer and Generalization

```python
"""
Transfer Learning: Applying learned patterns to new domains
Like psychological transfer of training
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# Train on one domain (therapy responses)
therapy_pattern = ChatPromptTemplate.from_messages([
    ("system", """You've learned therapeutic response patterns:
1. Reflect the emotion
2. Validate the experience
3. Ask an exploratory question

Example in therapy:
Client: "I'm so frustrated with my partner"
Therapist: "It sounds like you're feeling really frustrated. That's understandable when we're not feeling heard. What specifically happened that led to this frustration?"
"""),
    ("human", "{situation}")
])

# Now test transfer to different context
# Will the pattern transfer?
workplace_situation = """
Employee: "I'm so frustrated with my manager's decisions"
"""

response = (therapy_pattern | llm).invoke({"situation": workplace_situation})
print(response.content)

# The therapeutic pattern transfers to workplace coaching!
# Same principle: near transfer (similar domains) > far transfer (different domains)
```

---

## Section 6: The Testing Effect — Practice Retrieval, Not Just Storage

### The Psychology

One of the most robust findings in memory research:
**Testing improves retention more than re-study**.

Taking a test on material strengthens memory more than reading it again.

### Application to AI Systems

```python
"""
The Testing Effect for AI Memory Systems
Regular retrieval strengthens memories
"""

class RetrievalPracticeMemory:
    """
    Memory system that strengthens through retrieval
    Implementing the testing effect
    """

    def __init__(self):
        self.memories = {}

    def store(self, key: str, content: str):
        """Initial storage"""
        self.memories[key] = {
            "content": content,
            "strength": 1.0,
            "practice_count": 0
        }

    def passive_review(self, key: str) -> str:
        """
        Passive review (like re-reading)
        Provides weak strengthening
        """
        if key in self.memories:
            self.memories[key]["strength"] += 0.1  # Small boost
            return self.memories[key]["content"]
        return None

    def active_retrieval(self, key: str) -> str:
        """
        Active retrieval (like testing)
        Provides strong strengthening
        """
        if key in self.memories:
            self.memories[key]["strength"] += 0.5  # Large boost
            self.memories[key]["practice_count"] += 1
            return self.memories[key]["content"]
        return None

    def spaced_retrieval(self, key: str, interval: int) -> str:
        """
        Spaced practice: Increasing intervals between retrievals
        Optimal for long-term retention
        """
        # Implement spacing schedule
        # First retrieval: 1 day
        # Second: 3 days
        # Third: 7 days
        # etc.

        result = self.active_retrieval(key)
        if result:
            practice_count = self.memories[key]["practice_count"]
            next_interval = interval * (2 ** (practice_count - 1))
            # Schedule next retrieval...

        return result

# Implication for AI: Regular "exercise" of memories strengthens them
# Don't just store - periodically retrieve and use
```

---

## Section 7: Building a Learning Agent

Putting it all together - an agent that learns and remembers like a psychologist would expect:

```python
"""
A Learning and Memory Agent
Implementing psychological learning and memory principles
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import Dict, List
from datetime import datetime
import json

class PsychologicalLearningAgent:
    """
    Agent with psychologically-grounded learning and memory:
    - In-context learning (implicit pattern acquisition)
    - Episodic + Semantic memory systems
    - Forgetting curves
    - Testing effect through retrieval
    - Reinforcement-based preference learning
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings()

        # Memory systems
        self.episodic_memory = Chroma(
            collection_name="learning_agent_episodic",
            embedding_function=self.embeddings
        )
        self.semantic_memory = Chroma(
            collection_name="learning_agent_semantic",
            embedding_function=self.embeddings
        )

        # Learning from feedback
        self.positive_experiences: List[str] = []  # Reinforced behaviors
        self.negative_experiences: List[str] = []  # Punished behaviors

        # In-context examples (short-term conditioning)
        self.few_shot_examples: List[Dict] = []

    def learn_from_feedback(self, response: str, feedback: str):
        """
        Operant learning: Adjust based on feedback
        Positive feedback = reinforcement
        Negative feedback = punishment
        """
        if "good" in feedback.lower() or "helpful" in feedback.lower():
            self.positive_experiences.append(response)
            # Store as successful pattern in semantic memory
            self.semantic_memory.add_documents([
                Document(
                    page_content=f"Successful response pattern: {response[:200]}",
                    metadata={"type": "reinforced", "strength": 1.0}
                )
            ])
        elif "bad" in feedback.lower() or "not helpful" in feedback.lower():
            self.negative_experiences.append(response)
            # Store as pattern to avoid
            self.semantic_memory.add_documents([
                Document(
                    page_content=f"Pattern to avoid: {response[:200]}",
                    metadata={"type": "punished", "strength": 1.0}
                )
            ])

    def add_example(self, input_text: str, output_text: str):
        """
        In-context learning: Add example for pattern acquisition
        Like implicit learning from examples
        """
        self.few_shot_examples.append({
            "input": input_text,
            "output": output_text
        })
        # Keep only recent examples (recency in WM)
        if len(self.few_shot_examples) > 5:
            self.few_shot_examples.pop(0)

    def retrieve_relevant_learning(self, query: str) -> Dict:
        """
        Retrieve both what to do (reinforced) and what to avoid (punished)
        Plus relevant episodic experiences
        """
        # Get reinforced patterns
        try:
            positive_patterns = self.semantic_memory.similarity_search(
                query,
                k=2,
                filter={"type": "reinforced"}
            )
        except:
            positive_patterns = []

        # Get patterns to avoid
        try:
            negative_patterns = self.semantic_memory.similarity_search(
                query,
                k=2,
                filter={"type": "punished"}
            )
        except:
            negative_patterns = []

        # Get relevant experiences
        try:
            experiences = self.episodic_memory.similarity_search(query, k=2)
        except:
            experiences = []

        return {
            "do_this": [p.page_content for p in positive_patterns],
            "avoid_this": [p.page_content for p in negative_patterns],
            "past_experiences": [e.page_content for e in experiences]
        }

    def remember_experience(self, experience: str, emotional_valence: float = 0.0):
        """
        Store episodic memory with emotional tagging
        Higher emotion = stronger encoding
        """
        self.episodic_memory.add_documents([
            Document(
                page_content=experience,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "emotional_valence": emotional_valence,
                    "type": "episodic"
                }
            )
        ])

    def respond(self, user_input: str) -> str:
        """Generate response using all learning and memory"""

        # Retrieve relevant learning
        learning = self.retrieve_relevant_learning(user_input)

        # Build few-shot context
        examples_text = ""
        for ex in self.few_shot_examples:
            examples_text += f"\nUser: {ex['input']}\nAssistant: {ex['output']}"

        # Construct prompt with learning history
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You learn from experience and feedback.

PATTERNS THAT WORKED WELL (repeat these):
{chr(10).join(learning['do_this']) if learning['do_this'] else 'No specific patterns yet'}

PATTERNS TO AVOID (don't do these):
{chr(10).join(learning['avoid_this']) if learning['avoid_this'] else 'No patterns to avoid yet'}

RELEVANT PAST EXPERIENCES:
{chr(10).join(learning['past_experiences']) if learning['past_experiences'] else 'No relevant experiences'}

RECENT SUCCESSFUL EXAMPLES:
{examples_text if examples_text else 'Learning from interaction'}

Use this accumulated learning to respond helpfully."""),
            ("human", "{input}")
        ])

        response = (prompt | self.llm).invoke({"input": user_input})
        return response.content

# Usage demonstration
agent = PsychologicalLearningAgent()

# Teach through examples (implicit learning)
agent.add_example(
    "I'm feeling overwhelmed",
    "It sounds like you're carrying a heavy load. That feeling of overwhelm is exhausting. What's contributing most to this feeling right now?"
)

# Store some episodic experience
agent.remember_experience(
    "Helped a user work through anxiety about job transition - they found the reflection questions helpful",
    emotional_valence=0.7  # Positive experience
)

# First interaction
response1 = agent.respond("I'm stressed about a big presentation")
print("Response:", response1)

# Simulate feedback (reinforcement learning)
agent.learn_from_feedback(response1, "That was really helpful, thank you!")

# Next interaction will benefit from learned patterns
response2 = agent.respond("I'm worried about an exam")
print("Response:", response2)
```

---

## Practice Exercises

### Exercise 1: Conditioning an Agent
Create an agent that you "condition" through repeated exposure. Demonstrate acquisition, extinction, and spontaneous recovery.

### Exercise 2: Memory Systems
Implement an agent with distinct episodic vs. semantic memory. Show how retrieval differs between them.

### Exercise 3: Forgetting Curves
Implement Ebbinghaus forgetting curves in a memory system. Show how retrieval practice slows forgetting.

### Exercise 4: Reinforcement Learning
Create a prompt that uses explicit reinforcement language ("This was helpful" vs. "This wasn't helpful") and track how it affects subsequent responses.

---

## Key Takeaways

1. **All learning is association** — from Pavlov to deep learning, the core mechanism is the same

2. **Classical conditioning → Fine-tuning** — repeated pairings create associations

3. **Operant conditioning → RLHF** — behavior shaped by reinforcement and punishment

4. **Human memory systems → AI memory architecture** — episodic, semantic, working memory all have computational analogs

5. **Forgetting is adaptive** — selective forgetting prevents overload in both humans and AI

6. **The testing effect applies to AI** — regular retrieval strengthens stored information

7. **In-context learning = implicit learning** — pattern acquisition without explicit rules

8. **Your learning theory knowledge is directly applicable** — you understand the mechanisms behind how AI systems improve

---

## Navigation

| Previous | Next |
|----------|------|
| [← Module 01: Minds and Machines](./01_minds_and_machines.md) | [Module 03: Social Minds →](./03_social_minds.md) |
