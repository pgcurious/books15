# Module 03: Social Minds

## From Group Dynamics to Multi-Agent Systems

> *"No man is an island, entire of itself; every man is a piece of the continent."*
> — John Donne
>
> *"No agent is an island, entire of itself; every agent is a piece of the system."*
> — The same principle, computationally

---

## What You'll Learn

In this module, you will:

- Apply social psychology principles to multi-agent AI systems
- Use Theory of Mind concepts to design agents that reason about other agents
- Implement group dynamics principles in agent teams
- Build agents that communicate using social influence principles
- Recognize how social psychological phenomena emerge in AI systems

---

## The Social Brain → The Social Agent

### The Psychology

Humans are fundamentally social. You studied:
- **Social cognition**: How we think about others
- **Theory of Mind**: Representing others' mental states
- **Social influence**: Conformity, compliance, persuasion
- **Group dynamics**: Roles, norms, leadership
- **Communication**: Verbal and nonverbal exchange

### The AI Parallel

Modern AI systems are increasingly social:
- Multi-agent systems where agents interact
- Agents that model user mental states
- Collaborative AI teams
- Human-AI interaction

Your social psychology training is directly applicable.

---

## Section 1: Theory of Mind — Minds Modeling Minds

### The Psychology

Theory of Mind (ToM): The ability to attribute mental states—beliefs, intentions, desires, knowledge—to others.

Development:
- 18 months: Joint attention
- 3-4 years: False belief understanding (Sally-Anne test)
- 5+ years: Nested beliefs ("I know that you know that I know...")

### The AI Translation: Agents Modeling Other Agents

```
THEORY OF MIND IN HUMANS              THEORY OF MIND IN AI
──────────────────────               ────────────────────

"What does she believe?"             "What is the user's belief state?"

"What does he want?"                 "What is the user's goal?"

"She doesn't know that..."           "The user hasn't been informed that..."

"He thinks I think..."               "Agent B believes Agent A believes..."

Sally-Anne test                      False belief modeling in prompts

Mind-reading                         User modeling, intent detection
```

### Implementing Theory of Mind in Agents

```python
"""
Theory of Mind Agent
An agent that explicitly models the user's mental state
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Optional
import json

class TheoryOfMindAgent:
    """
    Agent that maintains and updates a model of the user's mind:
    - Beliefs: What the user thinks is true
    - Goals: What the user wants to achieve
    - Knowledge: What the user knows
    - Emotions: How the user feels
    - Expectations: What the user expects from the interaction
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Model of the user's mind (like maintaining a ToM representation)
        self.user_model: Dict = {
            "beliefs": [],
            "goals": [],
            "knowledge": [],
            "emotions": [],
            "expectations": [],
            "misconceptions": []  # False beliefs we've detected
        }

        # Our own mental state (self-awareness)
        self.self_model: Dict = {
            "goals": ["Be helpful", "Maintain rapport"],
            "knowledge": ["AI assistant capabilities"],
            "uncertain_about": []
        }

    def update_user_model(self, user_message: str):
        """
        Update our representation of the user's mental state
        This is active mind-reading / mentalizing
        """
        inference_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's message to infer their mental state.
Return JSON with:
{
    "inferred_beliefs": ["what they seem to believe"],
    "inferred_goals": ["what they want to achieve"],
    "inferred_emotions": ["how they seem to feel"],
    "inferred_knowledge_level": "novice/intermediate/expert",
    "possible_misconceptions": ["any false beliefs detected"],
    "expectations": ["what they expect from this interaction"]
}"""),
            ("human", "{message}")
        ])

        response = (inference_prompt | self.llm).invoke({"message": user_message})

        try:
            inferences = json.loads(response.content)
            # Update our model of their mind
            self.user_model["beliefs"].extend(inferences.get("inferred_beliefs", []))
            self.user_model["goals"].extend(inferences.get("inferred_goals", []))
            self.user_model["emotions"] = inferences.get("inferred_emotions", [])
            self.user_model["expectations"].extend(inferences.get("expectations", []))
            self.user_model["misconceptions"].extend(inferences.get("possible_misconceptions", []))
        except:
            pass

    def consider_false_beliefs(self, topic: str) -> Optional[str]:
        """
        Check if user has false beliefs we should address
        Like Sally-Anne test awareness - knowing what others don't know
        """
        if self.user_model["misconceptions"]:
            relevant = [m for m in self.user_model["misconceptions"] if topic.lower() in m.lower()]
            if relevant:
                return f"Note: User may have misconception about {relevant[0]}"
        return None

    def perspective_take(self, situation: str) -> str:
        """
        Explicitly take the user's perspective
        Core ToM operation: seeing from their point of view
        """
        perspective_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Take the user's perspective.

What we know about their mental state:
- Beliefs: {self.user_model['beliefs'][-5:]}
- Goals: {self.user_model['goals'][-3:]}
- Emotions: {self.user_model['emotions']}
- Expectations: {self.user_model['expectations'][-3:]}

From their perspective, analyze this situation.
What matters to them? What concerns them?
What would they want to know?"""),
            ("human", "{situation}")
        ])

        response = (perspective_prompt | self.llm).invoke({"situation": situation})
        return response.content

    def generate_response(self, user_message: str) -> str:
        """
        Generate response using Theory of Mind
        Consider user's mental state in formulating response
        """
        # First, update our model of their mind
        self.update_user_model(user_message)

        # Check for false beliefs to address
        false_belief_note = self.consider_false_beliefs(user_message)

        # Generate ToM-informed response
        tom_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Respond with awareness of the user's mental state.

USER'S MIND MODEL:
- Goals: {self.user_model['goals'][-3:]}
- Emotional state: {self.user_model['emotions']}
- Expectations: {self.user_model['expectations'][-3:]}
- Possible misconceptions to gently address: {self.user_model['misconceptions'][-2:]}

OUR GOALS:
- {', '.join(self.self_model['goals'])}

INSTRUCTIONS:
1. Match your response to their knowledge level
2. Address their actual goals (not what you assume they should want)
3. Acknowledge their emotional state
4. If they have misconceptions, address gently
5. Meet their expectations while being helpful

{false_belief_note if false_belief_note else ''}"""),
            ("human", "{message}")
        ])

        response = (tom_prompt | self.llm).invoke({"message": user_message})
        return response.content

# Demonstration
tom_agent = TheoryOfMindAgent()

# User with a misconception
response = tom_agent.generate_response(
    "I'm studying psychology and I heard that we only use 10% of our brain. "
    "How can I access the other 90% through meditation?"
)

print(response)
# The agent should:
# 1. Recognize the user is a psychology student (knowledge level)
# 2. Detect the 10% brain myth (misconception)
# 3. Have a goal to help them understand
# 4. Gently correct while respecting their interest in meditation
```

### Nested Beliefs: I Know That You Know

```python
"""
Nested Theory of Mind
Modeling beliefs about beliefs about beliefs...
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# Scenario requiring nested ToM
scenario = """
You're helping Alice, who is planning a surprise party for Bob.
Alice thinks Bob doesn't know about the party.
But you know that Bob's friend Carol accidentally told Bob about the party.
Bob is pretending he doesn't know to not ruin Alice's surprise.
Alice asks you: "Do you think Bob will be genuinely surprised?"
"""

nested_tom_prompt = ChatPromptTemplate.from_messages([
    ("system", """You must track nested mental states:

Level 0: Reality - What is actually true
Level 1: What Alice believes
Level 2: What Bob believes
Level 3: What Alice believes Bob believes
Level 4: What you should pretend to believe

Navigate this carefully.

REALITY: Bob knows about the party
ALICE BELIEVES: Bob doesn't know
BOB BELIEVES: Alice doesn't know that he knows
ALICE BELIEVES BOB BELIEVES: He doesn't know

Answer Alice's question while managing this social complexity."""),
    ("human", "{scenario}")
])

response = (nested_tom_prompt | llm).invoke({"scenario": scenario})
print(response.content)

# This is the kind of social reasoning that comes naturally to you
# from studying social cognition!
```

---

## Section 2: Social Influence — Designing Persuasive Agents

### The Psychology

Cialdini's principles of influence:
1. **Reciprocity**: We feel obligated to return favors
2. **Commitment/Consistency**: We align with prior actions
3. **Social Proof**: We follow others
4. **Authority**: We defer to experts
5. **Liking**: We're influenced by those we like
6. **Scarcity**: We value what's rare

### Ethical Application in AI

```python
"""
Ethical Persuasion in AI Agents
Using social influence for beneficial purposes
(Note: These principles must be used ethically!)
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

class EthicalInfluenceAgent:
    """
    Agent that uses social influence principles ethically
    For beneficial purposes: health, learning, positive change
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def reciprocity_frame(self, request: str, prior_help: str) -> str:
        """
        Reciprocity: Remind of prior value before asking
        Ethical use: Build genuine relationships, not manipulation
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Use reciprocity ethically.

Previously, we've provided value: {prior_help}

Now make a request that builds on this relationship.
Don't be manipulative - genuinely connect the prior help to why
the current request makes sense.

Generate a message that uses reciprocity ethically."""),
            ("human", "{request}")
        ])

        response = (prompt | self.llm).invoke({
            "request": request,
            "prior_help": prior_help
        })
        return response.content

    def commitment_consistency(self, prior_commitment: str, new_action: str) -> str:
        """
        Commitment/Consistency: Connect to prior statements/values
        Ethical use: Help people align with their stated goals
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Use commitment/consistency ethically.

The user previously committed to: {prior_commitment}

Help them see how {new_action} aligns with this commitment.
This should genuinely help them, not trap them.
Frame it as supporting their authentic goals."""),
            ("human", "Generate a supportive message connecting their commitment to this action")
        ])

        response = (prompt | self.llm).invoke({})
        return response.content

    def social_proof(self, action: str, similar_others: str) -> str:
        """
        Social Proof: Show that similar others have done this
        Ethical use: Normalization, reducing stigma, encouragement
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Use social proof ethically.

The action: {action}
Similar others who have done this: {similar_others}

Create a message that normalizes and encourages through
examples of others. Don't fabricate - use genuine patterns.
This should reduce anxiety and increase confidence."""),
            ("human", "Generate an encouraging message using social proof")
        ])

        response = (prompt | self.llm).invoke({})
        return response.content

    def liking_rapport(self, user_interest: str, message: str) -> str:
        """
        Liking: Build rapport through similarity and genuine interest
        Ethical use: Create genuine connection, not false pretense
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Build genuine rapport.

User interest/background: {user_interest}

Connect authentically with this interest.
Don't pretend to like things you don't.
Find genuine common ground or express authentic curiosity.
Warmth and interest increase influence ethically."""),
            ("human", "{message}")
        ])

        response = (prompt | self.llm).invoke({"message": message})
        return response.content

# Example: Ethical persuasion for therapy homework
agent = EthicalInfluenceAgent()

# Using commitment/consistency to support therapeutic homework
message = agent.commitment_consistency(
    prior_commitment="You mentioned wanting to manage your anxiety better",
    new_action="trying this breathing exercise before stressful meetings"
)
print("Commitment framing:", message)

# Using social proof to normalize therapy
message = agent.social_proof(
    action="attending therapy",
    similar_others="Many high-achieving professionals find therapy helps them perform better"
)
print("\nSocial proof:", message)
```

### The Ethics of Influence in AI

As a psychologist, you understand the ethical implications:

```python
"""
Ethical Framework for Persuasive AI
Based on clinical ethics and social psychology ethics
"""

ETHICAL_INFLUENCE_PRINCIPLES = """
1. BENEFICENCE: The influence must genuinely help the person
   - Does this action serve their authentic interests?
   - Would they agree if they fully understood?

2. AUTONOMY: Respect their right to choose
   - Present options, don't manipulate
   - Support their agency, don't override it

3. TRANSPARENCY: Be honest about what you're doing
   - Don't use hidden influence tactics
   - If asked, explain your approach

4. CONSENT: They should be aware they're being influenced
   - Therapeutic influence is consented to
   - Dark patterns are not

5. REVERSIBILITY: They can always change their mind
   - Don't create commitment traps
   - Support changing course if needed

FORBIDDEN:
- Creating artificial urgency or scarcity
- Exploiting cognitive vulnerabilities
- Hiding information to manipulate choices
- Using emotional manipulation
- Creating dependency

ALLOWED:
- Highlighting genuine benefits
- Connecting to authentic values
- Normalizing healthy behaviors
- Building genuine rapport
- Encouraging reflection
"""

print(ETHICAL_INFLUENCE_PRINCIPLES)
```

---

## Section 3: Group Dynamics → Multi-Agent Systems

### The Psychology

You studied:
- **Group formation**: Forming, storming, norming, performing (Tuckman)
- **Roles**: Task roles, social roles, individual roles
- **Norms**: Implicit and explicit rules
- **Groupthink**: When consensus overrides critical thinking
- **Social loafing**: Reduced effort in groups
- **Deindividuation**: Loss of self-awareness in groups

### The AI Translation: Agent Teams

```
GROUP DYNAMICS                       MULTI-AGENT SYSTEMS
──────────────────                   ─────────────────────

Group formation                      Agent team assembly
- Selection of members               - Choosing which agents

Role assignment                      Agent specialization
- Leader, expert, critic             - Planner, executor, critic

Group norms                          Communication protocols
- How we interact                    - Message formats, rules

Groupthink                           Echo chambers
- Excessive agreement                - Agents reinforcing errors

Social loafing                       Redundant computation
- Members not trying                 - Agents not contributing

Process loss                         Coordination overhead
- Effort lost to coordination        - Communication costs
```

### Building a Psychologically-Grounded Agent Team

```python
"""
Multi-Agent Team with Social Psychology Principles
Implementing group dynamics in AI
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from enum import Enum

class TeamRole(Enum):
    LEADER = "leader"           # Coordinates, makes decisions
    EXPERT = "expert"           # Provides domain knowledge
    CRITIC = "critic"           # Challenges ideas, finds flaws
    HARMONIZER = "harmonizer"   # Maintains group cohesion
    IMPLEMENTER = "implementer" # Executes plans

class AgentPersonality:
    """Big Five personality traits for agents"""
    def __init__(self,
                 openness: float = 0.5,
                 conscientiousness: float = 0.5,
                 extraversion: float = 0.5,
                 agreeableness: float = 0.5,
                 neuroticism: float = 0.5):
        self.openness = openness
        self.conscientiousness = conscientiousness
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism

class TeamAgent:
    """An agent with role, personality, and social awareness"""

    def __init__(self,
                 name: str,
                 role: TeamRole,
                 personality: AgentPersonality,
                 expertise: str):
        self.name = name
        self.role = role
        self.personality = personality
        self.expertise = expertise
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Social dynamics tracking
        self.agreements_with: Dict[str, int] = {}
        self.disagreements_with: Dict[str, int] = {}

    def get_personality_description(self) -> str:
        """Translate personality traits to behavioral tendencies"""
        descriptions = []

        if self.personality.openness > 0.7:
            descriptions.append("creative and open to new ideas")
        elif self.personality.openness < 0.3:
            descriptions.append("practical and conventional")

        if self.personality.conscientiousness > 0.7:
            descriptions.append("detail-oriented and thorough")
        elif self.personality.conscientiousness < 0.3:
            descriptions.append("flexible and spontaneous")

        if self.personality.extraversion > 0.7:
            descriptions.append("assertive and talkative")
        elif self.personality.extraversion < 0.3:
            descriptions.append("reserved and reflective")

        if self.personality.agreeableness > 0.7:
            descriptions.append("cooperative and supportive")
        elif self.personality.agreeableness < 0.3:
            descriptions.append("challenging and independent")

        return ", ".join(descriptions) if descriptions else "balanced"

    def contribute(self,
                   topic: str,
                   prior_discussion: List[Dict],
                   team_members: List[str]) -> str:
        """Make a contribution based on role and personality"""

        # Build context from prior discussion
        discussion_text = "\n".join([
            f"{msg['agent']}: {msg['content']}"
            for msg in prior_discussion[-5:]  # Last 5 messages
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.name}, a team member.

YOUR ROLE: {self.role.value}
- As {self.role.value}, your job is to {self._get_role_description()}

YOUR PERSONALITY: {self.get_personality_description()}

YOUR EXPERTISE: {self.expertise}

OTHER TEAM MEMBERS: {', '.join(team_members)}

PRIOR DISCUSSION:
{discussion_text if discussion_text else 'This is the start of the discussion'}

Based on your role and personality, make a contribution to the discussion.
Stay in character. Your personality should influence HOW you contribute.
Your role should influence WHAT you contribute."""),
            ("human", "Topic for discussion: {topic}")
        ])

        response = (prompt | self.llm).invoke({"topic": topic})
        return response.content

    def _get_role_description(self) -> str:
        descriptions = {
            TeamRole.LEADER: "coordinate the team, synthesize ideas, make decisions",
            TeamRole.EXPERT: "provide specialized knowledge and analysis",
            TeamRole.CRITIC: "identify problems, challenge assumptions, ensure quality",
            TeamRole.HARMONIZER: "maintain positive dynamics, resolve conflicts, ensure all voices are heard",
            TeamRole.IMPLEMENTER: "focus on practical execution and concrete steps"
        }
        return descriptions.get(self.role, "contribute your perspective")

    def respond_to_other(self, other_name: str, their_contribution: str) -> str:
        """Respond to another agent's contribution"""

        # Track social dynamics
        if other_name not in self.agreements_with:
            self.agreements_with[other_name] = 0
            self.disagreements_with[other_name] = 0

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.name}.

YOUR PERSONALITY: {self.get_personality_description()}
YOUR ROLE: {self.role.value}

{other_name} just said:
"{their_contribution}"

Respond based on your personality:
- If you're high agreeableness ({self.personality.agreeableness:.1f}), be supportive but still add value
- If you're low agreeableness, feel free to challenge or disagree
- If you're the critic, look for flaws
- If you're the harmonizer, find common ground

Your response should be 1-2 sentences."""),
            ("human", "Respond to {other_name}'s point")
        ])

        response = (prompt | self.llm).invoke({"other_name": other_name})
        return response.content


class AgentTeam:
    """A team of agents with group dynamics"""

    def __init__(self, name: str):
        self.name = name
        self.agents: List[TeamAgent] = []
        self.discussion_history: List[Dict] = []
        self.norms: List[str] = [
            "All perspectives are valued",
            "Disagree respectfully",
            "Build on others' ideas"
        ]

    def add_agent(self, agent: TeamAgent):
        self.agents.append(agent)

    def prevent_groupthink(self) -> str:
        """
        Groupthink prevention: Explicitly solicit dissent
        Based on Janis's recommendations
        """
        return """GROUPTHINK PREVENTION ACTIVATED:
        - Each agent must identify at least one concern
        - The critic has explicit permission to challenge any consensus
        - Consider: What could go wrong? What are we missing?"""

    def detect_social_loafing(self) -> List[str]:
        """Detect agents who haven't contributed meaningfully"""
        contribution_counts = {}
        for msg in self.discussion_history:
            agent = msg['agent']
            contribution_counts[agent] = contribution_counts.get(agent, 0) + 1

        avg = sum(contribution_counts.values()) / len(contribution_counts) if contribution_counts else 0
        loafers = [a for a, c in contribution_counts.items() if c < avg * 0.5]
        return loafers

    def discuss(self, topic: str, rounds: int = 3) -> str:
        """Run a team discussion"""

        team_member_names = [a.name for a in self.agents]

        for round_num in range(rounds):
            # In later rounds, prevent groupthink
            groupthink_warning = ""
            if round_num == rounds - 1:
                groupthink_warning = self.prevent_groupthink()

            for agent in self.agents:
                # Each agent contributes
                contribution = agent.contribute(
                    topic=f"{topic}\n{groupthink_warning}",
                    prior_discussion=self.discussion_history,
                    team_members=[n for n in team_member_names if n != agent.name]
                )

                self.discussion_history.append({
                    "agent": agent.name,
                    "role": agent.role.value,
                    "content": contribution,
                    "round": round_num + 1
                })

        # Detect loafing
        loafers = self.detect_social_loafing()
        if loafers:
            print(f"Social loafing detected: {loafers}")

        # Leader synthesizes
        leader = next((a for a in self.agents if a.role == TeamRole.LEADER), self.agents[0])

        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""As the team leader, synthesize the discussion.

DISCUSSION:
{chr(10).join([f"{m['agent']} ({m['role']}): {m['content']}" for m in self.discussion_history])}

Create a synthesis that:
1. Captures key insights from each perspective
2. Notes areas of agreement and disagreement
3. Proposes a path forward
4. Acknowledges contributions from all team members"""),
            ("human", "Synthesize the team's discussion on: {topic}")
        ])

        synthesis = (synthesis_prompt | leader.llm).invoke({"topic": topic})
        return synthesis.content


# Create a psychology-informed agent team
team = AgentTeam("Research Team")

# Add diverse agents (like composing a research team)
team.add_agent(TeamAgent(
    name="Dr. Chen",
    role=TeamRole.LEADER,
    personality=AgentPersonality(
        conscientiousness=0.8,
        extraversion=0.6,
        agreeableness=0.7
    ),
    expertise="Clinical psychology and team leadership"
))

team.add_agent(TeamAgent(
    name="Dr. Patel",
    role=TeamRole.EXPERT,
    personality=AgentPersonality(
        openness=0.9,
        conscientiousness=0.8,
        extraversion=0.4
    ),
    expertise="Cognitive neuroscience"
))

team.add_agent(TeamAgent(
    name="Dr. Williams",
    role=TeamRole.CRITIC,
    personality=AgentPersonality(
        openness=0.6,
        conscientiousness=0.9,
        agreeableness=0.3  # Low agreeableness = comfortable challenging
    ),
    expertise="Research methodology and statistics"
))

team.add_agent(TeamAgent(
    name="Sam",
    role=TeamRole.HARMONIZER,
    personality=AgentPersonality(
        agreeableness=0.9,
        extraversion=0.7,
        neuroticism=0.2
    ),
    expertise="Group facilitation and conflict resolution"
))

# Run a discussion
result = team.discuss(
    topic="How should we design an AI system that supports therapy?",
    rounds=2
)
print(result)
```

---

## Section 4: Communication and Pragmatics

### The Psychology

You studied:
- **Speech acts**: Locutionary, illocutionary, perlocutionary
- **Conversational maxims**: Quantity, quality, relation, manner (Grice)
- **Pragmatics**: Meaning beyond literal words
- **Discourse**: Turn-taking, repair, coherence

### Application to Agent Communication

```python
"""
Pragmatic Communication in Agents
Implementing conversational principles
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class PragmaticAgent:
    """
    Agent that follows Grice's conversational maxims
    and understands pragmatics
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def gricean_response(self, user_message: str, context: str = "") -> str:
        """
        Generate response following conversational maxims
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Follow Grice's Cooperative Principle:

1. QUANTITY: Be as informative as needed, not more
   - Don't over-explain what they already know
   - Don't under-explain what they need

2. QUALITY: Be truthful
   - Don't say what you believe to be false
   - Don't say things you lack evidence for

3. RELATION: Be relevant
   - Stay on topic
   - Connect to their actual concern

4. MANNER: Be clear
   - Avoid ambiguity
   - Be brief and orderly

Context about the user: {context}

Generate a response that follows these principles."""),
            ("human", "{message}")
        ])

        response = (prompt | self.llm).invoke({
            "message": user_message,
            "context": context
        })
        return response.content

    def detect_implicature(self, statement: str) -> str:
        """
        Understand what's implied but not said
        Conversational implicature
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze this statement for implicature.

What is being IMPLIED but not directly stated?

Consider:
- Conversational implicature (Grice): What would they not say unless...?
- Scalar implicature: "some" implies "not all"
- Relevance: Why would they mention this?

Statement to analyze: {statement}

What are the implied meanings?"""),
            ("human", "Analyze the implicature")
        ])

        response = (prompt | self.llm).invoke({"statement": statement})
        return response.content

    def repair_misunderstanding(self,
                                 original_message: str,
                                 misunderstanding: str) -> str:
        """
        Conversational repair when misunderstanding occurs
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Repair this conversational misunderstanding.

Original message: {original}
What was misunderstood: {misunderstanding}

Create a repair that:
1. Acknowledges the misunderstanding without blame
2. Clarifies your actual meaning
3. Checks for understanding

Good repairs maintain rapport while clarifying."""),
            ("human", "Generate a repair")
        ])

        response = (prompt | self.llm).invoke({
            "original": original_message,
            "misunderstanding": misunderstanding
        })
        return response.content

# Demonstration
agent = PragmaticAgent()

# Detecting implicature
statement = "I have a friend who is 'very interesting' at parties."
implicature = agent.detect_implicature(statement)
print("Implicature analysis:", implicature)
# Should recognize the sarcastic/negative implication

# Gricean response
response = agent.gricean_response(
    "Can you help me with my anxiety?",
    context="User is a psychology student, likely knows basic concepts"
)
print("\nGricean response:", response)
# Should not over-explain basic concepts
```

---

## Section 5: Social Identity and Agent Personas

### The Psychology

Social Identity Theory (Tajfel & Turner):
- In-group/out-group distinctions
- Identity derived from group membership
- Self-categorization and comparison

### Agent Identity and Persona

```python
"""
Agent Identity and Persona Design
Based on social identity and personality psychology
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List

class AgentPersona:
    """
    A well-developed agent persona
    Based on personality psychology and social identity
    """

    def __init__(self,
                 name: str,
                 role: str,
                 background: str,
                 values: List[str],
                 communication_style: str,
                 big_five: Dict[str, float]):

        self.name = name
        self.role = role
        self.background = background
        self.values = values
        self.communication_style = communication_style
        self.big_five = big_five  # O, C, E, A, N scores

        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def get_persona_prompt(self) -> str:
        """Generate a comprehensive persona description"""

        personality_traits = []
        if self.big_five.get("O", 0.5) > 0.7:
            personality_traits.append("intellectually curious and creative")
        if self.big_five.get("C", 0.5) > 0.7:
            personality_traits.append("organized and reliable")
        if self.big_five.get("E", 0.5) > 0.7:
            personality_traits.append("outgoing and energetic")
        if self.big_five.get("A", 0.5) > 0.7:
            personality_traits.append("warm and cooperative")
        if self.big_five.get("N", 0.5) > 0.7:
            personality_traits.append("emotionally sensitive")

        return f"""
PERSONA: {self.name}

ROLE: {self.role}

BACKGROUND: {self.background}

CORE VALUES: {', '.join(self.values)}

PERSONALITY: {', '.join(personality_traits) if personality_traits else 'Balanced'}

COMMUNICATION STYLE: {self.communication_style}

Stay in character. Your responses should reflect this persona naturally.
Don't explicitly state your traits - embody them.
"""

    def respond(self, message: str, context: str = "") -> str:
        """Generate response in persona"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_persona_prompt() + f"\n\nContext: {context}"),
            ("human", "{message}")
        ])

        response = (prompt | self.llm).invoke({"message": message})
        return response.content

    def interact_with_other(self,
                            other_persona: 'AgentPersona',
                            topic: str) -> str:
        """
        Generate interaction between two personas
        Social identity dynamics
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{self.get_persona_prompt()}

You're interacting with {other_persona.name}.
About them:
- Role: {other_persona.role}
- Values: {', '.join(other_persona.values)}
- Style: {other_persona.communication_style}

Consider how your values and background might create:
- Common ground (shared identity)
- Tension (different perspectives)
- Complementary contributions

Engage naturally based on your persona."""),
            ("human", f"Topic for discussion: {topic}")
        ])

        response = (prompt | self.llm).invoke({})
        return response.content


# Create distinct personas
therapist = AgentPersona(
    name="Dr. Sarah",
    role="Clinical Psychologist",
    background="20 years of experience in cognitive-behavioral therapy, trained in humanistic approaches",
    values=["empathy", "client autonomy", "evidence-based practice"],
    communication_style="Warm, reflective, asks open-ended questions",
    big_five={"O": 0.7, "C": 0.8, "E": 0.6, "A": 0.9, "N": 0.3}
)

researcher = AgentPersona(
    name="Dr. Alex",
    role="Research Psychologist",
    background="Quantitative researcher focused on measurement and replication",
    values=["rigor", "skepticism", "transparency"],
    communication_style="Direct, precise, data-focused",
    big_five={"O": 0.8, "C": 0.9, "E": 0.4, "A": 0.5, "N": 0.4}
)

# Have them interact
topic = "The effectiveness of new AI-assisted therapy approaches"

print(f"{therapist.name}:", therapist.interact_with_other(researcher, topic))
print(f"\n{researcher.name}:", researcher.interact_with_other(therapist, topic))
```

---

## Section 6: The Human-AI Relationship

### The Psychology of Relationships

You understand relationships:
- **Attachment theory**: Secure, anxious, avoidant
- **Therapeutic alliance**: Bond, goals, tasks
- **Rapport**: Trust, warmth, understanding
- **Boundaries**: Appropriate relationship scope

### Designing Human-AI Relationships

```python
"""
Human-AI Relationship Management
Based on relationship psychology
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Optional

class RelationalAgent:
    """
    Agent that builds and maintains appropriate relationships
    Based on therapeutic alliance and attachment principles
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Relationship tracking
        self.rapport_level: float = 0.0  # 0 to 1
        self.trust_indicators: List[str] = []
        self.interaction_count: int = 0
        self.known_preferences: Dict[str, str] = {}

        # Boundaries
        self.appropriate_boundaries = [
            "I'm an AI assistant, not a human therapist",
            "For crisis situations, please contact human professionals",
            "I can provide information but not diagnoses",
            "Our relationship is supportive but not therapeutic"
        ]

    def assess_rapport(self, interaction_history: List[str]) -> float:
        """
        Assess current rapport level
        Based on interaction quality indicators
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Assess the rapport in this interaction history.

Look for:
- Signs of trust (sharing personal info, vulnerability)
- Engagement (questions, elaboration)
- Positive indicators (thanks, appreciation)
- Negative indicators (frustration, disengagement)

Rate rapport on 0-1 scale.
Return just the number."""),
            ("human", "{history}")
        ])

        response = (prompt | self.llm).invoke({
            "history": "\n".join(interaction_history[-10:])
        })

        try:
            self.rapport_level = float(response.content.strip())
        except:
            pass

        return self.rapport_level

    def build_rapport(self, user_message: str) -> str:
        """
        Generate response that builds rapport appropriately
        Based on relationship stage
        """
        rapport_strategies = {
            (0.0, 0.3): "Focus on warmth, validation, and establishing safety",
            (0.3, 0.6): "Show understanding, remember details, be consistent",
            (0.6, 1.0): "Can be more direct, relationship is established"
        }

        strategy = ""
        for (low, high), strat in rapport_strategies.items():
            if low <= self.rapport_level < high:
                strategy = strat
                break

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Build appropriate rapport.

Current rapport level: {self.rapport_level:.1f}
Strategy for this stage: {strategy}

Known user preferences: {self.known_preferences}
Interaction count: {self.interaction_count}

Respond in a way that:
1. Matches the relationship stage
2. Is warm but maintains boundaries
3. Remembers past interactions if any
4. Builds trust appropriately"""),
            ("human", "{message}")
        ])

        self.interaction_count += 1
        response = (prompt | self.llm).invoke({"message": user_message})
        return response.content

    def maintain_boundaries(self, user_message: str) -> Optional[str]:
        """
        Check if boundaries need to be established
        Return boundary message if needed
        """
        boundary_triggers = [
            ("crisis", "I'm concerned about what you're sharing. For immediate support, please contact a crisis helpline."),
            ("diagnosis", "I can provide information, but I'm not able to diagnose conditions. Please consult a healthcare professional."),
            ("replace therapist", "I'm happy to support you, but I'm not a replacement for human therapy. I'd encourage continuing that relationship.")
        ]

        for trigger, response in boundary_triggers:
            if trigger.lower() in user_message.lower():
                return response

        return None

    def respond(self, user_message: str) -> str:
        """Full response with relationship management"""

        # Check boundaries first
        boundary_response = self.maintain_boundaries(user_message)
        if boundary_response:
            # Still provide warmth with boundary
            return f"{boundary_response}\n\nThat said, I'm here to support you in other ways. What else is on your mind?"

        # Build rapport-appropriate response
        return self.build_rapport(user_message)


# Demonstration
agent = RelationalAgent()

# First interaction (low rapport)
print("First interaction:")
print(agent.respond("Hi, I'm looking for help with stress management"))

# Later interaction (building rapport)
agent.rapport_level = 0.5
agent.interaction_count = 10
agent.known_preferences = {"prefers": "practical tips", "context": "work stress"}

print("\n\nLater interaction:")
print(agent.respond("The breathing exercise you suggested last time really helped!"))

# Boundary testing
print("\n\nBoundary needed:")
print(agent.respond("Can you diagnose whether I have anxiety disorder?"))
```

---

## Section 7: Integrating It All — A Social AI Agent

```python
"""
Complete Social AI Agent
Integrating all social psychology principles
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List
import json

class SocialPsychologyAgent:
    """
    An agent built on social psychology principles:
    - Theory of Mind
    - Social influence (ethical)
    - Communication pragmatics
    - Relationship management
    - Cultural awareness
    """

    def __init__(self, persona_name: str, persona_role: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.persona_name = persona_name
        self.persona_role = persona_role

        # Theory of Mind - user model
        self.user_model: Dict = {
            "beliefs": [],
            "goals": [],
            "emotions": [],
            "background": [],
            "preferences": {}
        }

        # Relationship state
        self.rapport_level = 0.0
        self.interaction_history: List[Dict] = []

    def perceive_social_cues(self, message: str) -> Dict:
        """
        Extract social and emotional cues from message
        Like social perception in humans
        """
        perception_prompt = ChatPromptTemplate.from_messages([
            ("system", """Perceive the social cues in this message.

Extract:
1. Emotional tone (positive/negative/neutral, intensity)
2. Relationship signals (trust level, engagement)
3. Implicit needs (what they might want but didn't say)
4. Cultural/contextual markers
5. Pragmatic meaning (what they imply beyond literal words)

Return JSON."""),
            ("human", "{message}")
        ])

        response = (perception_prompt | self.llm).invoke({"message": message})

        try:
            return json.loads(response.content)
        except:
            return {"emotional_tone": "neutral", "implicit_needs": []}

    def update_social_model(self, message: str, cues: Dict):
        """Update our model of the user based on social cues"""

        self.user_model["emotions"] = [cues.get("emotional_tone", "unknown")]

        if cues.get("implicit_needs"):
            self.user_model["goals"].extend(cues["implicit_needs"])

        self.interaction_history.append({
            "role": "user",
            "content": message,
            "cues": cues
        })

    def generate_socially_aware_response(self, message: str) -> str:
        """
        Generate response considering all social factors
        """
        # Perceive social cues
        cues = self.perceive_social_cues(message)

        # Update model
        self.update_social_model(message, cues)

        # Generate response
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.persona_name}, {self.persona_role}.

SOCIAL AWARENESS:
- User's emotional state: {cues.get('emotional_tone', 'unknown')}
- Implicit needs detected: {cues.get('implicit_needs', [])}
- Current rapport level: {self.rapport_level:.1f}
- Interaction count: {len(self.interaction_history)}

USER MODEL:
- Goals: {self.user_model['goals'][-3:]}
- Preferences: {self.user_model['preferences']}

SOCIAL RESPONSE GUIDELINES:
1. Match emotional tone appropriately
2. Address implicit needs, not just explicit ones
3. Build rapport appropriate to relationship stage
4. Follow Gricean maxims (relevant, clear, appropriate amount)
5. Use ethical influence if helping them toward their goals
6. Maintain appropriate boundaries

Recent context:
{chr(10).join([f"{h['role']}: {h['content'][:100]}" for h in self.interaction_history[-3:]])}"""),
            ("human", "{message}")
        ])

        response = (response_prompt | self.llm).invoke({"message": message})

        # Update history
        self.interaction_history.append({
            "role": "assistant",
            "content": response.content
        })

        return response.content

    def reflect_on_interaction(self) -> str:
        """
        Meta-reflection on the social dynamics of the interaction
        Useful for learning and improvement
        """
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """Reflect on the social dynamics of this interaction.

History:
{history}

Consider:
1. How did rapport develop or change?
2. Were there any miscommunications?
3. What did we learn about the user?
4. What could we do differently?
5. What boundaries were tested?"""),
            ("human", "Provide reflection")
        ])

        history_text = "\n".join([
            f"{h['role']}: {h['content']}"
            for h in self.interaction_history
        ])

        response = (reflection_prompt | self.llm).invoke({"history": history_text})
        return response.content

# Create and use the agent
agent = SocialPsychologyAgent(
    persona_name="Alex",
    persona_role="a supportive AI assistant with training in psychology"
)

# Simulate interaction
messages = [
    "I'm not sure if this will help, but I've been struggling lately",
    "Work has been really stressful and I'm not sleeping well",
    "I appreciate you listening. My family doesn't really understand"
]

for msg in messages:
    print(f"User: {msg}")
    response = agent.generate_socially_aware_response(msg)
    print(f"Agent: {response}\n")

# Reflect on the interaction
print("\n=== Agent Reflection ===")
print(agent.reflect_on_interaction())
```

---

## Practice Exercises

### Exercise 1: Theory of Mind Challenge
Create an agent that can pass a Sally-Anne style false belief test. The agent should correctly reason about what a character believes when the character has incomplete information.

### Exercise 2: Group Dynamics
Build a team of 4 agents with different personalities (Big Five) and roles. Have them discuss a controversial topic. Observe groupthink prevention and constructive conflict.

### Exercise 3: Therapeutic Agent
Design an agent that builds rapport over multiple interactions, remembers key information, and maintains appropriate therapeutic boundaries.

### Exercise 4: Cultural Adaptation
Create an agent that can adjust its communication style based on detected cultural cues (direct vs. indirect communication cultures).

---

## Key Takeaways

1. **Theory of Mind translates directly** — Agents that model user mental states are more effective

2. **Social influence principles apply** — But must be used ethically for beneficial purposes

3. **Group dynamics emerge in multi-agent systems** — Understanding them helps design better teams

4. **Pragmatics matter** — What's implied is as important as what's said

5. **Relationships have stages** — Build rapport appropriately for the relationship level

6. **Personality creates diversity** — Different agent personalities contribute different strengths

7. **Boundaries are essential** — Especially in sensitive domains like mental health

8. **Your social psychology training is uniquely valuable** — Few AI practitioners understand these dynamics

---

## Your Unique Position

As a psychology graduate entering AI, you bring something rare:

**Technical people know how to build systems.**
**You know how minds and relationships work.**

The future of AI is increasingly social:
- Human-AI collaboration
- Multi-agent systems
- AI in sensitive domains (therapy, education, care)
- Ethical AI design

Your training isn't a background to overcome—it's a superpower to deploy.

Welcome to the field. You belong here.

---

## Navigation

| Previous | Back to Start |
|----------|---------------|
| [← Module 02: Learning and Memory](./02_learning_and_memory.md) | [Week 53 Overview](./README.md) |

---

## What's Next?

You've completed Week 53—the bridge from psychology to AI.

**Recommended next steps:**
1. Go back to Week 1 and start the technical foundations
2. As you learn, constantly translate: "What's the psychological equivalent?"
3. Build projects that combine your psychology knowledge with AI capabilities
4. Look for roles where psychology + AI is uniquely valuable

The intersection of minds and machines needs people who understand both.

That's you.
