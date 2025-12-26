# Module 7.3: Cost Optimization & Scaling

> "Premature optimization is the root of all evil—but mature optimization is the key to profitability." — Donald Knuth (adapted)

## What You'll Learn

- Understand the true cost drivers of AI agents
- Implement caching strategies that reduce LLM costs by 50-80%
- Design auto-scaling policies that balance cost and performance
- Apply model tiering for intelligent cost management
- Implement MCP (Model Context Protocol) for standardized agent communication
- Use A2A (Agent-to-Agent) protocols for enterprise systems
- Build cost dashboards and alerts

---

## First Principles: What Drives AI Agent Costs?

### The Cost Equation

Let's break down costs to their fundamental components:

```
TOTAL COST = COMPUTE + LLM_API + STORAGE + NETWORKING + OPERATIONS

Where:
├── COMPUTE: Running your containers
│   ├── CPU time
│   ├── Memory usage
│   └── Instance hours
│
├── LLM_API: The biggest cost driver (usually 60-80% of total)
│   ├── Input tokens (prompt)
│   ├── Output tokens (completion)
│   └── Model selection (GPT-4 vs GPT-4o-mini)
│
├── STORAGE: Persistent data
│   ├── Vector databases
│   ├── Cache storage
│   └── Logs retention
│
├── NETWORKING: Data transfer
│   ├── Egress (outbound)
│   ├── API calls
│   └── Cross-region traffic
│
└── OPERATIONS: Hidden costs
    ├── Monitoring tools
    ├── CI/CD pipelines
    └── Human time
```

### The Token Tax

Most AI agent costs come from LLM API calls:

```
LLM COST BREAKDOWN
─────────────────────────────────────────────────────────────────

                    TYPICAL AI AGENT COST DISTRIBUTION

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   LLM API Costs                                    70%  │
    │   ████████████████████████████████████████████████████  │
    │                                                         │
    │   Compute (Containers)                             15%  │
    │   ███████████                                           │
    │                                                         │
    │   Storage (Vectors, Cache)                          8%  │
    │   ██████                                                │
    │                                                         │
    │   Networking                                         5%  │
    │   ████                                                  │
    │                                                         │
    │   Other                                              2%  │
    │   ██                                                    │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

MODEL PRICING (as of 2024):
┌───────────────────┬──────────────┬──────────────┬──────────────┐
│ Model             │ Input/1M     │ Output/1M    │ Relative     │
├───────────────────┼──────────────┼──────────────┼──────────────┤
│ GPT-4             │ $30.00       │ $60.00       │ 100x         │
│ GPT-4 Turbo       │ $10.00       │ $30.00       │ 33x          │
│ GPT-4o            │ $2.50        │ $10.00       │ 8x           │
│ GPT-4o-mini       │ $0.15        │ $0.60        │ 1x (baseline)│
│ Claude 3.5 Sonnet │ $3.00        │ $15.00       │ 10x          │
│ Claude 3 Haiku    │ $0.25        │ $1.25        │ 1.5x         │
└───────────────────┴──────────────┴──────────────┴──────────────┘
```

**Key insight**: Choosing the right model for each task can reduce costs by 10-100x without significantly impacting quality.

---

## Analogical Thinking: Cost Optimization as Resource Management

```
RESTAURANT OPERATIONS                 AI AGENT OPERATIONS
───────────────────────────────────────────────────────────────────

Prep Work (Morning prep)              Caching
├── Pre-cut vegetables                ├── Pre-compute embeddings
├── Make sauces in bulk               ├── Cache common queries
├── Portion proteins                  ├── Pre-generate responses
└── Saves time during rush            └── Saves tokens during peak

Menu Engineering                      Model Tiering
├── High-margin items promoted        ├── Cheap models for simple tasks
├── Expensive items limited           ├── Expensive models when needed
├── Combo meals for value             ├── Hybrid routing
└── Different prices for same dish    └── Quality/cost trade-offs

Staff Scheduling                      Auto-Scaling
├── More staff during peaks           ├── More instances during peaks
├── Skeleton crew during slow times   ├── Scale to zero when idle
├── Cross-trained employees           ├── Generic vs. specialized instances
└── On-call for emergencies           └── Burst capacity for spikes

Inventory Management                  Token Management
├── Just-in-time ordering             ├── Minimal context windows
├── Track waste                       ├── Track token waste
├── Use scraps creatively             ├── Reuse context efficiently
└── Batch purchasing discounts        └── Batch API calls
```

---

## Emergence Thinking: Cost Efficiency from Simple Rules

Complex cost optimization emerges from simple, consistent rules:

```
SIMPLE RULES                          EMERGENT COST EFFICIENCY
─────────────────────────────────────────────────────────────────────

"Cache identical requests"            →  60%+ cost reduction
"Use cheap models for simple tasks"   →  70% of requests cost 1/10
"Set token limits"                    →  No runaway costs
"Scale based on queue depth"          →  Right-sized infrastructure
"Monitor cost per request"            →  Continuous optimization

                These rules combine to produce:

                ┌────────────────────────────────────────┐
                │                                        │
                │   SUSTAINABLE AI ECONOMICS             │
                │                                        │
                │   - Predictable costs                  │
                │   - Linear scaling with users          │
                │   - No surprise bills                  │
                │   - Room for growth                    │
                │                                        │
                └────────────────────────────────────────┘
```

---

## Caching Strategies

### Strategy 1: Semantic Caching

Cache based on meaning, not exact match:

```python
# code/03_semantic_cache.py

import hashlib
import json
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime, timedelta
import redis
from langchain_openai import OpenAIEmbeddings

class SemanticCache:
    """
    Cache that uses semantic similarity to match queries.

    Similar queries return cached results, even if not identical.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        similarity_threshold: float = 0.95,
        ttl_hours: int = 24
    ):
        self.redis = redis.from_url(redis_url)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)

        # Cache statistics
        self.stats = {"hits": 0, "misses": 0, "semantic_hits": 0}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        return np.array(self.embeddings.embed_query(text))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _exact_key(self, query: str) -> str:
        """Generate exact match cache key."""
        return f"exact:{hashlib.md5(query.encode()).hexdigest()}"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Try to get a cached response.

        First checks exact match, then semantic similarity.
        """
        # Try exact match first (fast)
        exact_key = self._exact_key(query)
        cached = self.redis.get(exact_key)

        if cached:
            self.stats["hits"] += 1
            return json.loads(cached)

        # Try semantic match (slower, but more hits)
        query_embedding = self._get_embedding(query)

        # Scan for similar queries
        for key in self.redis.scan_iter("semantic:*"):
            data = json.loads(self.redis.get(key))
            cached_embedding = np.array(data["embedding"])

            similarity = self._cosine_similarity(query_embedding, cached_embedding)

            if similarity >= self.similarity_threshold:
                self.stats["semantic_hits"] += 1
                self.stats["hits"] += 1
                return {
                    "response": data["response"],
                    "original_query": data["query"],
                    "similarity": similarity,
                    "cached": True
                }

        self.stats["misses"] += 1
        return None

    def set(self, query: str, response: str, metadata: Optional[Dict] = None):
        """Cache a response."""
        # Store exact match
        exact_key = self._exact_key(query)
        cache_data = {
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.redis.setex(exact_key, self.ttl, json.dumps(cache_data))

        # Store semantic match
        embedding = self._get_embedding(query)
        semantic_key = f"semantic:{hashlib.md5(query.encode()).hexdigest()}"
        semantic_data = {
            **cache_data,
            "embedding": embedding.tolist()
        }
        self.redis.setex(semantic_key, self.ttl, json.dumps(semantic_data))

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.2%}",
            "estimated_savings": f"{hit_rate * 100:.1f}% of LLM costs"
        }


# Usage example
async def cached_agent_call(query: str, cache: SemanticCache, agent) -> str:
    """Call agent with semantic caching."""

    # Check cache
    cached = cache.get(query)
    if cached:
        return cached["response"]

    # Call agent
    response = await agent.invoke(query)

    # Cache result
    cache.set(query, response)

    return response
```

### Strategy 2: Hierarchical Caching

Use multiple cache layers:

```python
# code/03_hierarchical_cache.py

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import time

class CacheLayer(ABC):
    """Abstract cache layer."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600):
        pass

class MemoryCache(CacheLayer):
    """In-memory cache (fastest, smallest)."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, tuple] = {}  # key -> (value, expiry)
        self.max_size = max_size

    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                return value
            del self.cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        # Simple LRU: remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest]
        self.cache[key] = (value, time.time() + ttl)

class RedisCache(CacheLayer):
    """Redis cache (fast, larger)."""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        await self.redis.setex(key, ttl, json.dumps(value))

class HierarchicalCache:
    """
    Multi-layer cache with fallback.

    Layer 1: Memory (L1) - ms latency, small capacity
    Layer 2: Redis (L2) - 1-10ms latency, medium capacity
    Layer 3: LLM (L3) - 100ms-10s latency, unlimited
    """

    def __init__(self, layers: list[CacheLayer]):
        self.layers = layers
        self.stats = {f"L{i+1}_hits": 0 for i in range(len(layers))}
        self.stats["misses"] = 0

    async def get(self, key: str) -> tuple[Optional[Any], int]:
        """
        Get from cache, returning (value, layer_hit).

        Returns (None, -1) if not found in any layer.
        """
        for i, layer in enumerate(self.layers):
            value = await layer.get(key)
            if value is not None:
                self.stats[f"L{i+1}_hits"] += 1

                # Populate higher layers (cache warming)
                for j in range(i):
                    await self.layers[j].set(key, value)

                return value, i + 1

        self.stats["misses"] += 1
        return None, -1

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set in all layers."""
        for layer in self.layers:
            await layer.set(key, value, ttl)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with cost analysis."""
        total = sum(self.stats.values())

        # Estimate costs (example: L1=free, L2=$0.001, L3=$0.01 per request)
        l1_cost = 0
        l2_cost = self.stats.get("L2_hits", 0) * 0.001
        l3_cost = self.stats["misses"] * 0.01

        return {
            **self.stats,
            "total_requests": total,
            "estimated_cost": f"${l1_cost + l2_cost + l3_cost:.2f}",
            "cost_without_cache": f"${total * 0.01:.2f}",
            "savings": f"${(total * 0.01) - (l1_cost + l2_cost + l3_cost):.2f}"
        }
```

---

## Model Tiering

### Intelligent Model Selection

Route requests to the most cost-effective model:

```python
# code/03_model_tiering.py

from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum
import re

class ModelTier(Enum):
    """Model tiers by capability and cost."""
    ECONOMY = "economy"      # GPT-4o-mini, Claude Haiku
    STANDARD = "standard"    # GPT-4o, Claude Sonnet
    PREMIUM = "premium"      # GPT-4, Claude Opus

@dataclass
class ModelConfig:
    """Configuration for a model tier."""
    tier: ModelTier
    model_name: str
    cost_per_1k_tokens: float
    max_complexity: float  # 0-1 scale

MODELS = {
    ModelTier.ECONOMY: ModelConfig(
        tier=ModelTier.ECONOMY,
        model_name="gpt-4o-mini",
        cost_per_1k_tokens=0.00015,
        max_complexity=0.4
    ),
    ModelTier.STANDARD: ModelConfig(
        tier=ModelTier.STANDARD,
        model_name="gpt-4o",
        cost_per_1k_tokens=0.0025,
        max_complexity=0.8
    ),
    ModelTier.PREMIUM: ModelConfig(
        tier=ModelTier.PREMIUM,
        model_name="gpt-4",
        cost_per_1k_tokens=0.03,
        max_complexity=1.0
    ),
}

class ComplexityAnalyzer:
    """Analyze query complexity to route to appropriate model."""

    def __init__(self):
        # Patterns indicating higher complexity
        self.complex_patterns = [
            r'\b(analyze|compare|evaluate|synthesize)\b',
            r'\b(multiple|several|various)\s+(factors|aspects|considerations)\b',
            r'\b(trade-?offs?|pros?\s+and\s+cons?)\b',
            r'\b(step[- ]by[- ]step|detailed|comprehensive)\b',
            r'\b(code|program|implement|algorithm)\b',
            r'\b(legal|medical|financial)\s+(advice|analysis)\b',
        ]

        # Patterns indicating lower complexity
        self.simple_patterns = [
            r'^(what|who|when|where)\s+is\b',
            r'\b(define|explain|describe)\s+\w+$',
            r'\b(yes|no)\s+(or|question)\b',
            r'\b(simple|quick|brief)\b',
        ]

    def analyze(self, query: str) -> float:
        """
        Analyze query complexity on 0-1 scale.

        Returns:
            float: Complexity score (0 = trivial, 1 = highly complex)
        """
        query_lower = query.lower()

        # Base complexity from length
        base_complexity = min(len(query) / 500, 0.5)

        # Adjust for patterns
        complexity_boost = 0
        for pattern in self.complex_patterns:
            if re.search(pattern, query_lower):
                complexity_boost += 0.15

        simplicity_reduction = 0
        for pattern in self.simple_patterns:
            if re.search(pattern, query_lower):
                simplicity_reduction += 0.2

        final_complexity = base_complexity + complexity_boost - simplicity_reduction
        return max(0, min(1, final_complexity))

class ModelRouter:
    """Route requests to appropriate model tier."""

    def __init__(self):
        self.analyzer = ComplexityAnalyzer()
        self.stats = {tier.value: 0 for tier in ModelTier}

    def select_model(self, query: str, min_tier: Optional[ModelTier] = None) -> ModelConfig:
        """
        Select the most cost-effective model for a query.

        Args:
            query: The user's query
            min_tier: Minimum tier to use (for quality requirements)

        Returns:
            ModelConfig for the selected model
        """
        complexity = self.analyzer.analyze(query)

        # Select based on complexity
        if complexity <= MODELS[ModelTier.ECONOMY].max_complexity:
            selected_tier = ModelTier.ECONOMY
        elif complexity <= MODELS[ModelTier.STANDARD].max_complexity:
            selected_tier = ModelTier.STANDARD
        else:
            selected_tier = ModelTier.PREMIUM

        # Enforce minimum tier if specified
        if min_tier:
            tier_order = [ModelTier.ECONOMY, ModelTier.STANDARD, ModelTier.PREMIUM]
            if tier_order.index(selected_tier) < tier_order.index(min_tier):
                selected_tier = min_tier

        self.stats[selected_tier.value] += 1
        return MODELS[selected_tier]

    def get_stats(self) -> dict:
        """Get routing statistics with cost analysis."""
        total = sum(self.stats.values())
        if total == 0:
            return {"message": "No requests yet"}

        # Calculate actual vs. premium-only costs
        actual_cost = sum(
            count * MODELS[ModelTier(tier)].cost_per_1k_tokens * 2  # Assume 2k tokens avg
            for tier, count in self.stats.items()
        )
        premium_cost = total * MODELS[ModelTier.PREMIUM].cost_per_1k_tokens * 2

        return {
            "distribution": {
                tier: f"{count/total:.1%}"
                for tier, count in self.stats.items()
            },
            "actual_cost": f"${actual_cost:.4f}",
            "premium_only_cost": f"${premium_cost:.4f}",
            "savings": f"{(1 - actual_cost/premium_cost):.1%}"
        }


# Example integration
class CostOptimizedAgent:
    """Agent that uses model tiering for cost optimization."""

    def __init__(self):
        self.router = ModelRouter()
        self.llms = {
            tier: ChatOpenAI(model=config.model_name)
            for tier, config in MODELS.items()
        }

    async def invoke(self, query: str, min_quality: Optional[ModelTier] = None) -> str:
        """Invoke with automatic model selection."""
        model_config = self.router.select_model(query, min_quality)
        llm = self.llms[model_config.tier]

        response = await llm.ainvoke(query)

        return {
            "response": response.content,
            "model_used": model_config.model_name,
            "tier": model_config.tier.value,
            "estimated_cost": model_config.cost_per_1k_tokens * 2  # Estimate
        }
```

---

## Auto-Scaling Strategies

### Request-Based Scaling

```python
# code/03_autoscaling.py

from dataclasses import dataclass
from typing import Callable
import asyncio
from collections import deque
import time

@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    min_instances: int = 1
    max_instances: int = 10
    target_requests_per_instance: int = 50  # Concurrent requests
    scale_up_threshold: float = 0.8  # 80% of target
    scale_down_threshold: float = 0.3  # 30% of target
    cooldown_seconds: int = 60
    scale_up_step: int = 2
    scale_down_step: int = 1

class AutoScaler:
    """
    Auto-scaler based on request queue depth.

    Simulates how cloud auto-scalers work.
    """

    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_instances = config.min_instances
        self.pending_requests = 0
        self.last_scale_time = 0
        self.history = deque(maxlen=100)

    def record_request(self, start: bool = True):
        """Record request start/end."""
        if start:
            self.pending_requests += 1
        else:
            self.pending_requests = max(0, self.pending_requests - 1)

        self.history.append({
            "time": time.time(),
            "pending": self.pending_requests,
            "instances": self.current_instances
        })

    def should_scale(self) -> tuple[bool, int]:
        """
        Determine if scaling is needed.

        Returns:
            (should_scale, new_instance_count)
        """
        # Check cooldown
        if time.time() - self.last_scale_time < self.config.cooldown_seconds:
            return False, self.current_instances

        # Calculate load
        capacity = self.current_instances * self.config.target_requests_per_instance
        load_ratio = self.pending_requests / capacity if capacity > 0 else 1.0

        # Scale up
        if load_ratio > self.config.scale_up_threshold:
            new_count = min(
                self.current_instances + self.config.scale_up_step,
                self.config.max_instances
            )
            if new_count > self.current_instances:
                return True, new_count

        # Scale down
        elif load_ratio < self.config.scale_down_threshold:
            new_count = max(
                self.current_instances - self.config.scale_down_step,
                self.config.min_instances
            )
            if new_count < self.current_instances:
                return True, new_count

        return False, self.current_instances

    def scale_to(self, new_count: int):
        """Apply scaling decision."""
        old_count = self.current_instances
        self.current_instances = new_count
        self.last_scale_time = time.time()

        return {
            "action": "scale_up" if new_count > old_count else "scale_down",
            "old_count": old_count,
            "new_count": new_count,
            "pending_requests": self.pending_requests,
            "timestamp": self.last_scale_time
        }

    def get_metrics(self) -> dict:
        """Get current scaling metrics."""
        capacity = self.current_instances * self.config.target_requests_per_instance
        return {
            "current_instances": self.current_instances,
            "pending_requests": self.pending_requests,
            "capacity": capacity,
            "utilization": f"{(self.pending_requests / capacity * 100):.1f}%",
            "can_scale_up": self.current_instances < self.config.max_instances,
            "can_scale_down": self.current_instances > self.config.min_instances
        }
```

### Cost-Aware Scaling

```python
# code/03_cost_aware_scaling.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
import math

@dataclass
class CostConfig:
    """Cost configuration for scaling decisions."""
    instance_cost_per_hour: float = 0.10
    llm_cost_per_request: float = 0.01
    max_daily_budget: float = 100.0
    target_cost_efficiency: float = 0.8  # 80% of budget should be LLM costs

class CostAwareScaler:
    """
    Scaler that considers cost in addition to load.

    Balances performance needs with budget constraints.
    """

    def __init__(self, cost_config: CostConfig):
        self.config = cost_config
        self.hourly_costs: Dict[int, float] = {}  # hour -> cost
        self.daily_spend = 0.0
        self.requests_today = 0

    def record_cost(self, cost: float, cost_type: str = "llm"):
        """Record a cost event."""
        hour = datetime.now().hour
        self.hourly_costs[hour] = self.hourly_costs.get(hour, 0) + cost
        self.daily_spend += cost
        if cost_type == "llm":
            self.requests_today += 1

    def get_budget_remaining(self) -> float:
        """Get remaining daily budget."""
        return max(0, self.config.max_daily_budget - self.daily_spend)

    def should_allow_request(self) -> tuple[bool, str]:
        """
        Determine if a new request should be accepted.

        Returns:
            (should_allow, reason)
        """
        remaining = self.get_budget_remaining()

        if remaining <= 0:
            return False, "Daily budget exhausted"

        # Calculate hours remaining in day
        hours_remaining = 24 - datetime.now().hour

        # Budget per remaining hour
        budget_per_hour = remaining / max(1, hours_remaining)

        # Current hour spend
        current_hour = datetime.now().hour
        current_hour_spend = self.hourly_costs.get(current_hour, 0)

        if current_hour_spend >= budget_per_hour * 1.5:  # 150% of hourly budget
            return False, "Hourly rate limit exceeded"

        return True, "OK"

    def recommend_instances(
        self,
        current_load: int,
        current_instances: int
    ) -> tuple[int, str]:
        """
        Recommend instance count based on load and budget.

        Returns:
            (recommended_instances, reason)
        """
        remaining_budget = self.get_budget_remaining()
        hours_remaining = 24 - datetime.now().hour

        # Max affordable instances
        max_affordable = math.floor(
            remaining_budget / (self.config.instance_cost_per_hour * hours_remaining)
        )

        # Load-based recommendation
        requests_per_instance = 50  # Assume 50 req/instance capacity
        load_based = math.ceil(current_load / requests_per_instance)

        # Take minimum of load-based and affordable
        recommended = min(max(1, load_based), max_affordable, 10)  # Cap at 10

        if recommended < load_based:
            reason = f"Budget limited (can afford {max_affordable} instances)"
        elif recommended == current_instances:
            reason = "Current count optimal"
        else:
            reason = f"Load requires {load_based} instances"

        return recommended, reason

    def get_cost_report(self) -> dict:
        """Generate cost report."""
        return {
            "daily_spend": f"${self.daily_spend:.2f}",
            "daily_budget": f"${self.config.max_daily_budget:.2f}",
            "remaining": f"${self.get_budget_remaining():.2f}",
            "utilization": f"{(self.daily_spend / self.config.max_daily_budget * 100):.1f}%",
            "requests_today": self.requests_today,
            "avg_cost_per_request": f"${self.daily_spend / max(1, self.requests_today):.4f}",
            "hourly_breakdown": {
                f"{h}:00": f"${c:.2f}"
                for h, c in sorted(self.hourly_costs.items())
            }
        }
```

---

## MCP: Model Context Protocol

MCP standardizes how agents share context and capabilities:

```python
# code/03_mcp.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json
import asyncio

class MCPMessageType(Enum):
    """MCP message types."""
    INITIALIZE = "initialize"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    CONTEXT_UPDATE = "context_update"

@dataclass
class MCPTool:
    """A tool exposed via MCP."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable

@dataclass
class MCPResource:
    """A resource accessible via MCP."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"

@dataclass
class MCPMessage:
    """An MCP protocol message."""
    type: MCPMessageType
    id: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class MCPServer:
    """
    MCP Server implementation.

    Exposes tools and resources to MCP clients (AI agents).
    """

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.context: Dict[str, Any] = {}

    def register_tool(self, tool: MCPTool):
        """Register a tool with the server."""
        self.tools[tool.name] = tool

    def register_resource(self, resource: MCPResource):
        """Register a resource with the server."""
        self.resources[resource.uri] = resource

    async def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Handle an incoming MCP message."""

        if message.type == MCPMessageType.INITIALIZE:
            return self._handle_initialize(message)

        elif message.type == MCPMessageType.TOOL_CALL:
            return await self._handle_tool_call(message)

        elif message.type == MCPMessageType.RESOURCE_REQUEST:
            return self._handle_resource_request(message)

        elif message.type == MCPMessageType.CONTEXT_UPDATE:
            return self._handle_context_update(message)

        else:
            return MCPMessage(
                type=MCPMessageType.TOOL_RESULT,
                id=message.id,
                payload={"error": f"Unknown message type: {message.type}"}
            )

    def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Handle initialization request."""
        return MCPMessage(
            type=MCPMessageType.INITIALIZE,
            id=message.id,
            payload={
                "server_name": self.name,
                "version": self.version,
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                    for t in self.tools.values()
                ],
                "resources": [
                    {
                        "uri": r.uri,
                        "name": r.name,
                        "description": r.description,
                        "mime_type": r.mime_type
                    }
                    for r in self.resources.values()
                ]
            }
        )

    async def _handle_tool_call(self, message: MCPMessage) -> MCPMessage:
        """Handle tool call request."""
        tool_name = message.payload.get("tool")
        arguments = message.payload.get("arguments", {})

        if tool_name not in self.tools:
            return MCPMessage(
                type=MCPMessageType.TOOL_RESULT,
                id=message.id,
                payload={"error": f"Unknown tool: {tool_name}"}
            )

        tool = self.tools[tool_name]

        try:
            result = await tool.handler(**arguments)
            return MCPMessage(
                type=MCPMessageType.TOOL_RESULT,
                id=message.id,
                payload={"result": result}
            )
        except Exception as e:
            return MCPMessage(
                type=MCPMessageType.TOOL_RESULT,
                id=message.id,
                payload={"error": str(e)}
            )

    def _handle_resource_request(self, message: MCPMessage) -> MCPMessage:
        """Handle resource request."""
        uri = message.payload.get("uri")

        if uri not in self.resources:
            return MCPMessage(
                type=MCPMessageType.RESOURCE_RESPONSE,
                id=message.id,
                payload={"error": f"Unknown resource: {uri}"}
            )

        # Return resource content (simplified)
        return MCPMessage(
            type=MCPMessageType.RESOURCE_RESPONSE,
            id=message.id,
            payload={
                "uri": uri,
                "content": self.context.get(uri, {}),
                "mime_type": self.resources[uri].mime_type
            }
        )

    def _handle_context_update(self, message: MCPMessage) -> MCPMessage:
        """Handle context update."""
        key = message.payload.get("key")
        value = message.payload.get("value")

        self.context[key] = value

        return MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE,
            id=message.id,
            payload={"status": "updated", "key": key}
        )


# Example: Create an MCP server for a research agent
def create_research_mcp_server():
    """Create MCP server with research capabilities."""

    server = MCPServer("research-agent", "1.0")

    # Register search tool
    async def web_search(query: str, num_results: int = 5):
        # Simulated search
        return [{"title": f"Result {i}", "url": f"https://example.com/{i}"}
                for i in range(num_results)]

    server.register_tool(MCPTool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "default": 5}
        },
        handler=web_search
    ))

    # Register knowledge base resource
    server.register_resource(MCPResource(
        uri="knowledge://research-findings",
        name="Research Findings",
        description="Accumulated research findings from previous queries"
    ))

    return server
```

---

## A2A: Agent-to-Agent Protocol

A2A enables direct agent communication:

```python
# code/03_a2a.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio
import uuid

class A2ACapability(Enum):
    """Standard A2A capabilities."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODING = "coding"
    DATA_PROCESSING = "data_processing"

@dataclass
class AgentCard:
    """
    Agent Card - describes an agent's capabilities.

    Used for agent discovery in A2A systems.
    """
    id: str
    name: str
    description: str
    capabilities: List[A2ACapability]
    endpoint: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_capability(self, required: A2ACapability) -> bool:
        """Check if agent has required capability."""
        return required in self.capabilities

@dataclass
class A2ATask:
    """A task delegated between agents."""
    id: str
    type: str
    payload: Dict[str, Any]
    source_agent: str
    target_agent: Optional[str] = None
    priority: int = 5  # 1-10, lower is higher priority
    timeout_seconds: int = 300
    status: str = "pending"
    result: Optional[Any] = None

class A2ARegistry:
    """
    Registry for agent discovery.

    Allows agents to register themselves and discover others.
    """

    def __init__(self):
        self.agents: Dict[str, AgentCard] = {}

    def register(self, agent: AgentCard):
        """Register an agent."""
        self.agents[agent.id] = agent

    def unregister(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]

    def find_by_capability(self, capability: A2ACapability) -> List[AgentCard]:
        """Find agents with a specific capability."""
        return [
            agent for agent in self.agents.values()
            if agent.matches_capability(capability)
        ]

    def find_best_match(
        self,
        capabilities: List[A2ACapability],
        exclude: Optional[List[str]] = None
    ) -> Optional[AgentCard]:
        """Find the best matching agent for required capabilities."""
        exclude = exclude or []

        best_match = None
        best_score = 0

        for agent in self.agents.values():
            if agent.id in exclude:
                continue

            score = sum(1 for cap in capabilities if cap in agent.capabilities)
            if score > best_score:
                best_score = score
                best_match = agent

        return best_match

class A2AAgent:
    """
    Base class for A2A-enabled agents.

    Provides agent-to-agent communication capabilities.
    """

    def __init__(
        self,
        card: AgentCard,
        registry: A2ARegistry
    ):
        self.card = card
        self.registry = registry
        self.pending_tasks: Dict[str, A2ATask] = {}
        self.task_handlers: Dict[str, callable] = {}

        # Register with the registry
        registry.register(card)

    def register_handler(self, task_type: str, handler: callable):
        """Register a handler for a task type."""
        self.task_handlers[task_type] = handler

    async def delegate_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        required_capability: A2ACapability,
        timeout: int = 300
    ) -> A2ATask:
        """
        Delegate a task to another agent.

        Finds a suitable agent and sends the task.
        """
        # Find suitable agent
        target = self.registry.find_best_match(
            [required_capability],
            exclude=[self.card.id]  # Don't delegate to self
        )

        if not target:
            raise ValueError(f"No agent found with capability: {required_capability}")

        # Create task
        task = A2ATask(
            id=str(uuid.uuid4()),
            type=task_type,
            payload=payload,
            source_agent=self.card.id,
            target_agent=target.id,
            timeout_seconds=timeout
        )

        self.pending_tasks[task.id] = task

        # In a real implementation, this would send to the target agent
        # Here we simulate local execution
        result = await self._execute_remote(target, task)

        task.status = "completed"
        task.result = result

        return task

    async def _execute_remote(self, target: AgentCard, task: A2ATask) -> Any:
        """Execute a task on a remote agent (simulated)."""
        # In production, this would make an HTTP call to target.endpoint
        # For demonstration, we'll simulate it

        return {
            "executed_by": target.id,
            "task_type": task.type,
            "status": "success"
        }

    async def handle_incoming_task(self, task: A2ATask) -> A2ATask:
        """Handle an incoming task from another agent."""

        if task.type not in self.task_handlers:
            task.status = "failed"
            task.result = {"error": f"Unknown task type: {task.type}"}
            return task

        handler = self.task_handlers[task.type]

        try:
            result = await handler(task.payload)
            task.status = "completed"
            task.result = result
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}

        return task


# Example: Multi-agent research system with A2A
async def create_research_team():
    """Create a team of A2A-enabled agents."""

    registry = A2ARegistry()

    # Research agent
    research_card = AgentCard(
        id="research-001",
        name="Research Agent",
        description="Searches and gathers information",
        capabilities=[A2ACapability.RESEARCH],
        endpoint="http://localhost:8001"
    )
    research_agent = A2AAgent(research_card, registry)

    # Analysis agent
    analysis_card = AgentCard(
        id="analysis-001",
        name="Analysis Agent",
        description="Analyzes and synthesizes information",
        capabilities=[A2ACapability.ANALYSIS],
        endpoint="http://localhost:8002"
    )
    analysis_agent = A2AAgent(analysis_card, registry)

    # Writing agent
    writing_card = AgentCard(
        id="writing-001",
        name="Writing Agent",
        description="Creates written content",
        capabilities=[A2ACapability.WRITING],
        endpoint="http://localhost:8003"
    )
    writing_agent = A2AAgent(writing_card, registry)

    # Coordinator can delegate to any agent
    coordinator_card = AgentCard(
        id="coordinator-001",
        name="Coordinator",
        description="Coordinates multi-agent tasks",
        capabilities=[],
        endpoint="http://localhost:8000"
    )
    coordinator = A2AAgent(coordinator_card, registry)

    return {
        "coordinator": coordinator,
        "research": research_agent,
        "analysis": analysis_agent,
        "writing": writing_agent,
        "registry": registry
    }
```

---

## Cost Dashboard

```python
# code/03_cost_dashboard.py

from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json

@dataclass
class CostEvent:
    """A single cost event."""
    timestamp: datetime
    category: str  # "llm", "compute", "storage", "network"
    amount: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class CostDashboard:
    """
    Real-time cost tracking and alerting.
    """

    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.events: List[CostEvent] = []
        self.alerts: List[Dict] = []
        self.alert_thresholds = {
            "warning": 0.7,   # 70% of budget
            "critical": 0.9,  # 90% of budget
            "exceeded": 1.0   # 100% of budget
        }

    def record(self, category: str, amount: float, **metadata):
        """Record a cost event."""
        event = CostEvent(
            timestamp=datetime.utcnow(),
            category=category,
            amount=amount,
            metadata=metadata
        )
        self.events.append(event)

        # Check for alerts
        self._check_alerts()

    def _check_alerts(self):
        """Check if any alert thresholds are crossed."""
        daily_total = self.get_daily_total()
        utilization = daily_total / self.daily_budget

        for level, threshold in self.alert_thresholds.items():
            if utilization >= threshold:
                # Check if we already alerted for this level today
                today = datetime.utcnow().date()
                already_alerted = any(
                    a["level"] == level and a["date"] == str(today)
                    for a in self.alerts
                )

                if not already_alerted:
                    self.alerts.append({
                        "level": level,
                        "date": str(today),
                        "utilization": f"{utilization:.1%}",
                        "total": f"${daily_total:.2f}",
                        "budget": f"${self.daily_budget:.2f}"
                    })

    def get_daily_total(self) -> float:
        """Get total cost for today."""
        today = datetime.utcnow().date()
        return sum(
            e.amount for e in self.events
            if e.timestamp.date() == today
        )

    def get_breakdown(self, period: str = "day") -> Dict[str, float]:
        """Get cost breakdown by category."""
        if period == "day":
            cutoff = datetime.utcnow() - timedelta(days=1)
        elif period == "week":
            cutoff = datetime.utcnow() - timedelta(weeks=1)
        else:
            cutoff = datetime.utcnow() - timedelta(days=30)

        breakdown = defaultdict(float)
        for event in self.events:
            if event.timestamp >= cutoff:
                breakdown[event.category] += event.amount

        return dict(breakdown)

    def get_hourly_trend(self) -> List[Dict]:
        """Get hourly cost trend for today."""
        today = datetime.utcnow().date()
        hourly = defaultdict(float)

        for event in self.events:
            if event.timestamp.date() == today:
                hour = event.timestamp.hour
                hourly[hour] += event.amount

        return [
            {"hour": h, "cost": hourly.get(h, 0)}
            for h in range(24)
        ]

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost report."""
        daily_total = self.get_daily_total()
        breakdown = self.get_breakdown("day")

        return {
            "summary": {
                "daily_total": f"${daily_total:.2f}",
                "daily_budget": f"${self.daily_budget:.2f}",
                "utilization": f"{daily_total/self.daily_budget:.1%}",
                "remaining": f"${max(0, self.daily_budget - daily_total):.2f}"
            },
            "breakdown": {
                cat: f"${amt:.2f} ({amt/daily_total*100:.1f}%)"
                for cat, amt in breakdown.items()
            } if daily_total > 0 else {},
            "hourly_trend": self.get_hourly_trend(),
            "active_alerts": [a for a in self.alerts if a["date"] == str(datetime.utcnow().date())],
            "recommendations": self._generate_recommendations(breakdown, daily_total)
        }

    def _generate_recommendations(
        self,
        breakdown: Dict[str, float],
        total: float
    ) -> List[str]:
        """Generate cost optimization recommendations."""
        recs = []

        if total > 0:
            llm_pct = breakdown.get("llm", 0) / total

            if llm_pct > 0.8:
                recs.append("LLM costs are 80%+ of total. Consider implementing semantic caching.")

            if breakdown.get("llm", 0) > 50:
                recs.append("High LLM spend. Consider model tiering for simple queries.")

        if total > self.daily_budget * 0.5 and datetime.utcnow().hour < 12:
            recs.append("Spending rate is high for time of day. Review rate limits.")

        if not recs:
            recs.append("Costs are within expected ranges.")

        return recs
```

---

## Key Takeaways

### 1. LLM Costs Dominate
60-80% of AI agent costs are typically LLM API calls. Focus optimization efforts there first.

### 2. Caching Is Your Best Friend
Semantic caching can reduce LLM costs by 50-80% with minimal impact on user experience.

### 3. Model Tiering Is Essential
Use cheap models (GPT-4o-mini, Haiku) for simple tasks, expensive models only when needed.

### 4. Scale Based on Value, Not Just Load
Consider cost per request when making scaling decisions, not just throughput.

### 5. MCP and A2A Enable Enterprise Patterns
Standard protocols allow agents to share capabilities and delegate work efficiently.

### 6. Monitor Costs in Real-Time
Set up dashboards and alerts before you get a surprise bill.

---

## Implementation Checklist

```
COST OPTIMIZATION CHECKLIST
─────────────────────────────────────────────────────────────────

□ CACHING
  □ Implement exact-match caching for common queries
  □ Add semantic caching for similar queries
  □ Set appropriate TTLs (balance freshness vs. cost)
  □ Monitor cache hit rates (target: 60%+)

□ MODEL TIERING
  □ Implement complexity analyzer
  □ Route simple queries to cheap models
  □ Allow override for quality-critical requests
  □ Track model distribution in metrics

□ AUTO-SCALING
  □ Configure min/max instances
  □ Set appropriate scaling thresholds
  □ Implement cost-aware scaling
  □ Test scale-to-zero behavior

□ MONITORING
  □ Track cost per request
  □ Set up budget alerts
  □ Create cost dashboard
  □ Review weekly cost reports

□ PROTOCOLS
  □ Implement MCP for tool sharing
  □ Set up A2A for agent delegation
  □ Register agents with discovery service
  □ Test inter-agent communication
```

---

## What's Next?

You've now learned how to:
- Build AI agents (Weeks 1-4)
- Make them collaborate safely (Weeks 5-6)
- Deploy them to production at scale (Week 7)

In **Week 8: Capstone Project**, you'll:
- Build an end-to-end AI agent system
- Apply all techniques learned
- Create a portfolio-ready project
- Present your work

You're ready to build production AI systems. Let's create something amazing!

[Start Week 8: Capstone Project →](../week8/README.md)
