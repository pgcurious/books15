# Module 4.3: Building Production Data Pipelines

## What You'll Learn
- Design multi-source data integration architectures
- Implement caching strategies for cost and performance optimization
- Build quality gates that prevent bad data from reaching users
- Add monitoring, logging, and observability to your pipelines
- Create a complete production-ready data agent

---

## First Principles: What Makes Production Different?

### The Gap Between Demo and Production

Let's be honest about what changes when you go to production:

```
Demo Environment:              Production Environment:
├── Single user                ├── Thousands of users
├── Happy path only            ├── Every edge case
├── Free API tier              ├── Paid usage at scale
├── "Works on my machine"      ├── Must work everywhere
├── Restart on failure         ├── Self-healing required
└── No cost concerns           └── Every token costs money
```

**First Principle #1:** Production systems must handle the unexpected.

Every API will fail. Every database will be slow sometimes. Users will ask impossible questions. Your system must not break.

**First Principle #2:** Cost scales with usage.

```
Demo:
├── 10 queries/day × $0.001 = $0.01/day
└── "Basically free!"

Production:
├── 10,000 queries/day × $0.001 = $10/day
├── × 30 days = $300/month
├── × embeddings = $600/month
├── × retries = $900/month
└── "We need to optimize!"
```

**First Principle #3:** Observability is not optional.

When something goes wrong at 3 AM:
- **Without observability:** "Something is broken. I have no idea what."
- **With observability:** "Latency spike at 2:47 AM. Weather API returned 503. Fallback activated. 12 queries affected."

---

## The Production Data Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DATA PIPELINE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                            │
│  │   User      │                                                            │
│  │   Query     │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         GATEWAY LAYER                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │   │
│  │  │ Rate     │  │ Auth     │  │ Input    │  │ Query               │ │   │
│  │  │ Limiting │─►│ Check    │─►│ Validate │─►│ Classification      │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘ │   │
│  └────────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                      │
│         ┌─────────────────────────────┼─────────────────────────────┐       │
│         │                             ▼                              │       │
│  ┌──────┴───────────────────────────────────────────────────────────┴────┐  │
│  │                         CACHE LAYER                                    │  │
│  │  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐  │  │
│  │  │ Response Cache │      │ Embedding Cache│      │ API Cache      │  │  │
│  │  │ (Full answers) │      │ (Vector store) │      │ (API results)  │  │  │
│  │  └────────────────┘      └────────────────┘      └────────────────┘  │  │
│  └────────────────────────────────┬──────────────────────────────────────┘  │
│                                   │ Cache Miss                               │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         DATA LAYER                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ External     │  │ Vector       │  │ Database     │               │   │
│  │  │ APIs         │  │ Store        │  │ (SQL/NoSQL)  │               │   │
│  │  │ ┌──────────┐ │  │              │  │              │               │   │
│  │  │ │ Weather  │ │  │   [Docs]     │  │  [User Data] │               │   │
│  │  │ │ News     │ │  │   [Index]    │  │  [History]   │               │   │
│  │  │ │ Finance  │ │  │              │  │              │               │   │
│  │  │ └──────────┘ │  │              │  │              │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └────────────────────────────────┬──────────────────────────────────────┘  │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       QUALITY GATES                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ Data         │  │ Relevance    │  │ Hallucination│               │   │
│  │  │ Validation   │  │ Check        │  │ Detection    │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └────────────────────────────────┬──────────────────────────────────────┘  │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       SYNTHESIS LAYER                                 │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │                           LLM                                   │ │   │
│  │  │            Generate grounded, cited response                    │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────┬──────────────────────────────────────┘  │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       OUTPUT VALIDATION                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ PII Check    │  │ Content      │  │ Format       │               │   │
│  │  │              │  │ Safety       │  │ Validation   │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └────────────────────────────────┬──────────────────────────────────────┘  │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌──────────────┐                                   │
│                          │   Response    │                                   │
│                          └──────────────┘                                   │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│  │                    OBSERVABILITY LAYER                               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │ Metrics │  │ Logs    │  │ Traces  │  │ Alerts  │  │ Costs   │   │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Multi-Source Data Integration

### The Multi-Source Pattern

Real questions often require multiple data sources:

```python
# See code/09_multi_source.py for executable version

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class DataSourceType(Enum):
    API = "api"
    VECTOR_STORE = "vector_store"
    DATABASE = "database"
    CACHE = "cache"

@dataclass
class DataResult:
    source: str
    source_type: DataSourceType
    success: bool
    data: Any
    latency_ms: float
    error: str = None

class MultiSourceOrchestrator:
    """Orchestrates queries across multiple data sources."""

    def __init__(self, sources: Dict[str, Any], timeout: float = 10.0):
        self.sources = sources
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=10)

    def query_source(self, name: str, source: Any, query: str) -> DataResult:
        """Query a single source with timing."""
        start = time.time()
        try:
            result = source.query(query)
            latency = (time.time() - start) * 1000
            return DataResult(
                source=name,
                source_type=source.source_type,
                success=True,
                data=result,
                latency_ms=latency
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return DataResult(
                source=name,
                source_type=source.source_type,
                success=False,
                data=None,
                latency_ms=latency,
                error=str(e)
            )

    def query_all(self, query: str) -> List[DataResult]:
        """Query all sources in parallel."""
        futures = {
            self.executor.submit(self.query_source, name, source, query): name
            for name, source in self.sources.items()
        }

        results = []
        for future in as_completed(futures, timeout=self.timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                name = futures[future]
                results.append(DataResult(
                    source=name,
                    source_type=DataSourceType.API,
                    success=False,
                    data=None,
                    latency_ms=self.timeout * 1000,
                    error=f"Timeout or error: {str(e)}"
                ))

        return results

    def query_with_fallback(
        self,
        query: str,
        primary_sources: List[str],
        fallback_sources: List[str]
    ) -> List[DataResult]:
        """Try primary sources first, use fallbacks if they fail."""

        # Try primary sources
        primary_results = []
        for name in primary_sources:
            if name in self.sources:
                result = self.query_source(name, self.sources[name], query)
                primary_results.append(result)

        # Check if we got enough successful results
        successful = [r for r in primary_results if r.success]
        if len(successful) >= len(primary_sources) // 2:
            return primary_results

        # Try fallbacks
        fallback_results = []
        for name in fallback_sources:
            if name in self.sources:
                result = self.query_source(name, self.sources[name], query)
                fallback_results.append(result)

        return primary_results + fallback_results
```

### Source Priority and Routing

```python
class SmartRouter:
    """Routes queries to the most appropriate sources."""

    def __init__(self, sources: Dict[str, Any]):
        self.sources = sources
        self.source_capabilities = {
            "weather_api": ["weather", "temperature", "forecast", "climate"],
            "news_api": ["news", "current events", "headlines", "breaking"],
            "finance_api": ["stock", "price", "market", "crypto", "trading"],
            "documentation": ["how to", "guide", "tutorial", "documentation"],
            "faq_store": ["question", "help", "support", "issue"]
        }

    def route_query(self, query: str) -> List[str]:
        """Determine which sources to query based on the question."""
        query_lower = query.lower()

        relevant_sources = []
        for source, keywords in self.source_capabilities.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_sources.append(source)

        # Always include general knowledge if no specific match
        if not relevant_sources:
            relevant_sources = ["documentation", "faq_store"]

        return relevant_sources
```

---

## Part 2: Caching Strategies

### Why Caching Matters

```
Without Caching:
├── Every query embeds text: $0.0001/query × 1000 = $0.10
├── Every query calls LLM: $0.01/query × 1000 = $10
├── Every query hits APIs: Variable, often rate limited
└── Total: High cost, high latency, rate limit risk

With Caching:
├── 70% queries hit cache: $0 × 700 = $0
├── 30% queries miss cache: $0.01 × 300 = $3
├── API results cached: Rate limits rarely hit
└── Total: 70% cost reduction, faster response
```

### Implementing a Multi-Layer Cache

```python
# See code/10_caching.py for executable version

import hashlib
import json
import time
from typing import Optional, Any, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class CacheEntry:
    value: Any
    created_at: float
    ttl_seconds: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl_seconds

class CacheLayer(ABC):
    """Abstract base for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float) -> None:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass

class InMemoryCache(CacheLayer):
    """Fast in-memory cache for single-instance deployments."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        entry = self.cache.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self.cache[key]
            return None
        entry.hit_count += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: float = 3600) -> None:
        # Evict if at capacity (simple LRU would be better)
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]

        self.cache[key] = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl
        )

    def delete(self, key: str) -> None:
        self.cache.pop(key, None)

class SemanticCache:
    """Cache that matches semantically similar queries."""

    def __init__(self, embeddings, similarity_threshold: float = 0.95):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        self.cache: Dict[str, tuple] = {}  # query_hash: (embedding, response)
        self.vectors = []
        self.keys = []

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        """Find semantically similar cached query."""
        if not self.vectors:
            return None

        query_embedding = self.embeddings.embed_query(query)

        # Find most similar cached query
        import numpy as np
        similarities = [
            np.dot(query_embedding, v) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(v))
            for v in self.vectors
        ]

        max_sim = max(similarities)
        if max_sim >= self.threshold:
            idx = similarities.index(max_sim)
            _, response = self.cache[self.keys[idx]]
            return response

        return None

    def set(self, query: str, response: Any) -> None:
        """Cache a query-response pair."""
        key = self._hash_query(query)
        embedding = self.embeddings.embed_query(query)
        self.cache[key] = (embedding, response)
        self.vectors.append(embedding)
        self.keys.append(key)

class ProductionCache:
    """Multi-layer cache for production use."""

    def __init__(self, embeddings=None):
        self.l1_cache = InMemoryCache(max_size=1000)  # Exact match
        self.l2_cache = SemanticCache(embeddings) if embeddings else None
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0
        }

    def _cache_key(self, query: str, source: str = "") -> str:
        """Generate cache key from query and source."""
        return hashlib.md5(f"{source}:{query}".encode()).hexdigest()

    def get(self, query: str, source: str = "") -> Optional[Any]:
        """Try all cache layers."""
        key = self._cache_key(query, source)

        # L1: Exact match
        result = self.l1_cache.get(key)
        if result is not None:
            self.stats["l1_hits"] += 1
            return result

        # L2: Semantic match
        if self.l2_cache:
            result = self.l2_cache.get(query)
            if result is not None:
                self.stats["l2_hits"] += 1
                # Promote to L1
                self.l1_cache.set(key, result)
                return result

        self.stats["misses"] += 1
        return None

    def set(self, query: str, response: Any, source: str = "", ttl: float = 3600) -> None:
        """Set in all cache layers."""
        key = self._cache_key(query, source)
        self.l1_cache.set(key, response, ttl)
        if self.l2_cache:
            self.l2_cache.set(query, response)

    def get_stats(self) -> Dict[str, Any]:
        total = sum(self.stats.values())
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "hit_rate": (self.stats["l1_hits"] + self.stats["l2_hits"]) / total
        }
```

### Cache Invalidation Patterns

```python
class CacheInvalidator:
    """Strategies for cache invalidation."""

    def __init__(self, cache: ProductionCache):
        self.cache = cache

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        count = 0
        # Implementation depends on cache backend
        return count

    def invalidate_by_age(self, max_age_seconds: float) -> int:
        """Invalidate entries older than max_age."""
        count = 0
        current_time = time.time()
        for key, entry in list(self.cache.l1_cache.cache.items()):
            if current_time - entry.created_at > max_age_seconds:
                self.cache.l1_cache.delete(key)
                count += 1
        return count

    def invalidate_by_source(self, source: str) -> int:
        """Invalidate all entries from a specific source."""
        # Useful when source data is updated
        count = 0
        # Implementation depends on how we track sources
        return count
```

---

## Part 3: Quality Gates

### What Are Quality Gates?

Quality gates are checkpoints that validate data before it reaches the user:

```
Data Flow:                      Quality Gate:
┌──────────────┐               ┌──────────────────────┐
│ Raw API Data │──────────────►│ Is this valid JSON?  │──┐
└──────────────┘               └──────────────────────┘  │
                                                          │
                               ┌──────────────────────┐  │
                         YES ◄─┤ Does it have required│◄─┘
                               │ fields?              │
                               └──────────────────────┘
                                          │
                               ┌──────────┴──────────┐
                               │   NO         YES    │
                               ▼                     ▼
                         ┌──────────┐         ┌──────────┐
                         │ Reject/  │         │ Continue │
                         │ Fallback │         │ Pipeline │
                         └──────────┘         └──────────┘
```

### Implementing Quality Gates

```python
# See code/11_quality_gates.py for executable version

from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class QualityStatus(Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"

@dataclass
class QualityResult:
    gate_name: str
    status: QualityStatus
    message: str
    details: Optional[Dict] = None

class QualityGate:
    """A single quality check."""

    def __init__(self, name: str, check_fn: Callable, severity: str = "error"):
        self.name = name
        self.check_fn = check_fn
        self.severity = severity  # "error", "warning", "info"

    def check(self, data: Any) -> QualityResult:
        try:
            passed, message = self.check_fn(data)
            if passed:
                return QualityResult(self.name, QualityStatus.PASSED, message)
            elif self.severity == "warning":
                return QualityResult(self.name, QualityStatus.WARNING, message)
            else:
                return QualityResult(self.name, QualityStatus.FAILED, message)
        except Exception as e:
            return QualityResult(
                self.name,
                QualityStatus.FAILED,
                f"Gate error: {str(e)}"
            )

class QualityPipeline:
    """Chain of quality gates."""

    def __init__(self, gates: List[QualityGate], fail_fast: bool = True):
        self.gates = gates
        self.fail_fast = fail_fast

    def run(self, data: Any) -> tuple[bool, List[QualityResult]]:
        """Run all quality gates on data."""
        results = []
        all_passed = True

        for gate in self.gates:
            result = gate.check(data)
            results.append(result)

            if result.status == QualityStatus.FAILED:
                all_passed = False
                if self.fail_fast:
                    break

        return all_passed, results

# Common quality gates for RAG
def create_rag_quality_gates() -> List[QualityGate]:
    """Standard quality gates for RAG pipelines."""

    def check_retrieval_relevance(data: Dict) -> tuple[bool, str]:
        """Ensure retrieved documents are relevant."""
        docs = data.get("retrieved_docs", [])
        scores = data.get("relevance_scores", [])

        if not docs:
            return False, "No documents retrieved"

        # Check minimum relevance threshold
        min_score = min(scores) if scores else 0
        if min_score < 0.5:
            return False, f"Low relevance scores (min: {min_score:.2f})"

        return True, f"Retrieval quality OK (min score: {min_score:.2f})"

    def check_context_length(data: Dict) -> tuple[bool, str]:
        """Ensure context isn't too long or too short."""
        context = data.get("context", "")
        length = len(context.split())

        if length < 50:
            return False, f"Context too short ({length} words)"
        if length > 4000:
            return False, f"Context too long ({length} words)"

        return True, f"Context length OK ({length} words)"

    def check_answer_grounding(data: Dict) -> tuple[bool, str]:
        """Check if answer is grounded in context."""
        answer = data.get("answer", "").lower()
        context = data.get("context", "").lower()

        # Simple check: key terms in answer should appear in context
        answer_words = set(answer.split())
        context_words = set(context.split())

        common = answer_words.intersection(context_words)
        grounding_ratio = len(common) / len(answer_words) if answer_words else 0

        if grounding_ratio < 0.3:
            return False, f"Answer may not be grounded (ratio: {grounding_ratio:.2f})"

        return True, f"Answer grounding OK (ratio: {grounding_ratio:.2f})"

    def check_no_hallucination_markers(data: Dict) -> tuple[bool, str]:
        """Check for common hallucination patterns."""
        answer = data.get("answer", "")

        hallucination_markers = [
            "I don't have access to",
            "I cannot find",
            "As an AI",
            "I'm not sure but",
            "I believe",  # Uncertainty without citation
        ]

        for marker in hallucination_markers:
            if marker.lower() in answer.lower():
                return False, f"Potential hallucination: '{marker}'"

        return True, "No hallucination markers detected"

    return [
        QualityGate("retrieval_relevance", check_retrieval_relevance),
        QualityGate("context_length", check_context_length),
        QualityGate("answer_grounding", check_answer_grounding),
        QualityGate("no_hallucination", check_no_hallucination_markers, "warning"),
    ]
```

### Hallucination Detection

```python
class HallucinationDetector:
    """Detect potential hallucinations in LLM output."""

    def __init__(self, llm):
        self.llm = llm

    def check_factual_consistency(
        self,
        answer: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """Use LLM to verify answer against sources."""

        prompt = f"""
        Analyze if the following answer is factually consistent with the sources.

        Answer to verify:
        {answer}

        Sources:
        {chr(10).join(sources)}

        Respond with JSON:
        {{
            "is_consistent": true/false,
            "confidence": 0.0-1.0,
            "issues": ["list of inconsistencies if any"],
            "unsupported_claims": ["claims not found in sources"]
        }}
        """

        response = self.llm.invoke(prompt)
        # Parse JSON response
        # ...
        return {"is_consistent": True, "confidence": 0.9}

    def check_citation_accuracy(
        self,
        answer: str,
        sources: Dict[str, str]
    ) -> Dict[str, Any]:
        """Verify that citations match source content."""

        # Extract citations from answer (e.g., [1], [2])
        import re
        citations = re.findall(r'\[(\d+)\]', answer)

        results = {
            "citations_found": len(citations),
            "citations_valid": 0,
            "invalid_citations": []
        }

        # Verify each citation
        for cite_num in set(citations):
            if cite_num in sources:
                results["citations_valid"] += 1
            else:
                results["invalid_citations"].append(cite_num)

        return results
```

---

## Part 4: Monitoring and Observability

### The Three Pillars of Observability

```
Observability:
├── Metrics: Quantitative measurements over time
│   ├── Request count
│   ├── Latency percentiles
│   ├── Error rate
│   └── Token usage
│
├── Logs: Discrete events with context
│   ├── Request/response pairs
│   ├── Error details
│   ├── Quality gate results
│   └── Cache hit/miss
│
└── Traces: End-to-end request flow
    ├── Which sources were queried
    ├── How long each step took
    ├── Where failures occurred
    └── Full request context
```

### Implementing Observability

```python
# See code/12_observability.py for executable version

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import json

@dataclass
class Span:
    """A single operation in a trace."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

class Tracer:
    """Traces request flow through the pipeline."""

    def __init__(self, trace_id: str = None):
        self.trace_id = trace_id or self._generate_id()
        self.spans: list[Span] = []
        self.current_span: Optional[Span] = None

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())[:8]

    @contextmanager
    def span(self, name: str, **metadata):
        """Context manager for creating spans."""
        span = Span(
            name=name,
            start_time=time.time(),
            metadata=metadata
        )
        self.spans.append(span)
        previous_span = self.current_span
        self.current_span = span

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.error = str(e)
            raise
        finally:
            span.end_time = time.time()
            self.current_span = previous_span

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    "metadata": s.metadata,
                    "error": s.error
                }
                for s in self.spans
            ],
            "total_duration_ms": sum(s.duration_ms for s in self.spans)
        }

class Metrics:
    """Collects and exposes metrics."""

    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.histograms: Dict[str, list] = {}
        self.gauges: Dict[str, float] = {}

    def increment(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        key = self._make_key(name, tags)
        self.counters[key] = self.counters.get(key, 0) + value

    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        key = self._make_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        key = self._make_key(name, tags)
        self.gauges[key] = value

    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"

    def get_percentile(self, name: str, percentile: float) -> Optional[float]:
        values = self.histograms.get(name, [])
        if not values:
            return None
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def summary(self) -> Dict[str, Any]:
        return {
            "counters": self.counters,
            "histograms": {
                k: {
                    "count": len(v),
                    "p50": self.get_percentile(k, 50),
                    "p95": self.get_percentile(k, 95),
                    "p99": self.get_percentile(k, 99),
                }
                for k, v in self.histograms.items()
            },
            "gauges": self.gauges
        }

class PipelineLogger:
    """Structured logging for pipeline operations."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter for structured logs
        handler = logging.StreamHandler()
        handler.setFormatter(self._json_formatter())
        self.logger.addHandler(handler)

    def _json_formatter(self):
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name
                }
                if hasattr(record, 'extra'):
                    log_data.update(record.extra)
                return json.dumps(log_data)
        return JsonFormatter()

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra={"extra": kwargs})

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra={"extra": kwargs})

    def request(self, query: str, trace_id: str, **kwargs):
        self.info("Request received", query=query, trace_id=trace_id, **kwargs)

    def response(self, trace_id: str, duration_ms: float, **kwargs):
        self.info("Response sent", trace_id=trace_id, duration_ms=duration_ms, **kwargs)
```

### Cost Tracking

```python
class CostTracker:
    """Track API and token costs."""

    # Pricing per 1K tokens (adjust based on actual pricing)
    PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "text-embedding-3-small": {"input": 0.00002},
        "text-embedding-3-large": {"input": 0.00013},
    }

    def __init__(self):
        self.usage = {}
        self.total_cost = 0.0

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0
    ):
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})

        cost = (
            (input_tokens / 1000) * pricing.get("input", 0) +
            (output_tokens / 1000) * pricing.get("output", 0)
        )

        if model not in self.usage:
            self.usage[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "calls": 0
            }

        self.usage[model]["input_tokens"] += input_tokens
        self.usage[model]["output_tokens"] += output_tokens
        self.usage[model]["cost"] += cost
        self.usage[model]["calls"] += 1
        self.total_cost += cost

    def get_summary(self) -> Dict[str, Any]:
        return {
            "by_model": self.usage,
            "total_cost": round(self.total_cost, 4),
            "total_calls": sum(m["calls"] for m in self.usage.values())
        }
```

---

## Part 5: Complete Production Pipeline

### Bringing It All Together

```python
# See code/13_production_pipeline.py for executable version

from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    cache_ttl: float = 3600
    max_retries: int = 3
    timeout: float = 30.0
    enable_quality_gates: bool = True
    enable_caching: bool = True
    log_level: str = "INFO"

class ProductionRAGPipeline:
    """Complete production-ready RAG pipeline."""

    def __init__(
        self,
        vector_store,
        llm,
        embeddings,
        config: PipelineConfig = None
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or PipelineConfig()

        # Initialize components
        self.cache = ProductionCache(embeddings)
        self.quality_gates = QualityPipeline(create_rag_quality_gates())
        self.metrics = Metrics()
        self.logger = PipelineLogger("rag-pipeline")
        self.cost_tracker = CostTracker()

    def query(self, question: str, user_id: str = None) -> Dict[str, Any]:
        """Process a query through the full pipeline."""

        tracer = Tracer()

        with tracer.span("total_request", question=question[:100]):

            # Step 1: Check cache
            with tracer.span("cache_lookup"):
                if self.config.enable_caching:
                    cached = self.cache.get(question)
                    if cached:
                        self.metrics.increment("cache_hit")
                        return {
                            "answer": cached,
                            "source": "cache",
                            "trace": tracer.to_dict()
                        }
                    self.metrics.increment("cache_miss")

            # Step 2: Retrieve context
            with tracer.span("retrieval") as span:
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 4}
                )
                docs = retriever.invoke(question)
                span.metadata["num_docs"] = len(docs)

            # Step 3: Format context
            with tracer.span("context_formatting"):
                context = self._format_context(docs)

            # Step 4: Quality check on retrieval
            if self.config.enable_quality_gates:
                with tracer.span("quality_check_retrieval"):
                    retrieval_data = {
                        "retrieved_docs": docs,
                        "relevance_scores": [0.8] * len(docs),  # Simplified
                        "context": context
                    }
                    passed, results = self.quality_gates.run(retrieval_data)
                    if not passed:
                        return {
                            "answer": "I couldn't find relevant information to answer your question.",
                            "quality_issues": [r.message for r in results if r.status.value == "failed"],
                            "trace": tracer.to_dict()
                        }

            # Step 5: Generate answer
            with tracer.span("llm_generation") as span:
                prompt = self._build_prompt(question, context)
                response = self.llm.invoke(prompt)
                answer = response.content

                # Track costs (simplified token counting)
                input_tokens = len(prompt.split()) * 1.3
                output_tokens = len(answer.split()) * 1.3
                self.cost_tracker.record_usage(
                    "gpt-4o-mini",
                    int(input_tokens),
                    int(output_tokens)
                )

            # Step 6: Quality check on answer
            if self.config.enable_quality_gates:
                with tracer.span("quality_check_answer"):
                    answer_data = {
                        "answer": answer,
                        "context": context,
                        "question": question
                    }
                    # Run answer-specific gates
                    passed, results = self.quality_gates.run(answer_data)

            # Step 7: Cache result
            if self.config.enable_caching:
                with tracer.span("cache_store"):
                    self.cache.set(question, answer, ttl=self.config.cache_ttl)

            # Record metrics
            self.metrics.histogram(
                "request_duration_ms",
                tracer.spans[0].duration_ms
            )
            self.metrics.increment("requests_total")

            # Log request
            self.logger.response(
                trace_id=tracer.trace_id,
                duration_ms=tracer.spans[0].duration_ms,
                cache_hit=False,
                num_docs=len(docs)
            )

            return {
                "answer": answer,
                "sources": [doc.metadata.get("source") for doc in docs],
                "trace": tracer.to_dict(),
                "metrics": {
                    "duration_ms": tracer.spans[0].duration_ms,
                    "docs_retrieved": len(docs)
                }
            }

    def _format_context(self, docs) -> str:
        """Format retrieved documents as context."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the RAG prompt."""
        return f"""Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say so.
Always cite your sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer:"""

    def get_health(self) -> Dict[str, Any]:
        """Get pipeline health status."""
        return {
            "status": "healthy",
            "cache_stats": self.cache.get_stats(),
            "metrics": self.metrics.summary(),
            "costs": self.cost_tracker.get_summary()
        }
```

---

## Analogical Thinking: The Factory Quality System

A production data pipeline is like a manufacturing quality control system:

| Factory Concept | Pipeline Equivalent |
|-----------------|---------------------|
| Incoming inspection | Input validation |
| Production line | Data processing chain |
| Quality checkpoints | Quality gates |
| Defect detection | Hallucination detection |
| Rework station | Fallback/retry logic |
| Final inspection | Output validation |
| Shipping | Response delivery |
| Metrics dashboard | Observability |
| Recall system | Cache invalidation |

---

## Emergence Thinking: From Chaos to Reliability

```
Without Production Patterns:
├── Random failures
├── Silent errors
├── Cost surprises
├── No visibility
└── Firefighting mode

With Production Patterns:
├── Predictable behavior
├── Graceful degradation
├── Cost control
├── Full visibility
└── Confidence and trust
```

**The emergence:**
- Individual components handle their own failures
- Quality gates prevent cascading errors
- Caching reduces load and cost
- Observability enables improvement
- Together they create a **reliable system**

---

## Summary

### What We Learned

1. **Production Reality**
   - Scale changes everything
   - Failures are inevitable
   - Cost matters at scale
   - Visibility is essential

2. **Multi-Source Integration**
   - Query sources in parallel
   - Use fallbacks for resilience
   - Route based on query type

3. **Caching Strategies**
   - Multi-layer caching reduces load
   - Semantic caching handles variations
   - Invalidation is as important as caching

4. **Quality Gates**
   - Validate at every step
   - Fail fast or degrade gracefully
   - Detect hallucinations

5. **Observability**
   - Metrics for quantitative understanding
   - Logs for debugging
   - Traces for flow analysis
   - Cost tracking for optimization

### Production Checklist

```
Before Going to Production:
├── [ ] Caching implemented and tested
├── [ ] Quality gates in place
├── [ ] Error handling comprehensive
├── [ ] Rate limiting configured
├── [ ] Fallbacks defined
├── [ ] Monitoring dashboard ready
├── [ ] Alerts configured
├── [ ] Cost tracking enabled
├── [ ] Load testing completed
└── [ ] Runbook documented
```

---

## Practice Exercises

### Exercise 1: Build Your Caching Layer
Implement:
- Response caching with TTL
- Semantic similarity cache
- Cache hit rate monitoring

### Exercise 2: Create Custom Quality Gates
Build gates for:
- PII detection
- Content policy compliance
- Response length validation

### Exercise 3: Full Observability
Add to your pipeline:
- Request/response logging
- Latency histograms
- Cost per query tracking

---

## Week 4 Complete!

Congratulations! You've learned to:
- Connect agents to real APIs
- Build RAG pipelines for document Q&A
- Create production-ready data systems

### The Journey Continues

```
Week 1: Foundations           ✅
Week 2: Build Your First Agent ✅
Week 3: LangGraph Workflows   ✅
Week 4: APIs & Real Data      ✅ ← YOU COMPLETED THIS!
Week 5: Advanced Tool Design  → Next
```

---

*"A production system is not just code that works—it's code that works when things go wrong, that tells you when it's struggling, and that you can trust at 3 AM."*
