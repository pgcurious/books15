"""
Module 4.3: Production Pipeline Components
==========================================
Demonstrates production-ready patterns:
- Multi-layer caching
- Quality gates
- Observability (metrics, logging, tracing)
- Cost tracking
"""

import time
import hashlib
import json
import logging
from typing import Optional, Any, Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from enum import Enum
import uuid


# =============================================================================
# Caching Layer
# =============================================================================

@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    value: Any
    created_at: float
    ttl_seconds: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl_seconds


class InMemoryCache:
    """Fast in-memory cache with TTL and LRU eviction."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        entry = self.cache.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self.cache[key]
            return None
        entry.hit_count += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: float = 3600) -> None:
        """Set a value in cache."""
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove oldest entry
            oldest_key = min(self.cache, key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]

        self.cache[key] = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl
        )

    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "entries": len(self.cache)
        }


# =============================================================================
# Quality Gates
# =============================================================================

class QualityStatus(Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class QualityResult:
    """Result of a quality gate check."""
    gate_name: str
    status: QualityStatus
    message: str
    details: Optional[Dict] = None


class QualityGate:
    """A single quality check."""

    def __init__(self, name: str, check_fn: Callable, severity: str = "error"):
        self.name = name
        self.check_fn = check_fn
        self.severity = severity

    def check(self, data: Any) -> QualityResult:
        """Run the quality check."""
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
        """Run all quality gates."""
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


def create_rag_quality_gates() -> List[QualityGate]:
    """Create standard quality gates for RAG pipelines."""

    def check_context_length(data: Dict) -> tuple[bool, str]:
        context = data.get("context", "")
        length = len(context.split())
        if length < 20:
            return False, f"Context too short ({length} words)"
        if length > 4000:
            return False, f"Context too long ({length} words)"
        return True, f"Context length OK ({length} words)"

    def check_answer_length(data: Dict) -> tuple[bool, str]:
        answer = data.get("answer", "")
        length = len(answer.split())
        if length < 5:
            return False, f"Answer too short ({length} words)"
        if length > 1000:
            return False, f"Answer too long ({length} words)"
        return True, f"Answer length OK ({length} words)"

    def check_no_empty_fields(data: Dict) -> tuple[bool, str]:
        required = ["question", "context", "answer"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            return False, f"Missing fields: {', '.join(missing)}"
        return True, "All required fields present"

    return [
        QualityGate("context_length", check_context_length),
        QualityGate("answer_length", check_answer_length),
        QualityGate("required_fields", check_no_empty_fields),
    ]


# =============================================================================
# Observability
# =============================================================================

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
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.spans: List[Span] = []
        self.current_span: Optional[Span] = None

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
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "status": s.status,
                    "metadata": s.metadata,
                    "error": s.error
                }
                for s in self.spans
            ],
            "total_duration_ms": round(sum(s.duration_ms for s in self.spans), 2)
        }


class Metrics:
    """Collects and exposes metrics."""

    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.gauges: Dict[str, float] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def histogram(self, name: str, value: float) -> None:
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def get_percentile(self, name: str, percentile: float) -> Optional[float]:
        values = self.histograms.get(name, [])
        if not values:
            return None
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def summary(self) -> Dict[str, Any]:
        result = {
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": {}
        }
        for name, values in self.histograms.items():
            if values:
                result["histograms"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": self.get_percentile(name, 50),
                    "p95": self.get_percentile(name, 95),
                    "p99": self.get_percentile(name, 99),
                }
        return result


class CostTracker:
    """Track API and token costs."""

    PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "text-embedding-3-small": {"input": 0.00002},
        "text-embedding-3-large": {"input": 0.00013},
    }

    def __init__(self):
        self.usage: Dict[str, Dict] = {}
        self.total_cost = 0.0

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0
    ) -> float:
        """Record token usage and return cost."""
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

        return cost

    def get_summary(self) -> Dict[str, Any]:
        return {
            "by_model": self.usage,
            "total_cost": round(self.total_cost, 6),
            "total_calls": sum(m["calls"] for m in self.usage.values())
        }


# =============================================================================
# Demo
# =============================================================================

def main():
    """Demonstrate production pipeline components."""
    print("=" * 60)
    print("Production Pipeline Components Demo")
    print("=" * 60)

    # Demo 1: Caching
    print("\n--- Caching Demo ---")
    cache = InMemoryCache(max_size=100)

    cache.set("query:weather:london", "Sunny, 22°C", ttl=3600)
    cache.set("query:crypto:bitcoin", "$45,000", ttl=60)

    print(f"Cache hit: {cache.get('query:weather:london')}")
    print(f"Cache miss: {cache.get('query:nonexistent')}")
    print(f"Cache stats: {cache.stats()}")

    # Demo 2: Quality Gates
    print("\n--- Quality Gates Demo ---")
    gates = create_rag_quality_gates()
    pipeline = QualityPipeline(gates)

    # Good data
    good_data = {
        "question": "What is the weather?",
        "context": "The weather today is sunny with temperatures around 22 degrees Celsius. " * 5,
        "answer": "The weather is sunny and warm at 22°C."
    }
    passed, results = pipeline.run(good_data)
    print(f"Good data passed: {passed}")
    for r in results:
        print(f"  {r.gate_name}: {r.status.value} - {r.message}")

    # Bad data
    bad_data = {
        "question": "What?",
        "context": "Short",
        "answer": ""
    }
    passed, results = pipeline.run(bad_data)
    print(f"\nBad data passed: {passed}")
    for r in results:
        print(f"  {r.gate_name}: {r.status.value} - {r.message}")

    # Demo 3: Tracing
    print("\n--- Tracing Demo ---")
    tracer = Tracer()

    with tracer.span("total_request"):
        with tracer.span("cache_lookup"):
            time.sleep(0.01)  # Simulated cache lookup

        with tracer.span("retrieval", num_docs=5):
            time.sleep(0.05)  # Simulated retrieval

        with tracer.span("llm_generation", model="gpt-4o-mini"):
            time.sleep(0.1)  # Simulated generation

    print(json.dumps(tracer.to_dict(), indent=2))

    # Demo 4: Metrics
    print("\n--- Metrics Demo ---")
    metrics = Metrics()

    # Simulate some requests
    for i in range(100):
        metrics.increment("requests_total")
        metrics.histogram("latency_ms", 50 + (i % 50))  # 50-100ms
        if i % 10 == 0:
            metrics.increment("cache_hits")
        else:
            metrics.increment("cache_misses")

    metrics.gauge("active_connections", 42)

    print(json.dumps(metrics.summary(), indent=2))

    # Demo 5: Cost Tracking
    print("\n--- Cost Tracking Demo ---")
    cost_tracker = CostTracker()

    # Simulate some API calls
    cost_tracker.record_usage("gpt-4o-mini", input_tokens=500, output_tokens=200)
    cost_tracker.record_usage("gpt-4o-mini", input_tokens=300, output_tokens=150)
    cost_tracker.record_usage("text-embedding-3-small", input_tokens=1000)

    print(json.dumps(cost_tracker.get_summary(), indent=2))

    print("\n" + "=" * 60)
    print("Production Pipeline Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
