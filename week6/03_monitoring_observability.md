# Module 6.3: Monitoring & Observability

> "You can't improve what you can't measure." — Peter Drucker

## What You'll Learn

- Why observability is critical for AI systems
- What metrics matter for LLM-based agents
- How to implement comprehensive tracing
- Building dashboards that drive action
- Alert design for anomaly detection
- Creating feedback loops for continuous improvement

---

## First Principles: What is Observability?

Observability is the ability to understand the internal state of a system by examining its external outputs.

### The Observability Equation

```
Observability = Metrics + Traces + Logs

Where:
├── METRICS: Aggregated measurements over time
│   ├── Counters (total requests, errors)
│   ├── Gauges (current active requests)
│   ├── Histograms (latency distribution)
│   └── Summaries (percentiles)
│
├── TRACES: Request paths through the system
│   ├── Spans (individual operations)
│   ├── Context propagation
│   ├── Parent-child relationships
│   └── Timing information
│
└── LOGS: Discrete events with context
    ├── Structured data (JSON)
    ├── Severity levels
    ├── Timestamps
    └── Correlation IDs
```

### Why AI Systems Need Special Observability

```
TRADITIONAL SOFTWARE                 AI SYSTEMS
───────────────────────────────────────────────────────────────────

Deterministic                        Non-deterministic
├── Same input = same output        ├── Same input ≠ same output
├── Bugs are reproducible           ├── Issues may be statistical
└── Testing is straightforward      └── Testing needs distributions

Clear failure modes                  Subtle failure modes
├── Crashes, exceptions             ├── Hallucinations
├── Timeouts                        ├── Bias
├── Wrong answers                   ├── Inappropriate responses
└── Easy to detect                  └── Hard to detect automatically

Static behavior                      Evolving behavior
├── Code determines behavior        ├── Model updates change behavior
├── Version = behavior              ├── Same version, different results
└── Deployment is clear             └── Need continuous monitoring

Fixed costs                          Variable costs
├── CPU/memory based                ├── Token-based billing
├── Predictable                     ├── Can spike unexpectedly
└── Easy to budget                  └── Need cost monitoring
```

---

## Analogical Thinking: Observability as Medical Monitoring

### The Hospital ICU Analogy

```
ICU PATIENT MONITORING               AI AGENT MONITORING
───────────────────────────────────────────────────────────────────

VITAL SIGNS                          CORE METRICS
┌─────────────────┐                 ┌─────────────────┐
│ Heart Rate      │                 │ Request Rate    │
│ Blood Pressure  │                 │ Error Rate      │
│ Temperature     │                 │ Latency         │
│ Oxygen Level    │                 │ Token Usage     │
└─────────────────┘                 └─────────────────┘

CONTINUOUS MONITORING                REAL-TIME DASHBOARDS
┌─────────────────┐                 ┌─────────────────┐
│ EKG Display     │                 │ Request Traces  │
│ Real-time graphs│                 │ Live metrics    │
│ Trend analysis  │                 │ Anomaly alerts  │
└─────────────────┘                 └─────────────────┘

ALARMS                               ALERTS
┌─────────────────┐                 ┌─────────────────┐
│ Heart rate <40  │                 │ Error rate >5%  │
│ BP critical     │                 │ Latency >10s    │
│ O2 dropping     │                 │ Cost spike      │
└─────────────────┘                 └─────────────────┘

MEDICAL RECORDS                      AUDIT LOGS
┌─────────────────┐                 ┌─────────────────┐
│ Patient history │                 │ Request history │
│ Treatment notes │                 │ Decision logs   │
│ Lab results     │                 │ Output traces   │
└─────────────────┘                 └─────────────────┘

ROUNDS & REVIEW                      FEEDBACK LOOPS
┌─────────────────┐                 ┌─────────────────┐
│ Daily rounds    │                 │ Quality review  │
│ Case review     │                 │ Incident review │
│ Treatment adjust│                 │ Model updates   │
└─────────────────┘                 └─────────────────┘
```

---

## Emergence Thinking: Insight from Simple Measurements

Complex understanding emerges from simple, consistent measurements:

```
SIMPLE MEASUREMENTS                  →  EMERGENT INSIGHTS
─────────────────────────────────────────────────────────────────────

"Log every request timestamp"        →  Usage patterns, peak times
"Count tokens per request"           →  Cost trends, optimization
"Record latency per endpoint"        →  Bottleneck identification
"Track error messages"               →  Root cause analysis
"Capture user feedback"              →  Quality improvement

          Combined, these enable:

          ┌──────────────────────────────────────────────┐
          │                                              │
          │   OPERATIONAL INTELLIGENCE                   │
          │                                              │
          │   • Predict capacity needs                   │
          │   • Identify quality degradation            │
          │   • Optimize cost efficiency                │
          │   • Detect anomalies early                  │
          │   • Guide improvement priorities            │
          │                                              │
          └──────────────────────────────────────────────┘
```

---

## Core Metrics for AI Agents

### The Golden Signals for AI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE GOLDEN SIGNALS FOR AI                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. LATENCY                                                                 │
│   ─────────                                                                  │
│   • Time to First Token (TTFT): How quickly response starts                 │
│   • Time to Complete (TTC): Total response time                             │
│   • Tool execution time: How long each tool takes                           │
│   • Queue time: How long requests wait                                      │
│                                                                              │
│   Why it matters: User experience, timeout handling, SLA compliance         │
│                                                                              │
│   2. ERROR RATE                                                              │
│   ────────────                                                               │
│   • API errors: Rate limit, auth failures, server errors                    │
│   • Guardrail blocks: Input/output validation failures                      │
│   • Tool failures: External service issues                                  │
│   • Parse errors: Malformed LLM outputs                                     │
│                                                                              │
│   Why it matters: Reliability, user trust, debugging                        │
│                                                                              │
│   3. TOKEN USAGE (Cost)                                                      │
│   ─────────────────────                                                      │
│   • Input tokens: Context size                                              │
│   • Output tokens: Response length                                          │
│   • Total cost: Dollar amount per request                                   │
│   • Cost per user/feature: Attribution                                      │
│                                                                              │
│   Why it matters: Budget control, optimization, pricing                     │
│                                                                              │
│   4. QUALITY                                                                 │
│   ────────                                                                   │
│   • User satisfaction: Thumbs up/down, ratings                              │
│   • Task completion: Did the agent achieve the goal?                        │
│   • Accuracy: For verifiable tasks                                          │
│   • Safety incidents: Guardrail triggers, complaints                        │
│                                                                              │
│   Why it matters: Product value, safety, improvement prioritization         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementing Metrics Collection

```python
# code/03_metrics.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time
import statistics

@dataclass
class MetricPoint:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """
    Collects and aggregates metrics for AI agents.
    """

    def __init__(self):
        self.metrics: List[MetricPoint] = []
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        key = self._make_key(name, tags)
        self.counters[key] += value
        self._record(name, self.counters[key], tags)

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        key = self._make_key(name, tags)
        self.gauges[key] = value
        self._record(name, value, tags)

    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric."""
        key = self._make_key(name, tags)
        self.histograms[key].append(value)
        self._record(name, value, tags)

    def timer(self, name: str, tags: Dict[str, str] = None):
        """Return a context manager for timing operations."""
        return TimerContext(self, name, tags)

    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def _record(self, name: str, value: float, tags: Dict[str, str] = None):
        self.metrics.append(MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        ))

    def get_summary(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get summary statistics for a histogram metric."""
        key = self._make_key(name, tags)
        values = self.histograms.get(key, [])

        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_rate(self, name: str, window_seconds: int = 60, tags: Dict[str, str] = None) -> float:
        """Calculate rate over time window."""
        cutoff = datetime.now() - timedelta(seconds=window_seconds)

        count = sum(
            1 for m in self.metrics
            if m.name == name
            and m.timestamp > cutoff
            and (not tags or all(m.tags.get(k) == v for k, v in tags.items()))
        )

        return count / window_seconds


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.collector.histogram(self.name, elapsed, self.tags)


# === AI-Specific Metrics ===

class AIAgentMetrics:
    """
    Specialized metrics for AI agents.
    """

    def __init__(self):
        self.collector = MetricsCollector()

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        model: str,
        success: bool,
        user_id: str = None
    ):
        """Record metrics for a single request."""

        tags = {"model": model}
        if user_id:
            tags["user"] = user_id

        # Latency
        self.collector.histogram("request_latency_ms", latency_ms, tags)

        # Tokens
        self.collector.histogram("input_tokens", input_tokens, tags)
        self.collector.histogram("output_tokens", output_tokens, tags)
        total_tokens = input_tokens + output_tokens
        self.collector.histogram("total_tokens", total_tokens, tags)

        # Cost (simplified - in production use actual pricing)
        cost = self._calculate_cost(input_tokens, output_tokens, model)
        self.collector.histogram("request_cost_usd", cost, tags)

        # Success/Error
        if success:
            self.collector.increment("requests_success", tags=tags)
        else:
            self.collector.increment("requests_error", tags=tags)

        self.collector.increment("requests_total", tags=tags)

    def record_guardrail_trigger(self, guardrail_type: str, action: str):
        """Record when a guardrail is triggered."""
        self.collector.increment("guardrail_triggers", tags={
            "type": guardrail_type,
            "action": action
        })

    def record_tool_execution(self, tool_name: str, latency_ms: float, success: bool):
        """Record tool execution metrics."""
        tags = {"tool": tool_name}
        self.collector.histogram("tool_latency_ms", latency_ms, tags)

        if success:
            self.collector.increment("tool_success", tags=tags)
        else:
            self.collector.increment("tool_error", tags=tags)

    def record_user_feedback(self, rating: int, user_id: str = None):
        """Record user satisfaction feedback."""
        tags = {}
        if user_id:
            tags["user"] = user_id

        self.collector.histogram("user_rating", rating, tags)

        # Track positive/negative
        if rating >= 4:
            self.collector.increment("feedback_positive", tags=tags)
        elif rating <= 2:
            self.collector.increment("feedback_negative", tags=tags)
        else:
            self.collector.increment("feedback_neutral", tags=tags)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        return {
            "latency": self.collector.get_summary("request_latency_ms"),
            "tokens": {
                "input": self.collector.get_summary("input_tokens"),
                "output": self.collector.get_summary("output_tokens"),
            },
            "cost": self.collector.get_summary("request_cost_usd"),
            "error_rate": self._calculate_error_rate(),
            "guardrail_triggers": self._get_guardrail_stats(),
            "user_satisfaction": self.collector.get_summary("user_rating"),
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost based on token usage."""
        # Simplified pricing (actual pricing varies by model)
        pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
            "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
        }

        rates = pricing.get(model, pricing["gpt-4o-mini"])
        return (input_tokens * rates["input"]) + (output_tokens * rates["output"])

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        total = self.collector.counters.get("requests_total", 0)
        errors = self.collector.counters.get("requests_error", 0)

        if total == 0:
            return 0.0
        return errors / total

    def _get_guardrail_stats(self) -> Dict[str, int]:
        """Get guardrail trigger statistics."""
        stats = {}
        for key, value in self.collector.counters.items():
            if key.startswith("guardrail_triggers"):
                stats[key] = int(value)
        return stats
```

---

## Tracing AI Agent Requests

Traces show the complete path of a request through your system.

### Trace Structure

```
TRACE STRUCTURE FOR AI AGENTS
─────────────────────────────────────────────────────────────────

Request Trace
│
├── Span: Input Validation (5ms)
│   ├── attribute: input_length = 150
│   └── attribute: validation_result = pass
│
├── Span: Injection Detection (10ms)
│   ├── attribute: patterns_checked = 15
│   └── attribute: threats_detected = 0
│
├── Span: LLM Call (2500ms)
│   ├── attribute: model = gpt-4
│   ├── attribute: input_tokens = 500
│   ├── attribute: output_tokens = 200
│   └── child spans:
│       ├── Span: Token Counting (2ms)
│       ├── Span: API Request (2450ms)
│       └── Span: Response Parsing (48ms)
│
├── Span: Tool Execution (500ms)
│   ├── attribute: tool = web_search
│   ├── attribute: success = true
│   └── child spans:
│       ├── Span: Query Building (5ms)
│       └── Span: External API (495ms)
│
├── Span: Output Validation (20ms)
│   ├── attribute: pii_detected = false
│   ├── attribute: toxicity_score = 0.1
│   └── attribute: validation_result = pass
│
└── Span: Response Delivery (5ms)
    ├── attribute: response_length = 350
    └── attribute: streaming = true

Total Duration: 3040ms
```

### Implementing Tracing

```python
# code/03_tracing.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import uuid
import time
import json

@dataclass
class Span:
    """A single span in a trace."""
    span_id: str
    name: str
    trace_id: str
    parent_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

class Tracer:
    """
    Distributed tracing for AI agents.
    """

    def __init__(self):
        self.traces: Dict[str, List[Span]] = {}
        self.current_trace_id: Optional[str] = None
        self.current_span_id: Optional[str] = None

    @contextmanager
    def trace(self, name: str):
        """Start a new trace."""
        trace_id = str(uuid.uuid4())
        self.current_trace_id = trace_id
        self.traces[trace_id] = []

        with self.span(name) as span:
            yield span

        self.current_trace_id = None

    @contextmanager
    def span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a span within the current trace."""
        if not self.current_trace_id:
            raise ValueError("No active trace. Use tracer.trace() first.")

        span = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            trace_id=self.current_trace_id,
            parent_id=self.current_span_id,
            start_time=datetime.now(),
            attributes=attributes or {}
        )

        # Set as current span
        previous_span_id = self.current_span_id
        self.current_span_id = span.span_id

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.attributes["error.message"] = str(e)
            span.attributes["error.type"] = type(e).__name__
            raise
        finally:
            span.end_time = datetime.now()
            self.traces[self.current_trace_id].append(span)
            self.current_span_id = previous_span_id

    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add an event to the current span."""
        if self.current_trace_id and self.current_span_id:
            for span in self.traces[self.current_trace_id]:
                if span.span_id == self.current_span_id:
                    span.events.append({
                        "name": name,
                        "timestamp": datetime.now(),
                        "attributes": attributes or {}
                    })
                    break

    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the current span."""
        if self.current_trace_id and self.current_span_id:
            for span in self.traces[self.current_trace_id]:
                if span.span_id == self.current_span_id:
                    span.attributes[key] = value
                    break

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return self.traces.get(trace_id, [])

    def export_trace(self, trace_id: str) -> str:
        """Export trace as JSON."""
        spans = self.get_trace(trace_id)
        return json.dumps([
            {
                "span_id": s.span_id,
                "name": s.name,
                "trace_id": s.trace_id,
                "parent_id": s.parent_id,
                "start_time": s.start_time.isoformat(),
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes,
                "events": [
                    {**e, "timestamp": e["timestamp"].isoformat()}
                    for e in s.events
                ],
                "status": s.status
            }
            for s in spans
        ], indent=2)

    def visualize_trace(self, trace_id: str) -> str:
        """Generate ASCII visualization of trace."""
        spans = self.get_trace(trace_id)
        if not spans:
            return "No spans found"

        # Sort by start time
        spans = sorted(spans, key=lambda s: s.start_time)

        # Build tree structure
        lines = [f"Trace: {trace_id}", ""]

        def render_span(span: Span, indent: int = 0):
            prefix = "│   " * indent
            duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "running"
            status_icon = "✓" if span.status == "ok" else "✗"

            lines.append(f"{prefix}├── {status_icon} {span.name} ({duration})")

            # Show key attributes
            for key, value in list(span.attributes.items())[:3]:
                lines.append(f"{prefix}│   └── {key}: {value}")

            # Find and render children
            children = [s for s in spans if s.parent_id == span.span_id]
            for child in children:
                render_span(child, indent + 1)

        # Find root spans (no parent)
        roots = [s for s in spans if s.parent_id is None]
        for root in roots:
            render_span(root)

        return "\n".join(lines)


# === LangSmith Integration ===

class LangSmithTracer:
    """
    Integration with LangSmith for production tracing.

    LangSmith provides:
    - Automatic trace capture
    - UI for visualization
    - Dataset management
    - Evaluation pipelines
    """

    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        # In production, this would initialize the LangSmith client
        # from langsmith import Client
        # self.client = Client()

    def setup(self):
        """Setup LangSmith tracing."""
        import os

        # Environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = self.project_name

        print(f"LangSmith tracing enabled for project: {self.project_name}")

    def create_feedback(self, run_id: str, score: float, comment: str = None):
        """Record feedback for a trace."""
        # In production:
        # self.client.create_feedback(
        #     run_id=run_id,
        #     key="user_score",
        #     score=score,
        #     comment=comment
        # )
        print(f"Feedback recorded for run {run_id}: score={score}")


# === Traced Agent ===

class TracedAgent:
    """
    An agent with built-in tracing.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.tracer = Tracer()
        self.metrics = AIAgentMetrics()

    def process(self, user_input: str) -> Dict[str, Any]:
        """Process a request with full tracing."""

        with self.tracer.trace("agent_request") as root_span:
            root_span.attributes["user_input_length"] = len(user_input)

            # Input validation
            with self.tracer.span("input_validation") as span:
                validation_result = self._validate_input(user_input)
                span.attributes["result"] = validation_result["status"]

                if validation_result["status"] == "blocked":
                    return {"status": "blocked", "reason": validation_result["reason"]}

            # LLM call
            with self.tracer.span("llm_call") as span:
                start_time = time.time()

                # Simulate LLM call
                response = self._call_llm(user_input)

                latency = (time.time() - start_time) * 1000
                span.attributes["model"] = response["model"]
                span.attributes["input_tokens"] = response["input_tokens"]
                span.attributes["output_tokens"] = response["output_tokens"]
                span.attributes["latency_ms"] = latency

                # Record metrics
                self.metrics.record_request(
                    latency_ms=latency,
                    input_tokens=response["input_tokens"],
                    output_tokens=response["output_tokens"],
                    model=response["model"],
                    success=True
                )

            # Output validation
            with self.tracer.span("output_validation") as span:
                validated = self._validate_output(response["content"])
                span.attributes["modifications"] = validated["modifications"]

            root_span.attributes["total_tokens"] = (
                response["input_tokens"] + response["output_tokens"]
            )

            return {
                "status": "success",
                "response": validated["content"],
                "trace_id": self.tracer.current_trace_id
            }

    def _validate_input(self, text: str) -> Dict[str, Any]:
        """Validate input (mock)."""
        time.sleep(0.005)  # Simulate processing
        return {"status": "ok"}

    def _call_llm(self, text: str) -> Dict[str, Any]:
        """Call LLM (mock)."""
        time.sleep(0.5)  # Simulate LLM latency
        return {
            "content": f"Response to: {text}",
            "model": "gpt-4o-mini",
            "input_tokens": len(text.split()) * 2,
            "output_tokens": 50
        }

    def _validate_output(self, text: str) -> Dict[str, Any]:
        """Validate output (mock)."""
        time.sleep(0.02)  # Simulate processing
        return {"content": text, "modifications": []}

    def get_trace_visualization(self, trace_id: str) -> str:
        """Get visualization of a trace."""
        return self.tracer.visualize_trace(trace_id)


# === Demo ===

def demo_tracing():
    """Demonstrate tracing."""
    print("=" * 60)
    print("TRACING DEMO")
    print("=" * 60)

    agent = TracedAgent()

    # Process a request
    result = agent.process("What is the capital of France?")

    print(f"\nResult: {result['status']}")
    print(f"Trace ID: {result.get('trace_id', 'N/A')}")

    # Show trace visualization
    if result.get('trace_id'):
        print("\n" + "─" * 60)
        print("TRACE VISUALIZATION")
        print("─" * 60)
        print(agent.get_trace_visualization(result['trace_id']))


if __name__ == "__main__":
    demo_tracing()
```

---

## Building Dashboards

Dashboards translate metrics into actionable insights.

### Dashboard Design Principles

```
DASHBOARD DESIGN PRINCIPLES
─────────────────────────────────────────────────────────────────

1. START WITH QUESTIONS, NOT METRICS
   ✗ "What should we show?"
   ✓ "What questions do operators need to answer?"

   Key questions for AI systems:
   • Is the system healthy right now?
   • Are users getting good responses?
   • Are we within budget?
   • Are there any safety concerns?

2. USE VISUAL HIERARCHY
   ┌─────────────────────────────────────────────────────────────┐
   │  TOP: Critical health indicators (red/green)               │
   │  ───────────────────────────────────────────────            │
   │  MIDDLE: Key trends and patterns                           │
   │  ───────────────────────────────────────────────            │
   │  BOTTOM: Detailed breakdowns and drill-downs               │
   └─────────────────────────────────────────────────────────────┘

3. FOLLOW THE 5-SECOND RULE
   Operators should be able to assess system health
   within 5 seconds of looking at the dashboard.

4. SHOW CONTEXT, NOT JUST NUMBERS
   ✗ "Error rate: 2.5%"
   ✓ "Error rate: 2.5% (baseline: 1.2%, threshold: 5%)"

5. ENABLE DRILL-DOWN
   Overview → Category → Individual instances
```

### Dashboard Implementation

```python
# code/03_dashboard.py

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class DashboardPanel:
    """A single panel in the dashboard."""
    title: str
    metric_name: str
    current_value: float
    threshold_warning: float
    threshold_critical: float
    trend: str  # "up", "down", "stable"
    context: str

    @property
    def status(self) -> HealthStatus:
        if self.current_value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.current_value >= self.threshold_warning:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY

class AIDashboard:
    """
    Dashboard for AI agent monitoring.
    """

    def __init__(self, metrics: 'AIAgentMetrics'):
        self.metrics = metrics

    def get_health_overview(self) -> Dict[str, Any]:
        """Get overall health status."""

        panels = [
            self._get_error_rate_panel(),
            self._get_latency_panel(),
            self._get_cost_panel(),
            self._get_safety_panel(),
        ]

        # Overall status is the worst of all panels
        statuses = [p.status for p in panels]
        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall = HealthStatus.WARNING
        else:
            overall = HealthStatus.HEALTHY

        return {
            "overall_status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "panels": [
                {
                    "title": p.title,
                    "value": p.current_value,
                    "status": p.status.value,
                    "trend": p.trend,
                    "context": p.context
                }
                for p in panels
            ]
        }

    def _get_error_rate_panel(self) -> DashboardPanel:
        """Get error rate panel."""
        data = self.metrics.get_dashboard_data()
        error_rate = data.get("error_rate", 0) * 100

        return DashboardPanel(
            title="Error Rate",
            metric_name="error_rate_percent",
            current_value=error_rate,
            threshold_warning=2.0,
            threshold_critical=5.0,
            trend=self._calculate_trend("error_rate"),
            context=f"{error_rate:.1f}% (warn: 2%, crit: 5%)"
        )

    def _get_latency_panel(self) -> DashboardPanel:
        """Get latency panel."""
        data = self.metrics.get_dashboard_data()
        latency = data.get("latency", {})
        p95 = latency.get("p95", 0)

        return DashboardPanel(
            title="Latency (P95)",
            metric_name="latency_p95_ms",
            current_value=p95,
            threshold_warning=5000,  # 5 seconds
            threshold_critical=10000,  # 10 seconds
            trend=self._calculate_trend("latency"),
            context=f"{p95:.0f}ms (warn: 5s, crit: 10s)"
        )

    def _get_cost_panel(self) -> DashboardPanel:
        """Get cost panel."""
        data = self.metrics.get_dashboard_data()
        cost = data.get("cost", {})
        total = cost.get("mean", 0) * cost.get("count", 0)

        # Thresholds would be based on budget
        return DashboardPanel(
            title="Cost (Today)",
            metric_name="cost_usd",
            current_value=total,
            threshold_warning=100,
            threshold_critical=500,
            trend=self._calculate_trend("cost"),
            context=f"${total:.2f} (warn: $100, crit: $500)"
        )

    def _get_safety_panel(self) -> DashboardPanel:
        """Get safety incidents panel."""
        data = self.metrics.get_dashboard_data()
        triggers = data.get("guardrail_triggers", {})
        total_triggers = sum(triggers.values())

        return DashboardPanel(
            title="Safety Triggers",
            metric_name="guardrail_triggers",
            current_value=total_triggers,
            threshold_warning=10,
            threshold_critical=50,
            trend=self._calculate_trend("guardrail_triggers"),
            context=f"{total_triggers} triggers (warn: 10, crit: 50)"
        )

    def _calculate_trend(self, metric: str) -> str:
        """Calculate trend for a metric."""
        # Simplified - in production compare time windows
        return "stable"

    def render_ascii(self) -> str:
        """Render dashboard as ASCII."""
        health = self.get_health_overview()

        status_icons = {
            "healthy": "[OK]",
            "warning": "[!!]",
            "critical": "[XX]"
        }

        lines = [
            "=" * 70,
            f"AI AGENT DASHBOARD - {health['timestamp'][:19]}",
            "=" * 70,
            "",
            f"Overall Status: {status_icons[health['overall_status']]} {health['overall_status'].upper()}",
            "",
            "─" * 70,
        ]

        for panel in health["panels"]:
            icon = status_icons[panel["status"]]
            trend_icon = {"up": "↑", "down": "↓", "stable": "→"}[panel["trend"]]

            lines.append(f"{icon} {panel['title']}: {panel['context']} {trend_icon}")

        lines.append("─" * 70)

        return "\n".join(lines)


# === Alert System ===

@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    metric: str
    condition: str  # "gt", "lt", "eq"
    threshold: float
    severity: str  # "info", "warning", "critical"
    message_template: str

@dataclass
class Alert:
    """An triggered alert."""
    rule_name: str
    severity: str
    message: str
    timestamp: datetime
    metric_value: float

class AlertManager:
    """
    Manages alerts for AI system monitoring.
    """

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []

        # Default rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default alert rules."""
        self.rules = [
            AlertRule(
                name="high_error_rate",
                metric="error_rate",
                condition="gt",
                threshold=0.05,
                severity="critical",
                message_template="Error rate is {value:.1%}, exceeds threshold of {threshold:.1%}"
            ),
            AlertRule(
                name="high_latency",
                metric="latency_p95",
                condition="gt",
                threshold=10000,
                severity="warning",
                message_template="P95 latency is {value:.0f}ms, exceeds threshold of {threshold:.0f}ms"
            ),
            AlertRule(
                name="cost_spike",
                metric="hourly_cost",
                condition="gt",
                threshold=50,
                severity="warning",
                message_template="Hourly cost is ${value:.2f}, exceeds threshold of ${threshold:.2f}"
            ),
            AlertRule(
                name="safety_incidents",
                metric="guardrail_triggers_hourly",
                condition="gt",
                threshold=20,
                severity="critical",
                message_template="Safety triggers: {value:.0f} in last hour, exceeds threshold of {threshold:.0f}"
            ),
        ]

    def check_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check metrics against rules and generate alerts."""
        new_alerts = []

        for rule in self.rules:
            if rule.metric not in metrics:
                continue

            value = metrics[rule.metric]
            triggered = self._check_condition(value, rule.condition, rule.threshold)

            if triggered:
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=rule.message_template.format(
                        value=value,
                        threshold=rule.threshold
                    ),
                    timestamp=datetime.now(),
                    metric_value=value
                )
                new_alerts.append(alert)
                self.active_alerts.append(alert)
                self.alert_history.append(alert)

        return new_alerts

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if condition is met."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return self.active_alerts

    def acknowledge_alert(self, rule_name: str):
        """Acknowledge and clear an alert."""
        self.active_alerts = [
            a for a in self.active_alerts
            if a.rule_name != rule_name
        ]


# === Demo ===

def demo_dashboard():
    """Demonstrate dashboard and alerting."""
    print("=" * 60)
    print("DASHBOARD AND ALERTING DEMO")
    print("=" * 60)

    # Create metrics and record some data
    metrics = AIAgentMetrics()

    # Simulate some requests
    for i in range(100):
        metrics.record_request(
            latency_ms=500 + (i * 10),  # Increasing latency
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o-mini",
            success=i % 20 != 0,  # 5% error rate
        )

    # Add some guardrail triggers
    for _ in range(15):
        metrics.record_guardrail_trigger("injection_detection", "blocked")

    # Create and display dashboard
    dashboard = AIDashboard(metrics)
    print("\n" + dashboard.render_ascii())

    # Check alerts
    print("\n" + "─" * 70)
    print("ALERT CHECK")
    print("─" * 70)

    alert_manager = AlertManager()
    current_metrics = {
        "error_rate": 0.05,
        "latency_p95": 1500,
        "hourly_cost": 25,
        "guardrail_triggers_hourly": 15,
    }

    alerts = alert_manager.check_metrics(current_metrics)

    if alerts:
        for alert in alerts:
            icon = {"critical": "[!!]", "warning": "[!]", "info": "[i]"}[alert.severity]
            print(f"{icon} {alert.rule_name}: {alert.message}")
    else:
        print("No alerts triggered")


if __name__ == "__main__":
    demo_dashboard()
```

---

## Feedback Loops for Continuous Improvement

The goal of observability is not just monitoring—it's improvement.

### The Improvement Cycle

```
THE CONTINUOUS IMPROVEMENT CYCLE
─────────────────────────────────────────────────────────────────

        ┌─────────────────────────────────────────────────┐
        │                                                 │
        ▼                                                 │
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ MEASURE │ ──► │ ANALYZE │ ──► │ IMPROVE │ ──► │ DEPLOY  │
   │         │     │         │     │         │     │         │
   │ Collect │     │ Identify│     │ Design  │     │ Release │
   │ metrics,│     │ patterns│     │ fixes & │     │ changes │
   │ traces, │     │ & root  │     │ enhance-│     │         │
   │ feedback│     │ causes  │     │ ments   │     │         │
   └─────────┘     └─────────┘     └─────────┘     └────┬────┘
        ▲                                               │
        │                                               │
        └───────────────────────────────────────────────┘


FEEDBACK SOURCES:
• User ratings and comments
• Guardrail trigger analysis
• Error pattern analysis
• Cost optimization opportunities
• Performance bottleneck identification
```

### Implementing Feedback Collection

```python
# code/03_feedback.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import json

class FeedbackType(Enum):
    THUMBS = "thumbs"        # Simple thumbs up/down
    RATING = "rating"        # 1-5 star rating
    CORRECTION = "correction"  # User corrected the output
    REPORT = "report"        # User reported an issue

@dataclass
class UserFeedback:
    """User feedback on an agent response."""
    feedback_id: str
    trace_id: str
    user_id: Optional[str]
    feedback_type: FeedbackType
    value: Any  # thumbs: bool, rating: int, correction: str, report: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeedbackAnalysis:
    """Analysis of feedback patterns."""
    period_start: datetime
    period_end: datetime
    total_feedback: int
    positive_rate: float
    common_issues: List[Dict[str, Any]]
    improvement_suggestions: List[str]

class FeedbackCollector:
    """
    Collects and analyzes user feedback for continuous improvement.
    """

    def __init__(self):
        self.feedback: List[UserFeedback] = []
        self.trace_feedback_map: Dict[str, List[UserFeedback]] = {}

    def record_feedback(
        self,
        trace_id: str,
        feedback_type: FeedbackType,
        value: Any,
        user_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Record user feedback."""
        import uuid

        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            trace_id=trace_id,
            user_id=user_id,
            feedback_type=feedback_type,
            value=value,
            timestamp=datetime.now(),
            context=context or {}
        )

        self.feedback.append(feedback)

        if trace_id not in self.trace_feedback_map:
            self.trace_feedback_map[trace_id] = []
        self.trace_feedback_map[trace_id].append(feedback)

        return feedback.feedback_id

    def get_satisfaction_rate(self, window_hours: int = 24) -> float:
        """Calculate satisfaction rate over time window."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [f for f in self.feedback if f.timestamp > cutoff]

        if not recent:
            return 0.0

        positive = 0
        total = 0

        for fb in recent:
            if fb.feedback_type == FeedbackType.THUMBS:
                total += 1
                if fb.value:  # True = thumbs up
                    positive += 1
            elif fb.feedback_type == FeedbackType.RATING:
                total += 1
                if fb.value >= 4:  # 4-5 stars = positive
                    positive += 1

        return positive / total if total > 0 else 0.0

    def get_common_issues(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify common issues from feedback."""
        from collections import Counter

        issues = []

        # Collect reported issues
        for fb in self.feedback:
            if fb.feedback_type == FeedbackType.REPORT:
                issues.append(fb.value)
            elif fb.feedback_type == FeedbackType.THUMBS and not fb.value:
                # Negative feedback with context
                if fb.context.get("issue"):
                    issues.append(fb.context["issue"])

        # Count and rank
        counter = Counter(issues)
        return [
            {"issue": issue, "count": count}
            for issue, count in counter.most_common(top_n)
        ]

    def analyze(self, window_hours: int = 168) -> FeedbackAnalysis:
        """Generate feedback analysis report."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [f for f in self.feedback if f.timestamp > cutoff]

        return FeedbackAnalysis(
            period_start=cutoff,
            period_end=datetime.now(),
            total_feedback=len(recent),
            positive_rate=self.get_satisfaction_rate(window_hours),
            common_issues=self.get_common_issues(),
            improvement_suggestions=self._generate_suggestions()
        )

    def _generate_suggestions(self) -> List[str]:
        """Generate improvement suggestions based on feedback."""
        suggestions = []
        issues = self.get_common_issues(5)

        for issue in issues:
            if "slow" in issue["issue"].lower():
                suggestions.append("Consider optimizing latency-heavy operations")
            if "wrong" in issue["issue"].lower() or "incorrect" in issue["issue"].lower():
                suggestions.append("Review and improve response accuracy")
            if "rude" in issue["issue"].lower() or "tone" in issue["issue"].lower():
                suggestions.append("Adjust system prompt for better tone")

        return suggestions


# === Incident Response ===

@dataclass
class Incident:
    """A safety or quality incident."""
    incident_id: str
    severity: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    trace_ids: List[str]
    timestamp: datetime
    status: str = "open"  # "open", "investigating", "resolved"
    resolution: Optional[str] = None

class IncidentManager:
    """
    Manages incident tracking and response.
    """

    def __init__(self):
        self.incidents: List[Incident] = []

    def create_incident(
        self,
        severity: str,
        title: str,
        description: str,
        trace_ids: List[str] = None
    ) -> Incident:
        """Create a new incident."""
        import uuid

        incident = Incident(
            incident_id=str(uuid.uuid4())[:8],
            severity=severity,
            title=title,
            description=description,
            trace_ids=trace_ids or [],
            timestamp=datetime.now()
        )

        self.incidents.append(incident)
        self._notify_on_call(incident)

        return incident

    def update_status(self, incident_id: str, status: str, resolution: str = None):
        """Update incident status."""
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.status = status
                if resolution:
                    incident.resolution = resolution
                break

    def get_open_incidents(self) -> List[Incident]:
        """Get all open incidents."""
        return [i for i in self.incidents if i.status != "resolved"]

    def _notify_on_call(self, incident: Incident):
        """Notify on-call personnel."""
        # In production, this would send PagerDuty, Slack, etc.
        severity_icon = {
            "critical": "[!!!]",
            "high": "[!!]",
            "medium": "[!]",
            "low": "[i]"
        }

        print(f"\n{severity_icon[incident.severity]} NEW INCIDENT: {incident.title}")
        print(f"   Severity: {incident.severity}")
        print(f"   ID: {incident.incident_id}")

    def generate_postmortem(self, incident_id: str) -> str:
        """Generate incident postmortem template."""
        incident = None
        for i in self.incidents:
            if i.incident_id == incident_id:
                incident = i
                break

        if not incident:
            return "Incident not found"

        return f"""
# Incident Postmortem: {incident.title}

## Summary
- **Incident ID**: {incident.incident_id}
- **Severity**: {incident.severity}
- **Duration**: [START TIME] to [END TIME]
- **Status**: {incident.status}

## Timeline
- {incident.timestamp.isoformat()}: Incident detected
- [Add key events here]

## Root Cause
[Describe the root cause]

## Impact
[Describe user impact]

## Resolution
{incident.resolution or "[Describe how it was resolved]"}

## Lessons Learned
1. What went well?
2. What could be improved?
3. What action items do we have?

## Action Items
- [ ] [Action item 1]
- [ ] [Action item 2]

## Related Traces
{chr(10).join(f"- {tid}" for tid in incident.trace_ids)}
"""


# === Complete Observability System ===

class ObservabilitySystem:
    """
    Complete observability system combining all components.
    """

    def __init__(self):
        self.metrics = AIAgentMetrics()
        self.tracer = Tracer()
        self.feedback = FeedbackCollector()
        self.incidents = IncidentManager()
        self.dashboard = None

    def initialize(self):
        """Initialize the observability system."""
        self.dashboard = AIDashboard(self.metrics)
        print("Observability system initialized")
        print("- Metrics collection: Active")
        print("- Tracing: Active")
        print("- Feedback collection: Active")
        print("- Incident management: Active")

    def get_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "health": self.dashboard.get_health_overview() if self.dashboard else None,
            "feedback": {
                "satisfaction_rate": self.feedback.get_satisfaction_rate(),
                "total_feedback": len(self.feedback.feedback),
                "common_issues": self.feedback.get_common_issues(5)
            },
            "incidents": {
                "open": len(self.incidents.get_open_incidents()),
                "total": len(self.incidents.incidents)
            },
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations."""
        recommendations = []

        # Check satisfaction rate
        sat_rate = self.feedback.get_satisfaction_rate()
        if sat_rate < 0.8:
            recommendations.append(
                f"User satisfaction ({sat_rate:.0%}) is below target (80%). "
                "Review common issues and recent feedback."
            )

        # Check for open incidents
        open_incidents = self.incidents.get_open_incidents()
        if open_incidents:
            recommendations.append(
                f"There are {len(open_incidents)} open incident(s) requiring attention."
            )

        return recommendations


# === Demo ===

def demo_feedback_and_incidents():
    """Demonstrate feedback collection and incident management."""
    print("=" * 60)
    print("FEEDBACK AND INCIDENT MANAGEMENT DEMO")
    print("=" * 60)

    # Initialize system
    obs = ObservabilitySystem()
    obs.initialize()

    # Simulate some feedback
    print("\n" + "─" * 60)
    print("Recording feedback...")

    obs.feedback.record_feedback(
        trace_id="trace-001",
        feedback_type=FeedbackType.THUMBS,
        value=True
    )
    obs.feedback.record_feedback(
        trace_id="trace-002",
        feedback_type=FeedbackType.RATING,
        value=5
    )
    obs.feedback.record_feedback(
        trace_id="trace-003",
        feedback_type=FeedbackType.THUMBS,
        value=False,
        context={"issue": "Response was too slow"}
    )
    obs.feedback.record_feedback(
        trace_id="trace-004",
        feedback_type=FeedbackType.REPORT,
        value="Response contained incorrect information"
    )

    print(f"Satisfaction rate: {obs.feedback.get_satisfaction_rate():.0%}")

    # Create an incident
    print("\n" + "─" * 60)
    print("Creating incident...")

    incident = obs.incidents.create_incident(
        severity="high",
        title="Elevated error rate detected",
        description="Error rate increased to 8% over the last hour",
        trace_ids=["trace-005", "trace-006", "trace-007"]
    )

    # Generate system report
    print("\n" + "─" * 60)
    print("SYSTEM REPORT")
    print("─" * 60)

    report = obs.get_system_report()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    demo_feedback_and_incidents()
```

---

## Key Takeaways

### 1. Observability is Not Optional for AI
AI systems have unique failure modes (hallucinations, bias, subtle quality degradation) that require specialized monitoring.

### 2. Measure the Four Golden Signals
Track latency, error rate, token usage (cost), and quality metrics to understand system health.

### 3. Traces Tell the Story
When something goes wrong, traces show you exactly what happened. Invest in comprehensive tracing.

### 4. Dashboards Should Answer Questions
Design dashboards around the questions operators need to answer, not around the metrics you have.

### 5. Close the Feedback Loop
Observability without action is just data collection. Build systems that turn insights into improvements.

### 6. Plan for Incidents
Have incident response procedures ready. When things go wrong, you need a playbook.

---

## What You've Learned in Week 6

```
WEEK 6: GUARDRAILS AND SAFETY - COMPLETE
─────────────────────────────────────────────────────────────────

Module 6.1: Responsible AI Practices
├── AI ethics frameworks
├── Regulatory compliance (GDPR, EU AI Act)
├── Transparency and explainability
└── Human-in-the-loop patterns

Module 6.2: Guardrails Implementation
├── Input validation and sanitization
├── Prompt injection defense
├── Output validation and filtering
└── Hallucination detection

Module 6.3: Monitoring & Observability
├── Core metrics for AI systems
├── Distributed tracing
├── Dashboard design
└── Feedback loops and incident management

YOU CAN NOW:
✓ Design AI systems with built-in ethical constraints
✓ Implement defense-in-depth against attacks and failures
✓ Monitor AI systems for health, quality, and safety
✓ Create feedback loops for continuous improvement
✓ Respond to incidents systematically
```

---

## Next Steps

With Week 6 complete, you have the foundation for building **trustworthy AI agents**. The journey continues:

- **Week 7: Deployment & Scaling** - Take your safe, monitored agents to production
- **Week 8: Capstone Project** - Build an end-to-end AI agent system

Remember: **The safest AI system is one that knows its limitations and fails gracefully.** The techniques you've learned this week are not overhead—they're what makes AI systems production-ready.

**Build AI that users can trust!**
