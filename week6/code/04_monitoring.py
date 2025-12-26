"""
Module 6.3: Monitoring and Observability
=========================================
Complete monitoring system for AI agents:
- Metrics collection
- Distributed tracing
- Dashboard visualization
- Alerting
- Feedback collection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
import statistics
import time
import uuid
import json

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# METRICS COLLECTION
# ============================================================

@dataclass
class MetricPoint:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self):
        self.metrics: List[MetricPoint] = []
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """Increment a counter."""
        key = self._make_key(name, tags)
        self.counters[key] += value
        self._record(name, self.counters[key], tags)

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge value."""
        key = self._make_key(name, tags)
        self.gauges[key] = value
        self._record(name, value, tags)

    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        key = self._make_key(name, tags)
        self.histograms[key].append(value)
        self._record(name, value, tags)

    @contextmanager
    def timer(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations."""
        start = time.time()
        yield
        elapsed = (time.time() - start) * 1000  # ms
        self.histogram(name, elapsed, tags)

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
        """Get summary statistics for a histogram."""
        key = self._make_key(name, tags)
        values = self.histograms.get(key, [])

        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(len(sorted_values) * 0.95)] if len(sorted_values) > 1 else values[0],
            "p99": sorted_values[int(len(sorted_values) * 0.99)] if len(sorted_values) > 1 else values[0],
        }


# ============================================================
# AI AGENT METRICS
# ============================================================

class AIAgentMetrics:
    """Specialized metrics for AI agents."""

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
        """Record metrics for a request."""
        tags = {"model": model}
        if user_id:
            tags["user"] = user_id

        self.collector.histogram("request_latency_ms", latency_ms, tags)
        self.collector.histogram("input_tokens", input_tokens, tags)
        self.collector.histogram("output_tokens", output_tokens, tags)

        cost = self._calculate_cost(input_tokens, output_tokens, model)
        self.collector.histogram("request_cost_usd", cost, tags)

        if success:
            self.collector.increment("requests_success", tags=tags)
        else:
            self.collector.increment("requests_error", tags=tags)

        self.collector.increment("requests_total", tags=tags)

    def record_guardrail_trigger(self, guardrail_type: str, action: str):
        """Record guardrail trigger."""
        self.collector.increment("guardrail_triggers", tags={
            "type": guardrail_type,
            "action": action
        })

    def record_user_feedback(self, rating: int, user_id: str = None):
        """Record user feedback."""
        tags = {"user": user_id} if user_id else {}
        self.collector.histogram("user_rating", rating, tags)

    def get_error_rate(self) -> float:
        """Get current error rate."""
        total = self.collector.counters.get("requests_total", 0)
        errors = self.collector.counters.get("requests_error", 0)
        return errors / total if total > 0 else 0.0

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard."""
        return {
            "latency": self.collector.get_summary("request_latency_ms"),
            "tokens": {
                "input": self.collector.get_summary("input_tokens"),
                "output": self.collector.get_summary("output_tokens"),
            },
            "cost": self.collector.get_summary("request_cost_usd"),
            "error_rate": self.get_error_rate(),
            "total_requests": int(self.collector.counters.get("requests_total", 0)),
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost based on token usage."""
        pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
        }
        rates = pricing.get(model, pricing["gpt-4o-mini"])
        return (input_tokens * rates["input"]) + (output_tokens * rates["output"])


# ============================================================
# TRACING
# ============================================================

@dataclass
class Span:
    """A span in a trace."""
    span_id: str
    name: str
    trace_id: str
    parent_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


class Tracer:
    """Distributed tracing for AI agents."""

    def __init__(self):
        self.traces: Dict[str, List[Span]] = {}
        self.current_trace_id: Optional[str] = None
        self.current_span_id: Optional[str] = None

    @contextmanager
    def trace(self, name: str):
        """Start a new trace."""
        trace_id = str(uuid.uuid4())[:8]
        self.current_trace_id = trace_id
        self.traces[trace_id] = []

        with self.span(name) as span:
            yield span

        self.current_trace_id = None

    @contextmanager
    def span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a span."""
        if not self.current_trace_id:
            raise ValueError("No active trace")

        span = Span(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            trace_id=self.current_trace_id,
            parent_id=self.current_span_id,
            start_time=datetime.now(),
            attributes=attributes or {}
        )

        previous_span_id = self.current_span_id
        self.current_span_id = span.span_id

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            span.end_time = datetime.now()
            self.traces[self.current_trace_id].append(span)
            self.current_span_id = previous_span_id

    def visualize(self, trace_id: str) -> str:
        """Visualize a trace as ASCII."""
        spans = self.traces.get(trace_id, [])
        if not spans:
            return "No spans found"

        spans = sorted(spans, key=lambda s: s.start_time)
        lines = [f"Trace: {trace_id}", ""]

        def render_span(span: Span, indent: int = 0):
            prefix = "│   " * indent
            duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "running"
            icon = "✓" if span.status == "ok" else "✗"
            lines.append(f"{prefix}├── {icon} {span.name} ({duration})")

            for key, value in list(span.attributes.items())[:2]:
                lines.append(f"{prefix}│   └── {key}: {value}")

            children = [s for s in spans if s.parent_id == span.span_id]
            for child in children:
                render_span(child, indent + 1)

        roots = [s for s in spans if s.parent_id is None]
        for root in roots:
            render_span(root)

        return "\n".join(lines)


# ============================================================
# DASHBOARD
# ============================================================

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class AIDashboard:
    """Dashboard for AI agent monitoring."""

    def __init__(self, metrics: AIAgentMetrics):
        self.metrics = metrics

    def get_health(self) -> Dict[str, Any]:
        """Get health overview."""
        data = self.metrics.get_dashboard_data()

        # Determine health status
        error_rate = data.get("error_rate", 0)
        latency = data.get("latency", {}).get("p95", 0)

        if error_rate > 0.05 or latency > 10000:
            status = HealthStatus.CRITICAL
        elif error_rate > 0.02 or latency > 5000:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return {
            "status": status.value,
            "timestamp": datetime.now().isoformat(),
            "metrics": data
        }

    def render_ascii(self) -> str:
        """Render dashboard as ASCII."""
        health = self.get_health()
        data = health["metrics"]

        status_icons = {
            "healthy": "[OK]",
            "warning": "[!!]",
            "critical": "[XX]"
        }

        lines = [
            "=" * 60,
            f"AI AGENT DASHBOARD - {health['timestamp'][:19]}",
            "=" * 60,
            "",
            f"Status: {status_icons[health['status']]} {health['status'].upper()}",
            "",
            "─" * 60,
            f"Total Requests: {data.get('total_requests', 0)}",
            f"Error Rate: {data.get('error_rate', 0) * 100:.1f}%",
        ]

        latency = data.get("latency", {})
        if latency:
            lines.append(f"Latency (P95): {latency.get('p95', 0):.0f}ms")
            lines.append(f"Latency (Mean): {latency.get('mean', 0):.0f}ms")

        cost = data.get("cost", {})
        if cost:
            total_cost = cost.get("mean", 0) * cost.get("count", 0)
            lines.append(f"Total Cost: ${total_cost:.4f}")

        lines.append("─" * 60)
        return "\n".join(lines)


# ============================================================
# ALERTING
# ============================================================

@dataclass
class Alert:
    """An alert."""
    name: str
    severity: str
    message: str
    timestamp: datetime
    value: float


class AlertManager:
    """Manages alerts."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.rules = [
            {"name": "high_error_rate", "metric": "error_rate", "threshold": 0.05, "severity": "critical"},
            {"name": "high_latency", "metric": "latency_p95", "threshold": 10000, "severity": "warning"},
        ]

    def check(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check metrics against rules."""
        new_alerts = []

        for rule in self.rules:
            value = metrics.get(rule["metric"], 0)
            if value > rule["threshold"]:
                alert = Alert(
                    name=rule["name"],
                    severity=rule["severity"],
                    message=f"{rule['metric']} is {value}, exceeds threshold {rule['threshold']}",
                    timestamp=datetime.now(),
                    value=value
                )
                new_alerts.append(alert)
                self.alerts.append(alert)

        return new_alerts


# ============================================================
# COMPLETE OBSERVABILITY SYSTEM
# ============================================================

class ObservabilitySystem:
    """Complete observability system."""

    def __init__(self):
        self.metrics = AIAgentMetrics()
        self.tracer = Tracer()
        self.alerts = AlertManager()
        self.dashboard = AIDashboard(self.metrics)

    def process_with_observability(self, user_input: str) -> Dict[str, Any]:
        """Process a request with full observability."""

        with self.tracer.trace("request") as trace_span:
            trace_span.attributes["input_length"] = len(user_input)
            start_time = time.time()

            # Simulate processing
            with self.tracer.span("validation"):
                time.sleep(0.01)

            with self.tracer.span("llm_call") as llm_span:
                time.sleep(0.2)
                llm_span.attributes["model"] = "gpt-4o-mini"
                llm_span.attributes["tokens"] = 150

            with self.tracer.span("output_validation"):
                time.sleep(0.01)

            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.metrics.record_request(
                latency_ms=latency,
                input_tokens=100,
                output_tokens=50,
                model="gpt-4o-mini",
                success=True
            )

            return {
                "status": "success",
                "trace_id": self.tracer.current_trace_id,
                "latency_ms": latency
            }

    def get_report(self) -> str:
        """Generate system report."""
        lines = [
            self.dashboard.render_ascii(),
            "",
            "Recent Traces:",
        ]

        for trace_id in list(self.tracer.traces.keys())[-3:]:
            lines.append(self.tracer.visualize(trace_id))
            lines.append("")

        return "\n".join(lines)


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate monitoring and observability."""
    print("=" * 60)
    print("MONITORING AND OBSERVABILITY DEMO")
    print("=" * 60)

    obs = ObservabilitySystem()

    # Simulate some requests
    print("\nProcessing requests...")
    for i in range(10):
        result = obs.process_with_observability(f"Query {i}")
        if i == 0:
            print(f"  Request completed: {result['latency_ms']:.0f}ms")

    # Show dashboard
    print("\n" + obs.dashboard.render_ascii())

    # Show a trace
    print("\nSample Trace:")
    if obs.tracer.traces:
        trace_id = list(obs.tracer.traces.keys())[0]
        print(obs.tracer.visualize(trace_id))

    # Check alerts
    print("\n" + "─" * 60)
    print("ALERT CHECK")
    print("─" * 60)

    current_metrics = {
        "error_rate": 0.03,
        "latency_p95": 6000,
    }

    alerts = obs.alerts.check(current_metrics)
    if alerts:
        for alert in alerts:
            icon = "[!!]" if alert.severity == "critical" else "[!]"
            print(f"{icon} {alert.name}: {alert.message}")
    else:
        print("No alerts triggered")


if __name__ == "__main__":
    demo()
