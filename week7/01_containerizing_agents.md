# Module 7.1: Containerizing Agents

> "Build once, run anywhere." — Docker's promise (and how to actually achieve it)

## What You'll Learn

- Why containerization is essential for AI agents
- Building FastAPI wrappers that expose agents as APIs
- Creating production-ready Dockerfiles with multi-stage builds
- Implementing health checks and graceful shutdown
- Testing containers locally before deployment
- Best practices for AI-specific container optimization

---

## First Principles: Why Containers for AI Agents?

### The Deployment Problem

Before containers, deploying software meant:

```
TRADITIONAL DEPLOYMENT NIGHTMARE
─────────────────────────────────────────────────────────────────

Developer Machine                Production Server
┌─────────────────────┐         ┌─────────────────────┐
│ Python 3.11         │         │ Python 3.8          │ ← Version mismatch!
│ PyTorch 2.0         │   ???   │ PyTorch 1.9         │ ← Different version!
│ Ubuntu 22.04        │ ──────► │ CentOS 7            │ ← Different OS!
│ CUDA 12.0           │         │ CUDA 11.4           │ ← GPU mismatch!
│ My custom config    │         │ ???                 │ ← Lost in translation
└─────────────────────┘         └─────────────────────┘

Result: "It works on my machine!" 🤷
```

### The Container Solution

Containers package everything together:

```
CONTAINERIZED DEPLOYMENT
─────────────────────────────────────────────────────────────────

┌───────────────────────────────────────────────────────────────┐
│                        CONTAINER IMAGE                         │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │  Your Agent Code                                         │ │
│   │  ├── agent.py                                            │ │
│   │  ├── tools.py                                            │ │
│   │  └── prompts/                                            │ │
│   └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │  Dependencies (exactly as tested)                        │ │
│   │  ├── langchain==0.1.0                                    │ │
│   │  ├── openai==1.6.0                                       │ │
│   │  └── fastapi==0.109.0                                    │ │
│   └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │  Runtime Environment                                     │ │
│   │  ├── Python 3.11.7                                       │ │
│   │  ├── System libraries                                    │ │
│   │  └── Configuration                                       │ │
│   └─────────────────────────────────────────────────────────┘ │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
                              │
                              │ Runs identically on:
                              ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Your       │    │  AWS ECS    │    │  GCP Cloud  │
    │  Laptop     │    │             │    │  Run        │
    └─────────────┘    └─────────────┘    └─────────────┘
```

### First Principles Summary

```
CONTAINER = ISOLATED_FILESYSTEM + ISOLATED_PROCESSES + PORTABLE_FORMAT

At the atomic level:
├── Code: Your agent implementation
├── Dependencies: Exact versions of all libraries
├── Runtime: The language runtime (Python)
├── Config: Environment variables, settings
└── Entrypoint: How to start the application
```

---

## Analogical Thinking: Containers as Shipping Containers

```
SHIPPING INDUSTRY                 SOFTWARE CONTAINERS
───────────────────────────────────────────────────────────────────

Standard size container           Standard container format
├── Any cargo fits inside         ├── Any application fits inside
├── Known dimensions              ├── Known interface (ports, etc.)
└── Works on any ship/truck       └── Works on any host

Seal once, trust everywhere       Build once, run anywhere
├── Sealed at origin              ├── Built on dev machine
├── Not opened during transit     ├── Not modified during deploy
└── Contents guaranteed           └── Behavior guaranteed

Port infrastructure               Container orchestration
├── Cranes handle any container   ├── K8s/ECS handle any container
├── Standard loading/unloading    ├── Standard start/stop
└── Track containers globally     └── Monitor containers globally

Shipping manifest                 Dockerfile
├── Contents list                 ├── Dependencies list
├── Origin                        ├── Base image
├── Handling instructions         ├── Build instructions
└── Destination requirements      └── Runtime requirements
```

**Key insight**: Just as standardized shipping containers revolutionized global trade, software containers revolutionized deployment by creating a universal, portable unit of software.

---

## Building FastAPI Wrappers

Before containerizing, we need to wrap our agents in an API. FastAPI is the ideal choice for AI agents.

### Why FastAPI?

```
FASTAPI ADVANTAGES FOR AI AGENTS
─────────────────────────────────────────────────────────────────

1. ASYNC NATIVE
   ├── Non-blocking I/O for LLM calls
   ├── Handle many concurrent requests
   └── Efficient streaming support

2. AUTOMATIC DOCUMENTATION
   ├── OpenAPI/Swagger UI built-in
   ├── Self-documenting endpoints
   └── Easy testing and exploration

3. PYDANTIC VALIDATION
   ├── Type-safe request/response
   ├── Automatic error messages
   └── Schema generation

4. PRODUCTION READY
   ├── High performance (Starlette + Uvicorn)
   ├── Easy to add middleware
   └── Built-in dependency injection
```

### Basic Agent API Wrapper

```python
# code/01_basic_api.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# === Pydantic Models ===

class AgentRequest(BaseModel):
    """Request to the agent."""
    query: str = Field(..., description="The user's query", min_length=1)
    context: Optional[dict] = Field(default=None, description="Additional context")
    stream: bool = Field(default=False, description="Whether to stream the response")

class AgentResponse(BaseModel):
    """Response from the agent."""
    response: str = Field(..., description="The agent's response")
    tokens_used: int = Field(..., description="Tokens consumed")
    model: str = Field(..., description="Model used")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    dependencies: dict

# === Agent Implementation ===

class SimpleAgent:
    """A simple agent for demonstration."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Be concise and accurate."),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm

    async def invoke(self, query: str, context: Optional[dict] = None) -> dict:
        """Invoke the agent with a query."""
        response = await self.chain.ainvoke({"query": query})
        return {
            "response": response.content,
            "tokens_used": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
            "model": "gpt-4o-mini"
        }

    async def stream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream the agent's response."""
        async for chunk in self.chain.astream({"query": query}):
            if chunk.content:
                yield chunk.content

# === FastAPI Application ===

app = FastAPI(
    title="AI Agent API",
    description="Production-ready AI agent API",
    version="1.0.0"
)

# Initialize agent at startup
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global agent
    agent = SimpleAgent()
    print("Agent initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down gracefully")

# === Endpoints ===

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and orchestrators."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        dependencies={
            "llm": "connected",
            "memory": "ok"
        }
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check - is the service ready to receive traffic?"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return {"ready": True}

@app.post("/v1/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """Invoke the agent with a query."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = await agent.invoke(request.query, request.context)
        return AgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/stream")
async def stream_agent(request: AgentRequest):
    """Stream the agent's response using Server-Sent Events."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def generate():
        try:
            async for chunk in agent.stream(request.query):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# === Run locally ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Production-Enhanced API

Now let's add production features:

```python
# code/01_production_api.py

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager
import asyncio
import time
import logging
import os
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# === Configuration ===

class Settings:
    """Application settings from environment."""
    APP_NAME: str = os.getenv("APP_NAME", "ai-agent")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "60"))

settings = Settings()

# === Logging ===

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# === Metrics ===

REQUEST_COUNT = Counter(
    'agent_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'agent_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

TOKEN_USAGE = Counter(
    'agent_tokens_total',
    'Total tokens used',
    ['model']
)

# === Models ===

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    context: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = Field(default=None, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)

class AgentResponse(BaseModel):
    id: str
    response: str
    tokens_used: int
    model: str
    latency_ms: float
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str

# === Agent with Production Features ===

class ProductionAgent:
    """Production-ready agent with caching and monitoring."""

    def __init__(self):
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=settings.MAX_TOKENS,
            request_timeout=settings.TIMEOUT_SECONDS
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm
        self._cache: Dict[str, Any] = {}

    async def invoke(self, query: str, **kwargs) -> dict:
        """Invoke with caching and metrics."""
        import hashlib
        import uuid

        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._cache:
            logger.info(f"Cache hit for query: {cache_key[:8]}")
            return self._cache[cache_key]

        start_time = time.time()

        try:
            response = await asyncio.wait_for(
                self.chain.ainvoke({"query": query}),
                timeout=settings.TIMEOUT_SECONDS
            )

            latency_ms = (time.time() - start_time) * 1000
            tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)

            # Update metrics
            TOKEN_USAGE.labels(model="gpt-4o-mini").inc(tokens)

            result = {
                "id": str(uuid.uuid4()),
                "response": response.content,
                "tokens_used": tokens,
                "model": "gpt-4o-mini",
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Cache result
            self._cache[cache_key] = result
            return result

        except asyncio.TimeoutError:
            logger.error(f"Request timeout after {settings.TIMEOUT_SECONDS}s")
            raise HTTPException(status_code=504, detail="Request timeout")

    async def stream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream response chunks."""
        async for chunk in self.chain.astream({"query": query}):
            if chunk.content:
                yield chunk.content

# === Lifespan Management ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    app.state.agent = ProductionAgent()
    app.state.start_time = time.time()
    logger.info("Agent initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down gracefully...")
    # Cleanup resources here
    logger.info("Shutdown complete")

# === Application ===

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Middleware ===

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(latency)

    return response

# === Endpoints ===

@app.get("/health")
async def health():
    """Liveness probe - is the process running?"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "uptime_seconds": round(time.time() - app.state.start_time, 2)
    }

@app.get("/ready")
async def ready():
    """Readiness probe - is the service ready to accept traffic?"""
    try:
        # Verify agent is initialized
        if not hasattr(app.state, 'agent') or app.state.agent is None:
            raise HTTPException(status_code=503, detail="Agent not ready")

        # Could add more checks here (database, external services, etc.)
        return {"ready": True}

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/v1/invoke", response_model=AgentResponse)
async def invoke(request: AgentRequest, background_tasks: BackgroundTasks):
    """Invoke the agent."""
    try:
        result = await app.state.agent.invoke(
            request.query,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return AgentResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during invocation")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/stream")
async def stream(request: AgentRequest):
    """Stream the agent's response."""
    async def generate():
        try:
            async for chunk in app.state.agent.stream(request.query):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Error during streaming")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# === Error Handlers ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred",
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
    )
```

---

## Creating Production Dockerfiles

### Basic Dockerfile

```dockerfile
# code/Dockerfile.basic

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Multi-Stage Dockerfile

```dockerfile
# code/Dockerfile.production

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Production
# ============================================
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN groupadd -r agent && useradd -r -g agent agent

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=agent:agent . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_VERSION=1.0.0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER agent

# Run with production settings
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools"]
```

### Optimized Dockerfile with Layer Caching

```dockerfile
# code/Dockerfile.optimized

# ============================================
# Stage 1: Dependencies
# ============================================
FROM python:3.11-slim as dependencies

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Production Image
# ============================================
FROM python:3.11-slim as production

# Labels for container metadata
LABEL org.opencontainers.image.title="AI Agent" \
      org.opencontainers.image.description="Production AI Agent Service" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="YourCompany"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r agent && useradd -r -g agent agent

# Copy virtual environment
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code (changes frequently, so last)
COPY --chown=agent:agent ./app ./app
COPY --chown=agent:agent ./main.py .

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root
USER agent

# Startup command
ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Emergence Thinking: Production Behavior from Simple Components

Let's see how simple components combine to create production-grade behavior:

```
SIMPLE COMPONENTS                    EMERGENT PRODUCTION BEHAVIOR
─────────────────────────────────────────────────────────────────────

Health Check Endpoint                Self-Healing System
├── Return 200 if OK                 ├── Orchestrator detects failure
├── Return 503 if not                ├── Automatically restarts
└── Check dependencies               └── Routes traffic away

Graceful Shutdown Handler            Zero-Downtime Deployments
├── Stop accepting new requests      ├── New version starts
├── Finish in-flight requests        ├── Traffic shifts gradually
└── Close connections cleanly        └── Old version terminates

Request Timeout                      Predictable Latency
├── Set max execution time           ├── No hung requests
├── Return error on timeout          ├── Resources freed
└── Log for debugging                └── User gets clear feedback

Structured Logging                   Debuggable System
├── JSON format                      ├── Searchable logs
├── Include request ID               ├── Traceable requests
└── Log level configuration          └── Appropriate verbosity

                Combined, these produce:

                ┌────────────────────────────────────────┐
                │                                        │
                │   PRODUCTION-GRADE SERVICE             │
                │                                        │
                │   - Survives failures                  │
                │   - Updates without downtime           │
                │   - Responds predictably               │
                │   - Debuggable in production           │
                │                                        │
                └────────────────────────────────────────┘
```

### Implementing Graceful Shutdown

```python
# code/01_graceful_shutdown.py

import asyncio
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

logger = logging.getLogger(__name__)

class GracefulShutdown:
    """Handle graceful shutdown of the application."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.active_requests = 0
        self._lock = asyncio.Lock()

    async def start_request(self):
        """Track start of a request."""
        async with self._lock:
            self.active_requests += 1

    async def end_request(self):
        """Track end of a request."""
        async with self._lock:
            self.active_requests -= 1

    async def wait_for_shutdown(self, timeout: float = 30.0):
        """Wait for all requests to complete or timeout."""
        logger.info(f"Waiting for {self.active_requests} active requests...")

        start_time = asyncio.get_event_loop().time()
        while self.active_requests > 0:
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(f"Shutdown timeout, {self.active_requests} requests still active")
                break
            await asyncio.sleep(0.1)

        logger.info("All requests completed, shutting down")

    def trigger_shutdown(self):
        """Trigger the shutdown event."""
        self.shutdown_event.set()

shutdown_handler = GracefulShutdown()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with graceful shutdown."""

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def handle_signal(sig):
        logger.info(f"Received signal {sig}, initiating shutdown")
        shutdown_handler.trigger_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    # Startup
    logger.info("Application starting up")
    yield

    # Shutdown
    logger.info("Application shutting down")
    await shutdown_handler.wait_for_shutdown()
    logger.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def track_requests(request, call_next):
    """Track active requests for graceful shutdown."""
    await shutdown_handler.start_request()
    try:
        response = await call_next(request)
        return response
    finally:
        await shutdown_handler.end_request()

@app.get("/health")
async def health():
    """Health check that respects shutdown state."""
    if shutdown_handler.shutdown_event.is_set():
        return {"status": "shutting_down"}, 503
    return {"status": "healthy"}
```

---

## Complete Example: Containerized Research Agent

Let's build a complete containerized agent:

### Project Structure

```
research-agent/
├── app/
│   ├── __init__.py
│   ├── agent.py          # Agent implementation
│   ├── api.py            # FastAPI routes
│   ├── config.py         # Configuration
│   └── models.py         # Pydantic models
├── main.py               # Entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

### Agent Implementation

```python
# app/agent.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_openai_tools_agent
from typing import AsyncGenerator, Optional
import asyncio

class ResearchAgent:
    """A research agent that can search the web and analyze information."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.tools = [DuckDuckGoSearchRun()]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant. You help users find and analyze information.

            You have access to a web search tool. Use it to find current information.

            Always cite your sources and be clear about what you found vs. what you inferred."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            return_intermediate_steps=True
        )

    async def invoke(self, query: str) -> dict:
        """Invoke the research agent."""
        result = await self.executor.ainvoke({"input": query})
        return {
            "response": result["output"],
            "steps": len(result.get("intermediate_steps", [])),
            "sources": self._extract_sources(result)
        }

    async def stream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream the agent's response."""
        async for event in self.executor.astream_events(
            {"input": query},
            version="v1"
        ):
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content

    def _extract_sources(self, result: dict) -> list:
        """Extract sources from intermediate steps."""
        sources = []
        for step in result.get("intermediate_steps", []):
            if hasattr(step[0], 'tool') and step[0].tool == "duckduckgo_search":
                sources.append(step[1][:200] + "...")
        return sources
```

### API Routes

```python
# app/api.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from .models import AgentRequest, AgentResponse, HealthResponse
from .agent import ResearchAgent
from .config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Agent instance (initialized at startup)
_agent: ResearchAgent = None

def get_agent() -> ResearchAgent:
    """Dependency to get the agent instance."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _agent

def init_agent():
    """Initialize the agent."""
    global _agent
    _agent = ResearchAgent(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE
    )
    logger.info("Research agent initialized")

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model=settings.MODEL_NAME
    )

@router.get("/ready")
async def ready(agent: ResearchAgent = Depends(get_agent)):
    """Readiness check."""
    return {"ready": True, "agent": "initialized"}

@router.post("/v1/research", response_model=AgentResponse)
async def research(
    request: AgentRequest,
    agent: ResearchAgent = Depends(get_agent)
):
    """Perform research on a topic."""
    try:
        result = await agent.invoke(request.query)
        return AgentResponse(
            query=request.query,
            response=result["response"],
            steps=result["steps"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.exception("Research failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/research/stream")
async def research_stream(
    request: AgentRequest,
    agent: ResearchAgent = Depends(get_agent)
):
    """Stream research results."""
    async def generate():
        try:
            async for chunk in agent.stream(request.query):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Stream failed")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### Main Application

```python
# main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from app.api import router, init_agent
from app.config import settings

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    init_agent()
    yield
    logger.info("Shutting down")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml

version: '3.8'

services:
  agent:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=DEBUG
      - MODEL_NAME=gpt-4o-mini
    volumes:
      - ./app:/app/app  # Hot reload during development
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

---

## Testing Containers Locally

### Build and Run

```bash
# Build the image
docker build -t research-agent:latest .

# Run the container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  research-agent:latest

# Or use docker-compose
docker-compose up --build
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Research query
curl -X POST http://localhost:8000/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in quantum computing?"}'

# Streaming query
curl -X POST http://localhost:8000/v1/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain machine learning in simple terms"}'
```

### Automated Testing Script

```python
# tests/test_container.py

import requests
import time
import subprocess
import sys

def wait_for_healthy(url: str, timeout: int = 60):
    """Wait for the service to be healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False

def test_container():
    """Test the containerized agent."""
    base_url = "http://localhost:8000"

    print("Waiting for service to be healthy...")
    if not wait_for_healthy(base_url):
        print("Service failed to become healthy")
        sys.exit(1)

    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("  Health check passed")

    print("Testing readiness endpoint...")
    response = requests.get(f"{base_url}/ready")
    assert response.status_code == 200
    print("  Readiness check passed")

    print("Testing research endpoint...")
    response = requests.post(
        f"{base_url}/v1/research",
        json={"query": "What is Python?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    print(f"  Research returned: {data['response'][:100]}...")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_container()
```

---

## Best Practices Summary

### Docker Best Practices

```
┌─────────────────────────────────────────────────────────────────┐
│                   DOCKER BEST PRACTICES                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. USE MULTI-STAGE BUILDS                                       │
│     ├── Separate build and runtime                               │
│     ├── Smaller final image                                      │
│     └── Faster deployments                                       │
│                                                                  │
│  2. ORDER LAYERS BY CHANGE FREQUENCY                             │
│     ├── System packages (rarely change)                          │
│     ├── Dependencies (occasionally change)                       │
│     └── Application code (frequently changes)                    │
│                                                                  │
│  3. RUN AS NON-ROOT                                              │
│     ├── Create dedicated user                                    │
│     ├── Minimize attack surface                                  │
│     └── Required by many orchestrators                           │
│                                                                  │
│  4. USE .dockerignore                                            │
│     ├── Exclude .git, __pycache__, .env                         │
│     ├── Faster builds                                            │
│     └── Smaller context                                          │
│                                                                  │
│  5. PIN DEPENDENCY VERSIONS                                      │
│     ├── Reproducible builds                                      │
│     ├── No surprise updates                                      │
│     └── Use pip freeze > requirements.txt                       │
│                                                                  │
│  6. INCLUDE HEALTH CHECKS                                        │
│     ├── HEALTHCHECK instruction in Dockerfile                   │
│     ├── Enables orchestrator monitoring                          │
│     └── Automatic recovery                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### .dockerignore Example

```
# .dockerignore

# Git
.git
.gitignore

# Python
__pycache__
*.py[cod]
*.egg-info
.eggs
*.egg
.pytest_cache
.mypy_cache

# Virtual environments
venv
.venv
env

# IDE
.idea
.vscode
*.swp
*.swo

# Build
dist
build

# Secrets (never include)
.env
.env.*
*.pem
*.key

# Tests (usually not needed in production)
tests
*_test.py
test_*.py

# Documentation
docs
*.md
!README.md

# Docker
Dockerfile*
docker-compose*
```

---

## Key Takeaways

### 1. Containers Solve the "Works on My Machine" Problem
By packaging code, dependencies, and runtime together, containers ensure consistent behavior everywhere.

### 2. FastAPI is Ideal for AI Agent APIs
Async support, automatic documentation, and Pydantic validation make it perfect for AI applications.

### 3. Multi-Stage Builds Reduce Image Size
Separate build and runtime stages result in smaller, faster-deploying images.

### 4. Health Checks Enable Self-Healing
Simple endpoints that return status enable orchestrators to automatically recover from failures.

### 5. Graceful Shutdown Prevents Data Loss
Handling SIGTERM properly ensures in-flight requests complete before shutdown.

### 6. Layer Ordering Affects Build Speed
Put frequently changing content (your code) last to maximize cache hits.

---

## What's Next?

In **Module 7.2: Cloud Deployment**, we'll take our containerized agent and deploy it to:
- AWS (App Runner, ECS, Lambda)
- GCP (Cloud Run, GKE)
- Azure (Container Apps, AKS)

We'll also set up CI/CD pipelines for automated deployments.

Your agent is containerized—now let's put it in the cloud!

[Continue to Module 7.2 →](02_cloud_deployment.md)
