# Module 2.2: Deploy Your First Working Agent

## What You'll Learn
- Deploy agents as REST APIs with FastAPI
- Use LangServe for rapid deployment
- Build an interactive playground
- Handle streaming responses
- Production deployment considerations

---

## First Principles: What Does "Deploy" Mean?

**First Principle #1:** Deployment = Making your code accessible to users/systems.

Your Jupyter notebook agent is powerful, but useless if only you can run it. Deployment exposes it to the world.

**First Principle #2:** Web APIs are the universal interface.

REST APIs let any system—web apps, mobile apps, other services—interact with your agent using standard HTTP.

```
Without Deployment:
┌──────────────┐
│  Your Code   │  ← Only you can use it
└──────────────┘

With Deployment:
┌──────────────┐     HTTP      ┌──────────────┐
│  Your Code   │◄────────────►│   Users/     │
│  (API)       │              │   Systems    │
└──────────────┘              └──────────────┘
```

---

## The Analogy: Restaurant Kitchen

Think of deployment like a restaurant:

| Restaurant | Agent Deployment |
|------------|------------------|
| Chef's home kitchen | Local Jupyter notebook |
| Restaurant kitchen | Deployed server |
| Menu | API documentation |
| Waiter taking orders | API endpoint |
| Plating the dish | Response formatting |
| Kitchen capacity | Server resources |

You want to go from cooking for yourself to serving many customers reliably.

---

## Part 1: FastAPI Fundamentals

FastAPI is a modern Python web framework, perfect for AI applications.

### Basic FastAPI Setup

```python
# code/06_basic_fastapi.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="My Agent API",
    description="A simple AI agent API",
    version="1.0.0"
)

# Request model
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

# Response model
class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agent-api"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat with the agent.

    - **message**: The user's message
    - **session_id**: Optional session ID for conversation continuity
    """
    # For now, just echo back
    return ChatResponse(
        response=f"You said: {request.message}",
        session_id=request.session_id
    )

# Run with: uvicorn 06_basic_fastapi:app --reload
```

### Running the API

```bash
# Install dependencies
pip install fastapi uvicorn

# Run the server
uvicorn 06_basic_fastapi:app --reload --port 8000

# Test it
curl http://localhost:8000/
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

---

## Part 2: Integrating Your Agent

Now let's integrate our Research Assistant from Module 2.1:

```python
# code/07_agent_api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

# =============================================================================
# TOOLS
# =============================================================================

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [Mock results - integrate real search API]"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        import math
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_date_time() -> str:
    """Get current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

# =============================================================================
# AGENT
# =============================================================================

class Agent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.tools = [search_web, calculate, get_date_time]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.sessions = {}

        self.system_prompt = """You are a helpful research assistant.
Use tools when they would help. Be concise and helpful."""

    def _get_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    def _execute_tools(self, tool_calls):
        results = []
        for call in tool_calls:
            tool_name = call["name"]
            if tool_name in self.tool_map:
                result = self.tool_map[tool_name].invoke(call["args"])
            else:
                result = f"Unknown tool: {tool_name}"
            results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return results

    def chat(self, message: str, session_id: str = "default") -> str:
        session = self._get_session(session_id)

        messages = [
            {"role": "system", "content": self.system_prompt},
            *[{"role": m.type, "content": m.content} for m in session.messages],
            {"role": "user", "content": message}
        ]

        response = self.llm_with_tools.invoke(messages)

        if response.tool_calls:
            messages.append(response)
            tool_results = self._execute_tools(response.tool_calls)
            for result in tool_results:
                messages.append(result)
            final_response = self.llm.invoke(messages)
            response_text = final_response.content
        else:
            response_text = response.content

        session.add_user_message(message)
        session.add_ai_message(response_text)

        return response_text

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].clear()

# =============================================================================
# API
# =============================================================================

app = FastAPI(
    title="Research Assistant API",
    description="An AI agent that can search, calculate, and assist with research",
    version="1.0.0"
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = Agent()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SessionRequest(BaseModel):
    session_id: str

# Endpoints
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "research-assistant"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a message to the agent.

    The agent will:
    - Use tools if helpful (search, calculate, etc.)
    - Maintain conversation context within the session
    - Return a helpful response
    """
    try:
        response = agent.chat(request.message, request.session_id)
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-session")
def clear_session(request: SessionRequest):
    """Clear conversation history for a session."""
    agent.clear_session(request.session_id)
    return {"status": "cleared", "session_id": request.session_id}

@app.get("/sessions")
def list_sessions():
    """List active sessions."""
    return {"sessions": list(agent.sessions.keys())}

# Run with: uvicorn 07_agent_api:app --reload --port 8000
```

---

## Part 3: Adding Streaming

For longer responses, streaming provides a better user experience:

```python
# code/08_streaming_agent.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio
from typing import AsyncGenerator

app = FastAPI(title="Streaming Agent API")

# Create a simple streaming chain
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer this question: {question}"
)
chain = prompt | llm | StrOutputParser()

class QuestionRequest(BaseModel):
    question: str

async def generate_stream(question: str) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    async for chunk in chain.astream({"question": question}):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/stream")
async def stream_chat(request: QuestionRequest):
    """
    Stream a response from the agent.

    Returns Server-Sent Events (SSE) that can be consumed by:
    - EventSource in JavaScript
    - Any HTTP client that handles streaming
    """
    return StreamingResponse(
        generate_stream(request.question),
        media_type="text/event-stream"
    )

@app.get("/")
def health():
    return {"status": "healthy", "streaming": True}

# Test with:
# curl -N -X POST http://localhost:8000/stream \
#   -H "Content-Type: application/json" \
#   -d '{"question": "Explain quantum computing in simple terms"}'
```

### JavaScript Client for Streaming

```javascript
// Example frontend code to consume the stream
async function streamChat(question) {
    const response = await fetch('http://localhost:8000/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') {
                    console.log('\n--- Stream complete ---');
                } else {
                    process.stdout.write(data);  // Print token by token
                }
            }
        }
    }
}
```

---

## Part 4: LangServe for Rapid Deployment

LangServe provides a simpler path to deployment with built-in features:

```python
# code/09_langserve_app.py

from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(
    title="LangServe Agent",
    description="Agent deployed with LangServe"
)

# Create your chain
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
You are a helpful research assistant. Answer the following question:

Question: {question}

Provide a clear, concise answer.
""")

chain = prompt | llm | StrOutputParser()

# Add LangServe routes - this gives you:
# - /invoke - synchronous invocation
# - /batch - batch processing
# - /stream - streaming
# - /playground - interactive UI
add_routes(
    app,
    chain,
    path="/research"
)

# You can add multiple chains
simple_chain = ChatPromptTemplate.from_template("Translate to French: {text}") | llm | StrOutputParser()
add_routes(app, simple_chain, path="/translate")

@app.get("/")
def root():
    return {
        "message": "Welcome to the LangServe Agent",
        "endpoints": {
            "research": "/research/playground",
            "translate": "/translate/playground"
        }
    }

# Run with: uvicorn 09_langserve_app:app --reload --port 8000
# Then visit: http://localhost:8000/research/playground
```

### LangServe Features

When you add routes with LangServe, you get:

| Endpoint | Purpose |
|----------|---------|
| `/invoke` | Synchronous single invocation |
| `/batch` | Process multiple inputs |
| `/stream` | Real-time streaming |
| `/stream_log` | Stream with intermediate steps |
| `/playground` | Interactive web UI |
| `/input_schema` | OpenAPI schema for inputs |
| `/output_schema` | OpenAPI schema for outputs |

---

## Part 5: Production Considerations

### Environment Configuration

```python
# code/10_production_config.py

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings from environment variables."""

    # API Keys
    openai_api_key: str

    # Model Configuration
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Usage
settings = get_settings()
print(f"Using model: {settings.model_name}")
```

### Error Handling

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Specific error handling
class AgentError(Exception):
    """Custom exception for agent errors."""
    pass

@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    return JSONResponse(
        status_code=400,
        content={"error": "Agent error", "detail": str(exc)}
    )
```

### Rate Limiting

```python
from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.post("/chat")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def chat(request: Request):
    return {"message": "Hello!"}
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

---

## Part 6: Complete Deployment Example

Here's a production-ready deployment:

```python
# code/11_production_app.py

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import logging
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# =============================================================================
# LIFESPAN (Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    logger.info("Starting up agent service...")
    # Initialize resources here
    yield
    logger.info("Shutting down agent service...")
    # Cleanup resources here

# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="Production Research Assistant",
    description="A production-ready AI agent API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MIDDLEWARE (Request Timing)
# =============================================================================

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")
    return response

# =============================================================================
# AGENT (Singleton)
# =============================================================================

class AgentService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.sessions = {}
        logger.info("Agent service initialized")

    def chat(self, message: str, session_id: str) -> str:
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({"role": "user", "content": message})

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            *self.sessions[session_id]
        ]

        response = self.llm.invoke(messages)
        response_text = response.content

        self.sessions[session_id].append({"role": "assistant", "content": response_text})

        return response_text

def get_agent() -> AgentService:
    return AgentService()

# =============================================================================
# ENDPOINTS
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: float

@app.get("/health")
def health_check():
    """Health check for load balancers."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: AgentService = Depends(get_agent)):
    """Chat with the agent."""
    try:
        response = agent.chat(request.message, request.session_id)
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Agent error")

@app.get("/")
def root():
    """API information."""
    return {
        "name": "Production Research Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
```

---

## Key Takeaways

1. **FastAPI is ideal for AI services**: Fast, async, with automatic docs
2. **LangServe accelerates deployment**: Built-in playground and streaming
3. **Streaming improves UX**: Users see progress immediately
4. **Production requires extras**: Error handling, logging, rate limiting
5. **Docker enables portability**: Deploy anywhere consistently

---

## Next Steps

In [Module 2.3: Debugging & Optimization](03_debugging_and_optimization.md), we'll learn:
- How to debug agent behavior
- Using LangSmith for observability
- Performance optimization techniques
