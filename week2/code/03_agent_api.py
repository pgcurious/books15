"""
Module 2.2: Agent API with FastAPI
==================================
Deploy your agent as a production-ready REST API.

Run with: uvicorn 03_agent_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import time
import math
import json
from datetime import datetime

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
        allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                   "pi": math.pi, "e": math.e, "log": math.log}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_datetime() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


# =============================================================================
# AGENT SERVICE
# =============================================================================

class AgentService:
    """Singleton agent service for the API."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the agent."""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.tools = [search_web, calculate, get_datetime]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.sessions = {}

        self.system_prompt = """You are a helpful research assistant.
You have access to tools for search, calculation, and getting the current time.
Use them when they would help answer questions. Be concise and helpful."""

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

    def chat(self, message: str, session_id: str = "default") -> dict:
        """Process a chat message and return response with metadata."""
        start_time = time.time()
        session = self._get_session(session_id)
        tools_used = []

        messages = [
            {"role": "system", "content": self.system_prompt},
            *[{"role": m.type, "content": m.content} for m in session.messages],
            {"role": "user", "content": message}
        ]

        response = self.llm_with_tools.invoke(messages)

        if response.tool_calls:
            tools_used = [call["name"] for call in response.tool_calls]
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

        return {
            "response": response_text,
            "session_id": session_id,
            "tools_used": tools_used,
            "processing_time": time.time() - start_time
        }

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def list_sessions(self):
        return list(self.sessions.keys())


# =============================================================================
# API MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="The message to send to the agent")
    session_id: Optional[str] = Field("default", description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="The agent's response")
    session_id: str = Field(..., description="The session ID used")
    tools_used: List[str] = Field(default_factory=list, description="Tools used to generate response")
    processing_time: float = Field(..., description="Processing time in seconds")


class SessionInfo(BaseModel):
    """Session information model."""
    session_id: str
    message_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Research Assistant API",
    description="An AI-powered research assistant with search, calculation, and conversation capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Agent service instance
agent = AgentService()


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_model=dict)
async def root():
    """API information and available endpoints."""
    return {
        "name": "Research Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat - Send a message",
            "stream": "POST /stream - Stream a response",
            "sessions": "GET /sessions - List active sessions",
            "clear": "POST /sessions/{session_id}/clear - Clear a session",
            "health": "GET /health - Health check"
        },
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the agent and receive a response.

    The agent will:
    - Use tools (search, calculate) when helpful
    - Maintain conversation context within the session
    - Return a helpful response

    **Example:**
    ```json
    {
        "message": "What is 15% of 200?",
        "session_id": "user-123"
    }
    ```
    """
    try:
        result = agent.chat(request.message, request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream a response from the agent.

    Returns Server-Sent Events (SSE) that can be consumed by EventSource.

    **Note:** Streaming does not currently support tool use.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    async def generate():
        async for chunk in chain.astream({"question": request.message}):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.get("/sessions", response_model=List[str])
async def list_sessions():
    """List all active session IDs."""
    return agent.list_sessions()


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific session."""
    if session_id not in agent.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = agent.sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        message_count=len(session.messages)
    )


@app.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear the conversation history for a session."""
    agent.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session entirely."""
    if session_id in agent.sessions:
        del agent.sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Research Assistant API...")
    print("Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
