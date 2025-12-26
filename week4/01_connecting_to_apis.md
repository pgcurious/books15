# Module 4.1: Connecting Agents to the Real World

## What You'll Learn
- Understand APIs from first principles as system communication protocols
- Build robust API tools that handle real-world complexity
- Implement authentication, error handling, and rate limiting
- Create practical API integrations for weather, news, and financial data
- Design resilient agents that gracefully handle API failures

---

## First Principles: Why Do Agents Need APIs?

### The Limitation of Static Knowledge

Let's think carefully about what an LLM actually knows:

```
LLM Training Data:
├── Cut-off date: January 2024 (or whenever trained)
├── No access to: Real-time events, current prices, live data
├── Cannot: Check actual systems, verify current facts
└── Tendency: Hallucinate when knowledge is uncertain
```

**First Principle #1:** LLMs are frozen snapshots of language patterns, not live data sources.

Ask an LLM: "What's the weather in Tokyo right now?"
Without APIs: It will guess, fabricate, or refuse.
With APIs: It can check and give you the actual temperature.

**First Principle #2:** Real utility requires real-time data.

Most valuable tasks involve current state:
- "Should I bring an umbrella?" → Needs weather API
- "Is it a good time to buy this stock?" → Needs finance API
- "What happened in the news today?" → Needs news API
- "Is the server healthy?" → Needs monitoring API

**First Principle #3:** APIs are the nervous system connecting agents to reality.

```
┌────────────────────────────────────────────────────────┐
│                    THE AGENT'S SENSES                   │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Without APIs:                With APIs:                │
│  ┌─────────┐                  ┌─────────┐              │
│  │  Agent  │                  │  Agent  │◄───┐        │
│  │ (Blind) │                  │ (Aware) │    │        │
│  └─────────┘                  └─────────┘    │        │
│       │                            │          │        │
│       ▼                            ▼          │        │
│   "I think..."              "I checked..."    │        │
│   (Guessing)                (Knowing)         │        │
│                                               │        │
│                              ┌────────────────┘        │
│                              │  Real World             │
│                              │  ├── Weather services   │
│                              │  ├── News feeds         │
│                              │  ├── Databases          │
│                              │  └── Any HTTP service   │
│                              └─────────────────────────┤
└────────────────────────────────────────────────────────┘
```

---

## Part 1: API Fundamentals

### What Is an API, Really?

Strip away the acronym (Application Programming Interface) and see the essence:

**An API is a contract for conversation between systems.**

It specifies:
1. **How to ask** (HTTP method, URL structure)
2. **What to provide** (Parameters, authentication)
3. **What you'll receive** (Response format)
4. **What can go wrong** (Error codes)

### The Anatomy of an HTTP Request

```python
# Every API call has these components:

# 1. METHOD: What action are you taking?
#    GET    = "Give me data"
#    POST   = "Here's data, process it"
#    PUT    = "Update this data"
#    DELETE = "Remove this data"

# 2. URL: Where are you sending the request?
#    https://api.weather.com/v1/current?city=Tokyo

# 3. HEADERS: Metadata about the request
#    Authorization: Bearer abc123
#    Content-Type: application/json

# 4. BODY: Data you're sending (for POST/PUT)
#    {"name": "New Item", "value": 42}
```

### The Anatomy of an HTTP Response

```python
# Every API response tells you:

# 1. STATUS CODE: What happened?
#    200 = Success
#    201 = Created
#    400 = Bad request (your fault)
#    401 = Unauthorized (wrong credentials)
#    403 = Forbidden (not allowed)
#    404 = Not found
#    429 = Too many requests (rate limited)
#    500 = Server error (their fault)

# 2. HEADERS: Metadata about the response
#    Content-Type: application/json
#    X-RateLimit-Remaining: 95

# 3. BODY: The actual data
#    {"temperature": 72, "conditions": "sunny"}
```

---

## Part 2: Building Your First API Tool

### The Simple Approach

Let's start with the most basic API tool:

```python
# See code/01_api_basics.py for executable version

import requests
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city to get weather for
    """
    # Simple API call
    response = requests.get(
        f"https://wttr.in/{city}?format=j1"
    )

    if response.status_code == 200:
        data = response.json()
        current = data["current_condition"][0]
        return f"Weather in {city}: {current['temp_C']}°C, {current['weatherDesc'][0]['value']}"
    else:
        return f"Could not fetch weather for {city}"
```

**What's wrong with this approach?**

Nothing—for a demo. But real APIs are messier:
- Network requests fail
- APIs go down
- Rate limits get hit
- Authentication expires
- Response formats change

---

## Part 3: Building Robust API Tools

### The Production-Ready Approach

```python
# See code/01_api_basics.py for executable version

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from typing import Optional
from langchain_core.tools import tool

class APIClient:
    """A robust API client with retry logic, timeouts, and error handling."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 10,
        rate_limit_delay: float = 1.0
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,  # Wait 0.5, 1, 2 seconds between retries
            status_forcelist=[500, 502, 503, 504]  # Retry on server errors
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _respect_rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _get_headers(self) -> dict:
        """Build request headers including authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request with full error handling."""
        self._respect_rate_limit()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )

            # Handle different status codes
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            elif response.status_code == 401:
                return {"success": False, "error": "Authentication failed. Check API key."}
            elif response.status_code == 403:
                return {"success": False, "error": "Access forbidden. Check permissions."}
            elif response.status_code == 404:
                return {"success": False, "error": f"Resource not found: {endpoint}"}
            elif response.status_code == 429:
                return {"success": False, "error": "Rate limit exceeded. Try again later."}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}

        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out. Service may be slow."}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection failed. Check network."}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
```

### Why This Matters

| Simple Approach | Robust Approach |
|-----------------|-----------------|
| Crashes on timeout | Retries with backoff |
| Fails on rate limit | Respects rate limits |
| No error messages | Clear error explanations |
| Single request | Session reuse (faster) |
| No authentication | Flexible auth support |

---

## Part 4: Real API Examples

### Example 1: Weather API Tool

```python
# See code/02_weather_tool.py for executable version

from langchain_core.tools import tool
import requests

class WeatherService:
    """Weather service using wttr.in (no API key required)."""

    def __init__(self):
        self.base_url = "https://wttr.in"

    def get_current(self, city: str) -> dict:
        """Get current weather for a city."""
        try:
            response = requests.get(
                f"{self.base_url}/{city}",
                params={"format": "j1"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                current = data["current_condition"][0]
                return {
                    "success": True,
                    "city": city,
                    "temperature_c": int(current["temp_C"]),
                    "temperature_f": int(current["temp_F"]),
                    "condition": current["weatherDesc"][0]["value"],
                    "humidity": int(current["humidity"]),
                    "wind_mph": int(current["windspeedMiles"]),
                    "feels_like_c": int(current["FeelsLikeC"])
                }
            else:
                return {"success": False, "error": f"City not found: {city}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_forecast(self, city: str, days: int = 3) -> dict:
        """Get weather forecast for upcoming days."""
        try:
            response = requests.get(
                f"{self.base_url}/{city}",
                params={"format": "j1"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                forecast = []
                for day in data["weather"][:days]:
                    forecast.append({
                        "date": day["date"],
                        "max_temp_c": int(day["maxtempC"]),
                        "min_temp_c": int(day["mintempC"]),
                        "condition": day["hourly"][4]["weatherDesc"][0]["value"]
                    })
                return {"success": True, "city": city, "forecast": forecast}
            else:
                return {"success": False, "error": f"City not found: {city}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


# Create tools from the service
weather_service = WeatherService()

@tool
def get_current_weather(city: str) -> str:
    """Get current weather conditions for a city.

    Args:
        city: Name of the city (e.g., 'Tokyo', 'New York', 'London')

    Returns:
        Current weather information including temperature and conditions.
    """
    result = weather_service.get_current(city)

    if result["success"]:
        return (
            f"Weather in {result['city']}:\n"
            f"- Temperature: {result['temperature_c']}°C ({result['temperature_f']}°F)\n"
            f"- Feels like: {result['feels_like_c']}°C\n"
            f"- Conditions: {result['condition']}\n"
            f"- Humidity: {result['humidity']}%\n"
            f"- Wind: {result['wind_mph']} mph"
        )
    else:
        return f"Could not get weather: {result['error']}"

@tool
def get_weather_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for upcoming days.

    Args:
        city: Name of the city
        days: Number of days to forecast (1-3)

    Returns:
        Weather forecast for the specified number of days.
    """
    result = weather_service.get_forecast(city, min(days, 3))

    if result["success"]:
        lines = [f"Forecast for {result['city']}:"]
        for day in result["forecast"]:
            lines.append(
                f"- {day['date']}: {day['min_temp_c']}°C to {day['max_temp_c']}°C, "
                f"{day['condition']}"
            )
        return "\n".join(lines)
    else:
        return f"Could not get forecast: {result['error']}"
```

### Example 2: News API Tool

```python
# See code/03_news_tool.py for executable version

from langchain_core.tools import tool
from duckduckgo_search import DDGS
from datetime import datetime

class NewsService:
    """News service using DuckDuckGo (no API key required)."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.ddgs = DDGS()

    def search_news(self, query: str, days: int = 7) -> dict:
        """Search for recent news articles."""
        try:
            # Map days to DuckDuckGo time filter
            if days <= 1:
                timelimit = "d"  # day
            elif days <= 7:
                timelimit = "w"  # week
            else:
                timelimit = "m"  # month

            results = list(self.ddgs.news(
                query,
                max_results=self.max_results,
                timelimit=timelimit
            ))

            articles = []
            for r in results:
                articles.append({
                    "title": r.get("title", ""),
                    "source": r.get("source", ""),
                    "date": r.get("date", ""),
                    "body": r.get("body", "")[:200] + "...",  # Truncate
                    "url": r.get("url", "")
                })

            return {"success": True, "query": query, "articles": articles}

        except Exception as e:
            return {"success": False, "error": str(e)}


news_service = NewsService()

@tool
def search_recent_news(query: str, days: int = 7) -> str:
    """Search for recent news articles on a topic.

    Args:
        query: Topic or keywords to search for
        days: How recent the news should be (1, 7, or 30)

    Returns:
        Summary of recent news articles on the topic.
    """
    result = news_service.search_news(query, days)

    if result["success"]:
        if not result["articles"]:
            return f"No recent news found for: {query}"

        lines = [f"Recent news about '{query}':"]
        for i, article in enumerate(result["articles"], 1):
            lines.append(f"\n{i}. {article['title']}")
            lines.append(f"   Source: {article['source']}")
            lines.append(f"   {article['body']}")

        return "\n".join(lines)
    else:
        return f"Could not search news: {result['error']}"
```

### Example 3: Financial Data Tool

```python
# See code/04_finance_tool.py for executable version

from langchain_core.tools import tool
import requests

class FinanceService:
    """Financial data service using free APIs."""

    def __init__(self):
        # Using a free crypto API for demo
        self.crypto_url = "https://api.coingecko.com/api/v3"

    def get_crypto_price(self, coin_id: str) -> dict:
        """Get current cryptocurrency price."""
        try:
            response = requests.get(
                f"{self.crypto_url}/simple/price",
                params={
                    "ids": coin_id,
                    "vs_currencies": "usd,eur",
                    "include_24hr_change": "true",
                    "include_market_cap": "true"
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if coin_id in data:
                    coin_data = data[coin_id]
                    return {
                        "success": True,
                        "coin": coin_id,
                        "price_usd": coin_data.get("usd", 0),
                        "price_eur": coin_data.get("eur", 0),
                        "change_24h": round(coin_data.get("usd_24h_change", 0), 2),
                        "market_cap_usd": coin_data.get("usd_market_cap", 0)
                    }
                else:
                    return {"success": False, "error": f"Coin not found: {coin_id}"}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_trending_coins(self) -> dict:
        """Get trending cryptocurrencies."""
        try:
            response = requests.get(
                f"{self.crypto_url}/search/trending",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                coins = []
                for item in data.get("coins", [])[:5]:
                    coin = item["item"]
                    coins.append({
                        "name": coin["name"],
                        "symbol": coin["symbol"],
                        "market_cap_rank": coin.get("market_cap_rank", "N/A")
                    })
                return {"success": True, "trending": coins}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


finance_service = FinanceService()

@tool
def get_crypto_price(coin: str) -> str:
    """Get current price of a cryptocurrency.

    Args:
        coin: Name of the cryptocurrency (e.g., 'bitcoin', 'ethereum', 'solana')

    Returns:
        Current price and 24h change for the cryptocurrency.
    """
    # Normalize coin name
    coin_id = coin.lower().replace(" ", "-")

    result = finance_service.get_crypto_price(coin_id)

    if result["success"]:
        change_symbol = "+" if result["change_24h"] >= 0 else ""
        return (
            f"💰 {result['coin'].title()}:\n"
            f"- Price: ${result['price_usd']:,.2f} (€{result['price_eur']:,.2f})\n"
            f"- 24h Change: {change_symbol}{result['change_24h']}%\n"
            f"- Market Cap: ${result['market_cap_usd']:,.0f}"
        )
    else:
        return f"Could not get price: {result['error']}"

@tool
def get_trending_crypto() -> str:
    """Get currently trending cryptocurrencies.

    Returns:
        List of trending cryptocurrencies with their ranks.
    """
    result = finance_service.get_trending_coins()

    if result["success"]:
        lines = ["🔥 Trending Cryptocurrencies:"]
        for coin in result["trending"]:
            lines.append(
                f"- {coin['name']} ({coin['symbol']}) - "
                f"Rank #{coin['market_cap_rank']}"
            )
        return "\n".join(lines)
    else:
        return f"Could not get trending: {result['error']}"
```

---

## Part 5: Creating an API-Powered Agent

### Putting It All Together

```python
# See code/05_api_agent.py for executable version

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# Import our tools (from previous examples)
from weather_tool import get_current_weather, get_weather_forecast
from news_tool import search_recent_news
from finance_tool import get_crypto_price, get_trending_crypto

# Create the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Combine all tools
tools = [
    get_current_weather,
    get_weather_forecast,
    search_recent_news,
    get_crypto_price,
    get_trending_crypto
]

# System prompt for the agent
system_prompt = """You are a helpful assistant with access to real-time data.

You have tools to check:
- Current weather and forecasts for any city
- Recent news on any topic
- Cryptocurrency prices and trends

When answering questions:
1. Use the appropriate tool to get current, accurate data
2. Synthesize information clearly and concisely
3. Cite your sources (the APIs you used)
4. If a tool fails, explain what happened and try an alternative

Never guess or make up data - always use your tools to get real information."""

# Create the agent
agent = create_react_agent(
    llm,
    tools,
    state_modifier=system_prompt
)

def ask_agent(question: str) -> str:
    """Ask the agent a question and get a response."""
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # Get the final response
    return result["messages"][-1].content


# Example usage
if __name__ == "__main__":
    questions = [
        "What's the weather like in Tokyo and should I bring an umbrella?",
        "What's happening with Bitcoin today?",
        "Give me a summary of recent AI news and the weather in San Francisco"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        print(ask_agent(question))
```

---

## Part 6: Handling API Failures Gracefully

### The Resilience Pattern

Real-world APIs fail. Your agent should handle this gracefully:

```python
# See code/05_api_agent.py for executable version

from typing import List, Callable, Any
import time

class ResilientToolWrapper:
    """Wraps tools with fallback and retry logic."""

    def __init__(self, primary_tool: Callable, fallback_tools: List[Callable] = None):
        self.primary = primary_tool
        self.fallbacks = fallback_tools or []
        self.max_retries = 2
        self.retry_delay = 1.0

    def __call__(self, *args, **kwargs) -> Any:
        """Execute with retry and fallback logic."""

        # Try primary tool
        for attempt in range(self.max_retries):
            try:
                result = self.primary(*args, **kwargs)
                if "error" not in str(result).lower():
                    return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

        # Try fallback tools
        for fallback in self.fallbacks:
            try:
                result = fallback(*args, **kwargs)
                if "error" not in str(result).lower():
                    return f"[Via fallback] {result}"
            except Exception:
                continue

        # All failed
        return "Unable to retrieve data. All sources are currently unavailable."


# Example: Weather with fallback
def get_weather_backup(city: str) -> str:
    """Backup weather source."""
    # Could use a different API here
    return f"Weather service temporarily unavailable for {city}"

resilient_weather = ResilientToolWrapper(
    primary_tool=get_current_weather,
    fallback_tools=[get_weather_backup]
)
```

---

## Analogical Thinking: APIs as Conversations

Consider how APIs mirror human professional interactions:

### The Restaurant Analogy

| Restaurant | API |
|------------|-----|
| Menu | API documentation |
| Placing an order | Making a request |
| "I'll have the steak" | GET /menu/steak |
| "Medium rare, no salt" | Query parameters |
| Membership card | API key |
| "Kitchen is busy, 20 min wait" | Rate limiting |
| Getting your meal | Response body |
| "Sorry, we're out of fish" | 404 Not Found |
| "Card declined" | 401 Unauthorized |

### The Embassy Analogy (For Authentication)

| Embassy Process | OAuth Flow |
|-----------------|------------|
| Apply for visa | Request authorization |
| Provide documents | Send credentials |
| Receive visa stamp | Get access token |
| Visa expires | Token expiration |
| Renew visa | Refresh token |
| Visa scope (tourist vs. work) | Token scopes |

---

## Emergence Thinking: The Network Effect

When you connect an agent to multiple APIs, something remarkable happens:

```
Single API:
┌─────────────────────────────────────┐
│  Agent + Weather API                │
│  = Can tell you the weather         │
│    (Limited utility)                │
└─────────────────────────────────────┘

Multiple APIs:
┌─────────────────────────────────────┐
│  Agent + Weather + News + Finance   │
│  = Can plan your day                │
│  = Can explain market movements     │
│  = Can connect events to impacts    │
│  = Can give holistic advice         │
│    (Emergent intelligence)          │
└─────────────────────────────────────┘
```

**The emergence pattern:**
1. Each API provides isolated data
2. The LLM synthesizes across sources
3. Patterns and insights emerge
4. The whole becomes greater than the parts

**Example of emergent intelligence:**

User: "Should I invest in tech stocks today?"

Agent with single API: "I don't have access to financial data."

Agent with multiple APIs:
1. Checks finance API → Tech stocks down 2%
2. Checks news API → Fed announced rate concerns
3. Checks weather API → (Not relevant, skips)
4. Synthesizes → "Tech stocks are down 2% today, likely due to Fed rate concerns announced this morning. This might create a buying opportunity, but consider waiting for market stabilization."

No single API provides this insight. It **emerges** from synthesis.

---

## Summary

### What We Learned

1. **APIs from First Principles**
   - APIs are formalized conversations between systems
   - They follow predictable patterns (request/response)
   - Understanding HTTP fundamentals unlocks any API

2. **Building Robust Tools**
   - Simple tools break in production
   - Retries, timeouts, and error handling are essential
   - Rate limiting protects both you and the service

3. **Real-World Integration**
   - Weather, news, and finance APIs demonstrate practical value
   - Free APIs exist for learning (no excuses!)
   - Consistent patterns make adding new APIs easy

4. **Emergent Intelligence**
   - Multiple data sources enable synthesis
   - The agent becomes more than the sum of its tools
   - Real utility comes from combination, not isolation

### Key Patterns

```
Robust API Tool Pattern:
├── Retry with exponential backoff
├── Timeout configuration
├── Error code handling
├── Rate limit respect
├── Clear error messages
└── Fallback options

Agent Tool Integration:
├── Clear tool descriptions
├── Typed parameters
├── Consistent return format
├── Error gracefully
└── Cite sources
```

---

## Practice Exercises

### Exercise 1: Build a GitHub API Tool
Create a tool that:
- Searches GitHub repositories
- Gets repository statistics
- Lists recent commits
- Handles rate limiting (60 requests/hour for unauthenticated)

### Exercise 2: Add a Translation API
Integrate a translation service:
- Detect source language
- Translate to target language
- Handle unsupported languages gracefully

### Exercise 3: Create a Multi-Source Agent
Build an agent that:
- Combines at least 3 different APIs
- Answers complex questions requiring multiple sources
- Cites which APIs provided which information

---

## Next Steps

In [Module 4.2](02_rag_fundamentals.md), we'll explore RAG (Retrieval-Augmented Generation):
- How to give agents access to your own documents
- Building semantic search over custom knowledge bases
- Grounding responses in authoritative sources

---

*"An API doesn't just connect systems—it extends what's possible. Every new API is a new capability, a new sense, a new way of perceiving and acting in the world."*
