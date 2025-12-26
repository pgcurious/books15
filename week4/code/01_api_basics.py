"""
Module 4.1: API Basics
======================
Demonstrates fundamental API integration concepts including:
- Basic API calls with requests
- Error handling and retries
- Rate limiting
- Building robust API clients
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# DEMO 1: Basic API Call
# =============================================================================

def demo_basic_api_call():
    """Make a simple API call and handle the response."""
    print("=" * 60)
    print("DEMO 1: Basic API Call")
    print("=" * 60)

    # Call a public API (no auth required)
    response = requests.get(
        "https://api.github.com/repos/langchain-ai/langchain",
        timeout=10
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Repository: {data['full_name']}")
        print(f"Stars: {data['stargazers_count']:,}")
        print(f"Description: {data['description'][:100]}...")
    else:
        print(f"Request failed with status: {response.status_code}")


# =============================================================================
# DEMO 2: Handling Different Response Codes
# =============================================================================

def demo_response_handling():
    """Demonstrate handling various HTTP response codes."""
    print("\n" + "=" * 60)
    print("DEMO 2: Response Code Handling")
    print("=" * 60)

    test_urls = [
        ("https://httpstat.us/200", "Success"),
        ("https://httpstat.us/404", "Not Found"),
        ("https://httpstat.us/500", "Server Error"),
    ]

    for url, expected in test_urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"\n{expected}:")
            print(f"  Status Code: {response.status_code}")
            print(f"  Response: {response.text[:50]}")
        except requests.RequestException as e:
            print(f"\n{expected}: Error - {e}")


# =============================================================================
# DEMO 3: Robust API Client
# =============================================================================

class RobustAPIClient:
    """A production-ready API client with retries, timeouts, and error handling."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 10,
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            max_retries: Number of retries for failed requests
            timeout: Request timeout in seconds
            rate_limit_delay: Minimum delay between requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
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

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """
        Make a GET request with full error handling.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters

        Returns:
            Dict with success status and data or error message
        """
        self._respect_rate_limit()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            elif response.status_code == 401:
                return {"success": False, "error": "Authentication failed"}
            elif response.status_code == 403:
                return {"success": False, "error": "Access forbidden"}
            elif response.status_code == 404:
                return {"success": False, "error": f"Not found: {endpoint}"}
            elif response.status_code == 429:
                return {"success": False, "error": "Rate limit exceeded"}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection failed"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def post(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """Make a POST request with error handling."""
        self._respect_rate_limit()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.post(
                url,
                headers=self._get_headers(),
                json=data,
                timeout=self.timeout
            )

            if response.status_code in [200, 201]:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}


def demo_robust_client():
    """Demonstrate the robust API client."""
    print("\n" + "=" * 60)
    print("DEMO 3: Robust API Client")
    print("=" * 60)

    client = RobustAPIClient(
        base_url="https://api.github.com",
        max_retries=3,
        timeout=10
    )

    # Make a request
    result = client.get("/repos/langchain-ai/langchain")

    if result["success"]:
        data = result["data"]
        print(f"Successfully fetched: {data['full_name']}")
        print(f"Stars: {data['stargazers_count']:,}")
    else:
        print(f"Request failed: {result['error']}")


# =============================================================================
# DEMO 4: Weather API (No Auth Required)
# =============================================================================

def demo_weather_api():
    """Demonstrate calling a real weather API."""
    print("\n" + "=" * 60)
    print("DEMO 4: Weather API")
    print("=" * 60)

    cities = ["London", "Tokyo", "New York"]

    for city in cities:
        try:
            response = requests.get(
                f"https://wttr.in/{city}",
                params={"format": "j1"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                current = data["current_condition"][0]
                print(f"\n{city}:")
                print(f"  Temperature: {current['temp_C']}°C ({current['temp_F']}°F)")
                print(f"  Conditions: {current['weatherDesc'][0]['value']}")
                print(f"  Humidity: {current['humidity']}%")
            else:
                print(f"\n{city}: Failed to fetch weather")

        except Exception as e:
            print(f"\n{city}: Error - {e}")

        time.sleep(0.5)  # Rate limiting


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    demo_basic_api_call()
    demo_response_handling()
    demo_robust_client()
    demo_weather_api()

    print("\n" + "=" * 60)
    print("API Basics Demo Complete!")
    print("=" * 60)
