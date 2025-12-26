"""
Module 4.1: API Tools for Agents
================================
Demonstrates building LangChain tools that wrap APIs:
- Weather tool
- News search tool
- Cryptocurrency tool
"""

import requests
from langchain_core.tools import tool
from typing import Optional
import time
import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Weather Tool
# =============================================================================

class WeatherService:
    """Weather service using wttr.in (no API key required)."""

    def __init__(self):
        self.base_url = "https://wttr.in"
        self.last_request = 0
        self.rate_limit = 0.5  # seconds between requests

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def get_current(self, city: str) -> dict:
        """Get current weather for a city."""
        self._rate_limit()

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
        self._rate_limit()

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


# Create service instance
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


# =============================================================================
# Cryptocurrency Tool
# =============================================================================

class CryptoService:
    """Cryptocurrency data using CoinGecko API (no key required)."""

    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.last_request = 0
        self.rate_limit = 1.0  # CoinGecko has strict rate limits

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def get_price(self, coin_id: str) -> dict:
        """Get current price for a cryptocurrency."""
        self._rate_limit()

        try:
            response = requests.get(
                f"{self.base_url}/simple/price",
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
                        "market_cap": coin_data.get("usd_market_cap", 0)
                    }
                else:
                    return {"success": False, "error": f"Coin not found: {coin_id}"}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_trending(self) -> dict:
        """Get trending cryptocurrencies."""
        self._rate_limit()

        try:
            response = requests.get(
                f"{self.base_url}/search/trending",
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
                        "rank": coin.get("market_cap_rank", "N/A")
                    })
                return {"success": True, "trending": coins}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


crypto_service = CryptoService()


@tool
def get_crypto_price(coin: str) -> str:
    """Get current price of a cryptocurrency.

    Args:
        coin: Name of the cryptocurrency (e.g., 'bitcoin', 'ethereum', 'solana')

    Returns:
        Current price and 24h change for the cryptocurrency.
    """
    coin_id = coin.lower().replace(" ", "-")
    result = crypto_service.get_price(coin_id)

    if result["success"]:
        change_symbol = "+" if result["change_24h"] >= 0 else ""
        return (
            f"{result['coin'].title()}:\n"
            f"- Price: ${result['price_usd']:,.2f} (EUR {result['price_eur']:,.2f})\n"
            f"- 24h Change: {change_symbol}{result['change_24h']}%\n"
            f"- Market Cap: ${result['market_cap']:,.0f}"
        )
    else:
        return f"Could not get price: {result['error']}"


@tool
def get_trending_crypto() -> str:
    """Get currently trending cryptocurrencies.

    Returns:
        List of trending cryptocurrencies with their ranks.
    """
    result = crypto_service.get_trending()

    if result["success"]:
        lines = ["Trending Cryptocurrencies:"]
        for coin in result["trending"]:
            lines.append(f"- {coin['name']} ({coin['symbol']}) - Rank #{coin['rank']}")
        return "\n".join(lines)
    else:
        return f"Could not get trending: {result['error']}"


# =============================================================================
# Demo
# =============================================================================

def demo_tools():
    """Test all the API tools."""
    print("=" * 60)
    print("API Tools Demo")
    print("=" * 60)

    # Weather tools
    print("\n--- Weather Tools ---")
    print(get_current_weather.invoke({"city": "London"}))
    print()
    print(get_weather_forecast.invoke({"city": "Tokyo", "days": 3}))

    # Crypto tools
    print("\n--- Crypto Tools ---")
    print(get_crypto_price.invoke({"coin": "bitcoin"}))
    print()
    print(get_trending_crypto.invoke({}))


if __name__ == "__main__":
    demo_tools()
