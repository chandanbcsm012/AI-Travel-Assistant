from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = "Search the web for current events or factual information."

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.run(query)

    async def _arun(self, query: str) -> str:
        return self._run(query)

class WeatherTool(BaseTool):
    name: str = "weather_tool"
    description: str = "Get current weather for a city."

    def _run(self, city: str) -> str:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            return "Error: OpenWeatherMap API key not found in environment variables. Please set OPENWEATHERMAP_API_KEY in your .env file."
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        try:
            response = requests.get(url).json()
            if response.get("cod") != 200:
                error_message = response.get('message', 'Unknown error from OpenWeatherMap API')
                return f"Error fetching weather for {city}: {error_message}"
            main = response['main']
            weather = response['weather'][0]['description']
            return f"{city}: {main['temp']}°C, {weather}"
        except requests.exceptions.RequestException as e:
            return f"Error connecting to OpenWeatherMap API: {e}"
        except KeyError:
            return "Error: Unexpected response format from OpenWeatherMap API. Check city name or API key."

    async def _arun(self, city: str) -> str:
        return self._run(city)

def verify_tools():
    print("--- Verifying DuckDuckGoSearchTool ---")
    duckduckgo_tool = DuckDuckGoSearchTool()
    test_query = "current time in London"
    print(f"Testing with query: '{test_query}'")
    result = duckduckgo_tool._run(test_query)
    print(f"Result: {result}")
    if "london" in result.lower() and "time" in result.lower():
        print("DuckDuckGoSearchTool seems to be working.")
    else:
        print("DuckDuckGoSearchTool might have issues. Check the output.")
    print("-" * 40)

    print("\n--- Verifying WeatherTool ---")
    weather_tool = WeatherTool()

    # Test with a valid city
    test_city_valid = "London"
    print(f"Testing with valid city: '{test_city_valid}'")
    result_weather_valid = weather_tool._run(test_city_valid)
    print(f"Result: {result_weather_valid}")
    if "°C" in result_weather_valid:
        print("WeatherTool seems to be working for a valid city.")
    else:
        print("WeatherTool might have issues for a valid city. Check the output and your API key.")
    print("-" * 40)

    # Test with an invalid city (optional, to see error handling)
    test_city_invalid = "NonExistentCity123"
    print(f"Testing with invalid city: '{test_city_invalid}'")
    result_weather_invalid = weather_tool._run(test_city_invalid)
    print(f"Result: {result_weather_invalid}")
    if "Error fetching weather" in result_weather_invalid or "city not found" in result_weather_invalid.lower():
        print("WeatherTool error handling for invalid city seems to be working.")
    else:
        print("WeatherTool might not be handling invalid cities correctly.")
    print("-" * 40)

    # Test without API key (optional, if you want to verify the error message)
    # Temporarily unset the API key to test this scenario
    original_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if original_api_key:
        print("\n--- Verifying WeatherTool (without API key) ---")
        del os.environ["OPENWEATHERMAP_API_KEY"]
        result_no_api_key = weather_tool._run("London")
        print(f"Result: {result_no_api_key}")
        if "Error: OpenWeatherMap API key not found" in result_no_api_key:
            print("WeatherTool's API key check seems to be working.")
        else:
            print("WeatherTool might not be correctly identifying missing API key.")
        os.environ["OPENWEATHERMAP_API_KEY"] = original_api_key # Restore the API key
        print("-" * 40)


if __name__ == "__main__":
    verify_tools()