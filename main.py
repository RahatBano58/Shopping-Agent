from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool
import os
import requests
import rich

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# Gemini Client Setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define a function tool for product search
@function_tool
def get_products():
    """Fetch a list of products from the online API."""
    url = "https://template-03-api.vercel.app/api/products"
    try:
        response = requests.get(url)
        response.raise_for_status()
        products = response.json()
        return products
    except requests.RequestException as e:
        return {"error": str(e)}

# Define the shopping agent
agent = Agent(
    name="Shopping Agent",
    instructions = "You are a helpful shopping agent. Show products from the available list and suggest one that seems generally useful or appealing.",
    tools=[get_products]
)

# Run the agent synchronously
result = Runner.run_sync(
    agent,
    input = "Which product you want to buy?",
    run_config=config
)

# Display the result
rich.print(result.final_output)
