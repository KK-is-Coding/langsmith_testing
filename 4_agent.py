import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

os.environ["LANGCHAIN_PROJECT"] = "agent_demo"

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherapi.com/v1/forecast.json?key=bb54235950ab4953856151824262511&q={city}&days=3&aqi=no&alerts=yes'

  response = requests.get(url)

  return response.json()

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=8)

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "Identify the birthplace of current Prime Minister of India and give its current temperature."})
print(response)

print(response['output'])