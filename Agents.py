from dotenv import load_dotenv
import os
import sys
import requests

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tavily import TavilyClient
from rich import print

# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY is not set in environment."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"
    
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    
    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool (Tavily)
# =========================

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set in environment."
        
    try:
        tavily_client = TavilyClient(api_key=api_key)
        response = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3
        )
    except Exception as e:
        return f"Error fetching news data: {str(e)}"
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

# =========================
# 🧠 LLM Setup & Human Approval
# =========================

llm = ChatMistralAI(model="mistral-small-latest")

raw_tools = [get_weather, get_news]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful city assistant."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, raw_tools, prompt)

def wrap_with_approval(func_tool):
    @tool(name=func_tool.name, description=func_tool.description, args_schema=func_tool.args_schema)
    def approved_tool(*args, **kwargs):
        confirm = input(f"Agent wants to call '{func_tool.name}'. Approve? (yes/no): ")
        if confirm.lower() != "yes":
            return "Tool call denied by user."
        tool_input = kwargs if kwargs else (args[0] if args else {})
        return func_tool.invoke(tool_input)
    return approved_tool

approved_tools = [wrap_with_approval(t) for t in raw_tools]
agent_executor = AgentExecutor(agent=agent, tools=approved_tools, verbose=False)

print("City Agent | type exit to quit")

chat_history = []

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break 
    result = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    bot_reply = result['output']
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=bot_reply))

    print("bot : ", bot_reply)