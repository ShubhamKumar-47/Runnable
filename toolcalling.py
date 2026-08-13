from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage
from rich import print 

#1 creating a tool 

@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}
llm = ChatMistralAI(model = "mistral-small-latest")

#tool binding 
llm_with_tool = llm.bind_tools([get_text_length])

message = []
prompt = input("You: ")
query = HumanMessage(prompt)
message.append(query)

result = llm_with_tool.invoke(message)

message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_fn = tools.get(tool_name)
    if tool_fn:
        tool_message = tool_fn.invoke(result.tool_calls[0])
        message.append(tool_message)
        result = llm_with_tool.invoke(message)

print(result.content)