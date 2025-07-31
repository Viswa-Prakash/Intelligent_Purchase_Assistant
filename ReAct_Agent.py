from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage, AnyMessage

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool 

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
alphavantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")


llm = init_chat_model("gpt-4.1", temperature=0.7)

serpapi_tool = Tool(
    name="serpapi",
    description="Optimizes e-commerce purchase flows by analyzing real-time search engine results for product visibility, competitor pricing, and market trends, improving conversion rates.",
    func=SerpAPIWrapper().run,
)

alpha_vantage_api = AlphaVantageAPIWrapper(alphavantage_api_key=alphavantage_api_key)

@tool
def currency_tool(from_currency: str, to_currency: str) -> str:
    """Get exchange rate between two currencies."""
    return alpha_vantage_api.run(from_currency=from_currency, to_currency=to_currency)

repl_tool = PythonREPLTool()

# Create the list of properly wrapped Tool instances
tools = [serpapi_tool, currency_tool, repl_tool]

react_prompt = """
You are an AI assistant specialized in helping users find the best value when shopping online.

You can perform the following actions/tools:
- Use a **Product Search Tool** to find current prices, sellers, and offers for the product specified by the user.
- Use a **Review Aggregator Tool** to gather and summarize average user ratings or top reviews.
- Use a **Currency Converter Tool** to convert prices between currencies.
- Use the **Python REPL tool** to perform any calculations, such as total cost, percent savings, or final price with tax.

For each request:
- ALWAYS break the user’s question into clear subtasks, think out loud about each step, and choose the right tool for each part.
- If a step fails, try an alternative approach or suggest other options.

When you have completed all necessary tool use and gathered all information:
- ALWAYS output a final message that **begins with the phrase "Final answer:"** (e.g., `Final answer: The Kindle Paperwhite is cheapest at ...`).
- Clearly summarize product(s) found with price, currency conversion, and seller.
- State the average user rating (if available).
- Calculate the total estimated cost in the user's preferred currency, including conversion and any important fees.
- Give brief advice or next steps, for example: "You may want to check warranty terms from overseas sellers."

**Do not give your final answer until all required tools have been used. Your very last message must always begin with "Final answer:".**
"""


class State(TypedDict):
    messages : Annotated[list[AnyMessage], add_messages]

def reasoning_node(state: State):
    # LLM with bound tools to enable tool-calling
    llm_with_tools = llm.bind_tools(tools)
    messages = [{"role": "system", "content": react_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}


tool_node = ToolNode(tools = tools)


def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "content") and "final answer:" in last_message.content.lower():
        return "end"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    if len(state["messages"]) > 20:
        return "end"
    # Otherwise, no tool_calls, not a final answer, so end gracefully
    return "end"


builder = StateGraph(State)
builder.add_node("reason", reasoning_node)
builder.add_node("action", tool_node)
builder.set_entry_point("reason")
builder.add_conditional_edges(
    "reason",
    should_continue,
    {
        "continue": "action",
        "end": END,
    }
)
builder.add_edge("action", "reason")
ecommerce_agent = builder.compile()
