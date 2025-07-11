from typing import TypedDict
from typing import Annotated
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START,END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt,Command

memory=MemorySaver()
load_dotenv()

class State(TypedDict):
    messages:Annotated[list,add_messages]

@tool
def get_stock_price(symbol:str)->float:
    '''Return the current price of a stock given the stock symbol
    :param symbol:stock symbol
    :return: current price of the stock
    '''
    return{
        "MSFT":200.3,
        "APPL":60.4,
        "AMZN":150.0,
        "RIL":87.6
    }.get(symbol,0.0)


@tool
def buy_stocks(symbol:str,quantity:int,total_price:float)->str:
    """BUy stocks given the symbol and quantity"""
    decision=interrupt(f"Aprrove")
    if decision=="yes":
        return f"You bought{quantity} shares of {symbol}"
    else:
        return "buying declined"

tools=[get_stock_price,buy_stocks]

llm=init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools=llm.bind_tools(tools) 

def chatbot(state:State)->State:
    state["messages"].append(llm_with_tools.invoke(state["messages"]))
    return state

builder=StateGraph(State)

builder.add_node("chatbot_node",chatbot)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"chatbot_node")
builder.add_conditional_edges("chatbot_node",tools_condition)
builder.add_edge("tools","chatbot_node")
builder.add_edge("chatbot_node",END)

graph=builder.compile(checkpointer=memory)

config={'configurable': {'thread_id':'1'}}

# state=graph.invoke({'messages':[{"role":"user","content":"what is the stock price of 20 AMZN stocks"}]},config=config)
# print(state["messages"][-1].content)

state=graph.invoke({'messages':[{"role":"user","content":"Buy 10 MSFT stocks at current price"}]},config=config)
print(state.get("__interrupt__"))

decision=input("Approve(yes/no):")
state=graph.invoke(Command(resume=decision),config=config)
print(state["messages"][-1].content)