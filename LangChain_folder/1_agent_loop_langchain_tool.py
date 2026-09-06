
from dotenv import load_dotenv

import os

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
#from langchain_core.globals import set_debug
from langsmith import traceable
import textwrap
import json

#set_debug(True)

load_dotenv()

MAX_ITERATIONS = 10

MODEL = "qwen2.5:3b"


#-------TOOLS (LangChain)-----------

@tool
def get_product_price(product:str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"  >> Executing get_product_prive(product='{product}')")
    prices ={"laptop": 1299.99, "headphones": 149.95, "keyboard":89.50}
    return prices.get(product)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the fincal price. Available tiers: bronze, silver, gold"""
    print(f"  >> Executing apply_discount(price={price}, tier={discount_tier})")
    discount_percentage = {"bronze":5, "silver":10, "gold":20}
    discount = discount_percentage.get(discount_tier)
    return round(price * (1 - discount / 100), 2)



#------- Agent Loop -----------


@traceable(name="My_loop_agent_using_langchain")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print("\n")

    print(f"Question: {question}")
    print("=" * 60)

    message = [SystemMessage(content="You are a helpful shopping assistant"
                                        "You have access to a product catalog tool"
                                        "and a discount tool.\n\n"
                                        "STRICT RULES - you must follow these exactly:\n"
                                        "1. NEVER guess or assume any product price."
                                        "You MUST call get_product_price first to get the real price.\n"
                                        "2. Only call apply_discount AFTER you have received"
                                        "returned by get_product_price - do NOT pass a made-up number.\n"
                                        "3. NEVER calculate discounts yourself tool.\n"
                                        "Always use the apply_discount tool. \n"
                                        "4. If the user does not specify a discount tier"
                                        "ask them which tier to use - do NOT assume one"
                                        ),
               HumanMessage(content=question)
               ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n===========>> Iteration {iteration} <<===========")


        ai_message = llm_with_tools.invoke(message)

        tool_call = ai_message.tool_calls

        if not tool_call:
            print(textwrap.fill(f"\nAnswer {iteration}:    {ai_message.content}",width=500))
            return ai_message.content

        tool_call = tool_call[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"\n [Tool Select] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f" [Tool Result] {observation}")

        message.append(ai_message)

        message.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max interations reached without a fincal answer")

    return None






if __name__ == "__main__":
    result = run_agent("What is the price of the laptop after applying a gold discount?")




