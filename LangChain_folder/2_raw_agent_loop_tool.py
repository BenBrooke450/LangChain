
from dotenv import load_dotenv

import os

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
#from langchain_core.globals import set_debug
from langsmith import traceable
import textwrap
import ollama
import json

#set_debug(True)

load_dotenv()

MAX_ITERATIONS = 10

MODEL = "qwen2.5:3b"


#-------TOOLS (LangChain)-----------

@traceable(run_type="tool")
def get_product_price(product:str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"  >> Executing get_product_prive(product='{product}')")
    prices ={"laptop": 1299.99, "headphones": 149.95, "keyboard":89.50}
    return prices.get(product)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the fincal price. Available tiers: bronze, silver, gold"""
    print(f"  >> Executing apply_discount(price={price}, tier={discount_tier})")
    discount_percentage = {"bronze":5, "silver":10, "gold":20}
    discount = discount_percentage.get(discount_tier)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name: 'laptop', 'headphones', 'keyboard'"
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "Discount tier: 'bronze', 'silver', 'gold'"
                    }
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]

@traceable(name="Ollama chat",run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


#------- Agent Loop -----------


@traceable(name="My_loop_agent_raw")
def run_agent(question:str):

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"\nQuestion: {question}")
    print("=" * 60)

    message = [{"role":"system",
                "content": ("You are a helpful shopping assistant\n"
                                        "You have access to a product catalog tool\n"
                                        "and a discount tool.\n\n"
                                        "STRICT RULES - you must follow these exactly:\n"
                                        "1. NEVER guess or assume any product price.\n\n"
                                        "You MUST call get_product_price first to get the real price.\n"
                                        "2. Only call apply_discount AFTER you have received\n"
                                        "returned by get_product_price - do NOT pass a made-up number.\n"
                                        "3. NEVER calculate discounts yourself tool.\n"
                                        "Always use the apply_discount tool. \n"
                                        "4. If the user does not specify a discount tier\n"
                                        "ask them which tier to use - do NOT assume one\n"
                                        ),
                },
               {"role":"user","content":question}
               ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n===========>> Iteration {iteration} <<===========")

        response = ollama_chat_traced(messages=message)

        ai_message = response.message

        print(f"[This is the AI message]:{ai_message}")

        print(f"[FULL AI message]:{response}")

        tool_call = ai_message.tool_calls

        if not tool_call:
            print(textwrap.fill(f"\nAnswer {iteration}:    {ai_message.content}",width=500))
            return ai_message.content

        tool_call = tool_call[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"\n [Tool Select] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")


        observation = tool_to_use(**tool_args)

        print(f" [Tool Result] {observation}")

        message.append(ai_message)

        message.append(
            {"role":"tool",
            "tool_name": tool_name,
            "content": str(observation)}

        )

    print("ERROR: Max interations reached without a fincal answer")

    return None






if __name__ == "__main__":
    result = run_agent("What is the price of the laptop after applying a gold discount?")




