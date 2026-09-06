
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
import re
import inspect


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




@traceable(name="Ollama chat",run_type="llm")
def ollama_chat_traced(model,messages,options):
    return ollama.chat(model=MODEL, messages=messages, options = options)


tools = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

def get_tool_description(tools_dict):
    description = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function,"__wrapped__",tool_function)

        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        description.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(description)

tool_description = get_tool_description(tools)
tool_names = ", ".join(tools.keys())


react_prompt = f'''Answer the following questions as best you can. You have access to the following tools:

1. NEVER guess or assume any product price.\n\n
2. Only call apply_discount AFTER you have received\n\n
3. NEVER calculate discounts yourself tool.\n\n
4. If the user does not specify a discount tier\n\n

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:'''



#------- Agent Loop -----------


@traceable(name="My_loop_agent_raw")
def run_agent(question:str):


    print(f"Question: {question}")
    print("=" * 60)


    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS+1):

        full_prompt = prompt + scratchpad

        print(f"\n===========>> Iteration {iteration} <<===========")

        response = ollama_chat_traced(model=MODEL,messages=[{"role":"user","content":full_prompt}],
                                                            options={"stop": ["\nObservation"],"temperature":0})

        output = response.message.content

        final_answer = re.search(f"Final Answer:\s*(.+)", output)

        if final_answer:
            final_answer = final_answer.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        print(f"[This is the AI message]:{final_answer}")

        tool_call = final_answer.tool_calls

        if not tool_call:
            print(textwrap.fill(f"\nAnswer {iteration}:    {ai_message.content}",width=500))
            return ai_message.content

        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"\n [Tool Select] {tool_name} with args: {tool_args}")

        tool_to_use = tools.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use(**tool_args)

        print(f" [Tool Result] {observation}")

        scratchpad.append(ai_message)

        scratchpad.append(
            {"role":"tool",
            "tool_name": tool_name,
            "content": str(observation)}

        )

    print("ERROR: Max interations reached without a fincal answer")

    return None






if __name__ == "__main__":
    result = run_agent("What is the price of the laptop after applying a gold discount?")




