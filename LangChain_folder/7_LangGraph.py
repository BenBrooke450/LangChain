

from dotenv import load_dotenv
from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()


def  should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON,run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT,tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
        END:END,
        ACT:ACT})

flow.add_edge(ACT,AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello")

    res = app.invoke({"messages":[HumanMessage(content="What is the weather in Tokyo? List it and then triple it")]})

    print(res["messages"][LAST].content)