
from dotenv import load_dotenv
from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap, TavilySearch
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


@tool
def triple(num:float) -> float:
    """
    :param num: a number to triple
    :return: the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1),triple]

llm = ChatOllama(model="qwen3:1.7b", temperature=0).bind_tools(tools)

message = llm.invoke("is this working?")

print(message.content)