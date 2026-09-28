

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


load_dotenv()


if __name__ == "__main__":
    print("Hello")