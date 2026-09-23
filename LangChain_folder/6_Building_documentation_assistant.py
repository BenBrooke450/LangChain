import asyncio
import chunk
import os
import ssl
from typing import Any, Dict, List
import certifi
import json

from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from logger import Colors, log_info, log_warning, log_error, log_header, log_success

ss_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

embedding = OllamaEmbeddings(model="embeddinggemma")

llm = ChatOllama(model="qwen3:1.7b")

""""
MODEL = "qwen2.5:1.5b"

llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

answer = llm.invoke("hello, where is the most search location")

print(answer.content)
"""

vectorstore = PineconeVectorStore(index_name="med-doc", embedding=embedding)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=1,max_breath=20, max_pages=10)
tavily_crawl = TavilyCrawl(max_depth=1,max_breath=20, max_pages=10)



text_splitter = CharacterTextSplitter(chunk_size=200,chunk_overlap=0)



async def main():
    split_docs = []

    """Main async function to orchestrate the entire process."""

    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("Starting to Crawl documentation from https://www.langchain.com/",Colors.PURPLE)

    res = tavily_crawl.invoke({"url": "https://www.langchain.com/", "max_depth" : 1, "extract_depth" : "advanced", "instructions":"content on ai agents"})

    all_docs = res["results"]

    print(json.dumps(all_docs, indent=4))

    documents = [
        Document(
            page_content=doc["raw_content"],
            metadata={"url": doc["url"]}
        )
        for doc in all_docs
    ]

    split_docs = text_splitter.split_documents(documents)

    print(split_docs)

    PineconeVectorStore.from_documents(split_docs, embedding=embedding, index_name=os.environ["INDEX_NAME"])



if __name__ == "__main__":
    asyncio.run(main())

    vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embedding,pinecone_api_key=os.getenv("PINECONE_API_KEY"))

    retrieve = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context:

        {context}

        Question: {question}

        Provide a detailed answer:
        """
    )

    query = "Does langchain integrate pytest?"

    docs = retrieve.invoke(query)

    context = "\n\n".join(document.page_content for document in docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    print(response.content)




