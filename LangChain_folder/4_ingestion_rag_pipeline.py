from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from mistralai.client import embeddings, Mistral
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
import os
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

load_dotenv()

location = "/Users/benjaminbrooke/Downloads/pinecone.txt"

if __name__ == "__main__":
    loader = TextLoader(file_path=location)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(documents)

    #embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=os.environ["MISTRAL_API_KEY"])

    embeddings = OllamaEmbeddings(model="embeddinggemma")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])



