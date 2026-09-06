from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from mistralai.client import embeddings, Mistral
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
import os

load_dotenv()

location = "/Users/benjaminbrooke/Downloads/Python-for-Data-Analysis.txt"

if __name__ == "__main__":
    loader = TextLoader(file_path=location)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(documents)

    embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=os.environ["MISTRAL_API_KEY"])

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])



