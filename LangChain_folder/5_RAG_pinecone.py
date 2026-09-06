from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from mistralai.client import embeddings, Mistral
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
import os

load_dotenv()


embedding = MistralAIEmbeddings()

llm = Mistral()

vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"),embedding=embedding)

retrieve = vectorstore.as_retriever(search_kwargs={"k":3})

promopt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following conext:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:
    
    """)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)