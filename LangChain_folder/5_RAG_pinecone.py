from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from mistralai.client import embeddings, Mistral
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import os

load_dotenv()


"""embedding = MistralAIEmbeddings(model="mistral-embed",api_key=os.environ["MISTRAL_API_KEY"])

llm = ChatMistralAI(model="mistral-small-latest",api_key=os.environ["MISTRAL_API_KEY"])"""

llm = ChatOllama(model="qwen2.5:3b")

embedding = OllamaEmbeddings(model="embeddinggemma")

vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"),embedding=embedding)

retrieve = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:
    
    """)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.

    Manually retrieves documents, formats them, and generates a response.

    Limitations:
     - Manual stap-by-step execution
     - No built-in streaming support
     - No async support without additional code
     - Harder to compose with other chain
     - More verbose and error-prone
    """
    docs =  retrieve.invoke(query)

    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context,question=query)

    response =  llm.invoke(messages)

    return response




if __name__ == "__main__":

    print("\n ============= IMPLEMENTATION 0 ============= \n ")

    query = "what is pinecone and what does it have to do with ML?"

    print(f"\n Question:{query} \n")

    answer = llm.invoke([HumanMessage(content=query)])

    print(f"\n Answer:{answer.content} \n")




    print("\n ============= IMPLEMENTATION 1 ============= \n ")

    query = "what is pinecone and what does it have to do with ML?"

    print(f"\n Question:{query} \n")

    answer = retrieval_chain_without_lcel(query)

    print(f"\n Answer:{answer} \n")





