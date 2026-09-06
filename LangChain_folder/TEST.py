import os

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.environ["MISTRAL_API_KEY"]
)

vector = embeddings.embed_query("What is Python?")

print("Embedding created!")
print("Dimensions:", len(vector))
print("First 5 values:", vector[:5])