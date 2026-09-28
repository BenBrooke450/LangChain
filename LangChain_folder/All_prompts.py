
from mistralai.client import Mistral
from dotenv import load_dotenv
import os
load_dotenv()


from mistralai.client import Mistral
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
response = client.chat.complete(model="mistral-small-latest",messages=[{"role": "user","content": "Explain what a transformer is in simple terms."}])
print(response.choices[0].message.content)



from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(model="mistral-small-latest",api_key=api_key)
response = llm.invoke("Explain what a transformer is in simple terms.")
print(response.content)



from langchain.chat_models import init_chat_model
MODEL = "qwen2.5:1.5b"
llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
answer = llm.invoke("hello, where is the most search location")
print(answer.content)
