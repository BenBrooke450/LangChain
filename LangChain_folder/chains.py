
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a viral Twitter influencer grading a tweet.
            Generate a critique and recommendations for the user.
            Always provide detailed recommendations, including suggestions
            for length, virality, and style."""
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Twitter tech influencer assistant tasked with
            writing excellent Twitter posts.

            Generate the best Twitter post possible for the user's request.

            If the user provides critique, respond with a revised version."""
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


llm = ChatOllama(model="qwen3:1.7b", temperature=0)
generation_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm