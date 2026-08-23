from dotenv import load_dotenv
import os
from langsmith import traceable
from mistralai.client import Mistral

load_dotenv()


@traceable(run_type="llm",metadata={"ls_provider": "mistral","ls_model_name": "mistral-small-latest"})


def query_mistral(prompt):
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message, response.choices[0].message.content


def main():

    response = query_mistral("Give me a few lines of how a electric engine works.")

    print(response[0],"\n", response[1])


if __name__ == "__main__":
    main()