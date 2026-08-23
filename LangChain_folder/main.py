from dotenv import load_dotenv
import os

from langsmith import traceable
from mistralai.client import Mistral

load_dotenv()


@traceable(run_type="llm",metadata={"ls_provider": "mistral","ls_model_name": "mistral-medium-latest"})


def main():
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "Give me a few lines of how a electric engine works."
            }
        ]
    )

    print(response)
    print("----------CHECK----------")
    print(response.choices[0].message)
    print("----------CHECK----------")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()

