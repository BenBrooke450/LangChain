from dotenv import load_dotenv
import os

load_dotenv()


if __name__ == "__main__":
    print(os.environ["INDEX_NAME"])