import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")
embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL")
embeddings = OpenAIEmbeddings(
    model=embedding_model,
    api_key=api_key,
    base_url=base_url
    )