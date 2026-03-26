from langchain_openai import OpenAIEmbeddings
import os

def get_embeddings():
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )