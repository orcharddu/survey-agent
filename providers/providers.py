from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL

from functools import lru_cache


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.7,
        max_tokens=4096
    )

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache(maxsize=10)
def get_vectorstore(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory="./vector_store/chroma_db",
    )
