import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    """Configuration settings for the RAG system."""
    google_api_key: str
    openai_api_key:str
    embedding_model: str = "intfloat/multilingual-e5-large"
    llm_model: str = "gemini-1.5-pro"
    chunk_size: int = 500
    chunk_overlap: int = 50
    temperature: float = 0.3
    persist_directory: str = os.path.abspath("./app/chroma_db2")
    collection_name: str = "arabic_docs"

def load_config() -> RAGConfig:
    """Load configuration from environment variables."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    return RAGConfig(google_api_key=google_api_key,openai_api_key=openai_api_key)
