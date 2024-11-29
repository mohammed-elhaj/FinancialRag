import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from .templates import get_collection_name, get_persist_directory

class VectorStoreHandler:
    """Handles vector storage operations with language support."""
    
    def __init__(self, persist_directory: str, collection_name: str, openai_api_key: str, lang: str):
        self.lang = lang
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Get language-specific settings
        self.persist_directory = get_persist_directory(lang)
        self.collection_name = get_collection_name(lang)
        
        # Load existing collection
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
    
    def get_vectorstore(self) -> Chroma:
        """Get the loaded vector store."""
        return self.vectorstore
