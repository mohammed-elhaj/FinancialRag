import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreHandler:
    """Handles vector storage operations with pre-generated database."""
    
    def __init__(self, persist_directory: str, collection_name: str, embedding_model: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Load existing collection
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
            client=self.chroma_client
        )
    
    def get_vectorstore(self) -> Chroma:
        """Get the loaded vector store."""
        return self.vectorstore
