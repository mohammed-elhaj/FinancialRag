import os
import shutil
from typing import List
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreHandler:
    """Handles vector storage operations."""
    
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
        self._init_chroma_client()

    def _init_chroma_client(self):
        """Initialize or reset ChromaDB client."""
        if os.path.exists(self.persist_directory):
            try:
                # Try to get existing collection
                client = chromadb.Client(Settings(
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False
                ))
                client.get_collection(self.collection_name)
            except ValueError:
                # Collection doesn't exist, clear directory
                shutil.rmtree(self.persist_directory)
        
        # Create new client
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get embedding dimension
        self.embedding_dimension = len(self.embeddings.embed_query("test"))

    def create_vectorstore(self, chunks: List[str]) -> Chroma:
        """Create or update vector store from text chunks."""
        try:
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except ValueError:
                pass
            
            # Create new collection
            self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "dimension": self.embedding_dimension}
            )
            
            # Create vector store
            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_metadata={"dimension": self.embedding_dimension},
                client=self.chroma_client,
                collection_name=self.collection_name
            )
            
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
