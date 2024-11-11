from typing import Dict, Any, Optional
import google.generativeai as genai
from .config import RAGConfig
from .vector_store import VectorStoreHandler
from .qa_chain import QAChainHandler

class ArabicRAGSystem:
    """Main RAG system using pre-generated database."""
    
    def __init__(self, config: RAGConfig):
        """Initialize the RAG system with configuration."""
        self.config = config
        genai.configure(api_key=config.google_api_key)
        
        # Initialize components
        self.vector_handler = VectorStoreHandler(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model
        )
        
        self.qa_handler = QAChainHandler(
            google_api_key=config.google_api_key,
            model_name=config.llm_model,
            temperature=config.temperature
        )
        
        # Setup QA chain with loaded vectorstore
        self.qa_handler.setup_chain(self.vector_handler.get_vectorstore())

    def query(self, question: str, chat_history: Optional[list] = None) -> Dict[str, Any]:
        """Process a query and return the response."""
        return self.qa_handler.query(question, chat_history)
