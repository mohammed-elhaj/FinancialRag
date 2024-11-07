from typing import Dict, Any, Optional
import google.generativeai as genai
from .config import RAGConfig
from .document_processor import DocumentProcessor
from .vector_store import VectorStoreHandler
from .qa_chain import QAChainHandler

class ArabicRAGSystem:
    """Main RAG system that coordinates all components."""
    
    def __init__(self, config: RAGConfig):
        """Initialize the RAG system with configuration."""
        self.config = config
        genai.configure(api_key=config.google_api_key)
        
        # Initialize components
        self.doc_processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
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
        
        self.vectorstore = None

    def process_document(self, file_path: str) -> None:
        """Process a document and prepare it for querying."""
        # Load and process document
        doc_text = self.doc_processor.load_word_document(file_path)
        chunks = self.doc_processor.process_text(doc_text)
        
        # Create vector store
        self.vectorstore = self.vector_handler.create_vectorstore(chunks)
        
        # Setup QA chain
        self.qa_handler.setup_chain(self.vectorstore)

    def query(self, question: str, chat_history: Optional[list] = None) -> Dict[str, Any]:
        """Process a query and return the response."""
        if not self.vectorstore:
            raise ValueError("No document has been processed. Please process a document first.")
        
        return self.qa_handler.query(question, chat_history)
