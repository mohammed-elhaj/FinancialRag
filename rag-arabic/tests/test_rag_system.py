import os
import pytest
from src.config import RAGConfig, load_config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreHandler
from src.qa_chain import QAChainHandler
from src.rag_system import ArabicRAGSystem

@pytest.fixture
def test_config():
    return RAGConfig(
        google_api_key="test_key",
        chunk_size=100,
        chunk_overlap=20
    )

@pytest.fixture
def doc_processor():
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)

def test_document_processor_initialization(doc_processor):
    assert doc_processor.text_splitter is not None
    assert doc_processor.text_splitter.chunk_size == 100
    assert doc_processor.text_splitter.chunk_overlap == 20

def test_text_processing(doc_processor):
    test_text = "This is a test document.\nIt has multiple lines.\nSome are in Arabic.\nمرحبا بالعالم"
    chunks = doc_processor.process_text(test_text)
    assert len(chunks) > 0
    assert all(len(chunk.strip()) > 50 for chunk in chunks)

@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not set")
def test_full_rag_system():
    config = load_config()
    rag_system = ArabicRAGSystem(config)
    
    # Test document processing
    test_text = "This is a test document.\n" * 10
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write(test_text)
    
    try:
        # Process document
        rag_system.process_document("test_doc.txt")
        
        # Test querying
        response = rag_system.query("What is this document about?")
        assert "answer" in response
        assert "source_documents" in response
        assert len(response["source_documents"]) > 0
        
    finally:
        # Cleanup
        if os.path.exists("test_doc.txt"):
            os.remove("test_doc.txt")

def test_vector_store_handler(test_config):
    handler = VectorStoreHandler(
        persist_directory="./test_chroma_db",
        collection_name="test_collection",
        embedding_model=test_config.embedding_model
    )
    assert handler.embedding_dimension > 0
    assert handler.chroma_client is not None

def test_qa_chain_handler(test_config):
    handler = QAChainHandler(
        google_api_key=test_config.google_api_key,
        model_name=test_config.llm_model,
        temperature=test_config.temperature
    )
    assert handler.llm is not None
    assert handler.qa_chain is None  # Should be None until setup_chain is called
