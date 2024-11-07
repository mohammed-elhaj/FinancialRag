from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma

class QAChainHandler:
    """Handles question-answering chain operations."""
    
    def __init__(self, google_api_key: str, model_name: str, temperature: float):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        self.qa_chain = None

    def setup_chain(self, vectorstore: Chroma):
        """Set up the question-answering chain."""
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True
        )

    def query(self, question: str, chat_history: List = None) -> Dict[str, Any]:
        """Process a query and return the response with sources."""
        if not self.qa_chain:
            raise ValueError("QA Chain not initialized. Please run setup_chain first.")
        
        try:
            response = self.qa_chain({
                "question": question,
                "chat_history": chat_history or []
            })
            
            return {
                "answer": response["answer"],
                "source_documents": response["source_documents"]
            }
        except Exception as e:
            raise Exception(f"Error during query: {str(e)}")
