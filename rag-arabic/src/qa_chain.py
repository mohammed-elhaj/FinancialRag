from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from .templates import get_prompt_template

class QAChainHandler:
    """Handles question-answering chain operations with language support."""
    
    def __init__(self, openai_api_key: str, model_name: str, temperature: float, lang: str):
        self.lang = lang
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature
        )
        self.qa_chain = None
        self.prompt_template = get_prompt_template(lang)

    def setup_chain(self, vectorstore: Chroma):
        """Set up the question-answering chain."""
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": self.prompt_template}
        )

    def format_source_document(self, doc) -> str:
        """Format source document for display with metadata based on language."""
        metadata = doc.metadata
        if self.lang == "ar":
            return (
                f"المادة {metadata['article_number']}\n"
                f"الفصل {metadata['chapter_number']}: {metadata['chapter_name']}\n"
                f"القسم {metadata['section_number']}: {metadata['section_name']}\n"
                f"ملخص: {metadata['summary']}\n"
                f"النص الكامل:\n{doc.page_content}"
            )
        else:
            return (
                f"Article {metadata['article_number']}\n"
                f"Chapter {metadata['chapter_number']}: {metadata['chapter_name']}\n"
                f"Section {metadata['section_number']}: {metadata['section_name']}\n"
                f"Summary: {metadata['summary']}\n"
                f"Full Text:\n{doc.page_content}"
            )

    def query(self, question: str, chat_history: List = None) -> Dict[str, Any]:
        """Process a query and return the response with sources."""
        if not self.qa_chain:
            raise ValueError("QA Chain not initialized. Please run setup_chain first.")
        
        chat_history = chat_history or []
        response = self.qa_chain({
            "question": question,
            "chat_history": chat_history
        })
        
        formatted_sources = [
            self.format_source_document(doc)
            for doc in response["source_documents"]
        ]

        return {
            "answer": response["answer"],
            "source_documents": response["source_documents"],
            "formatted_sources": formatted_sources
        }
