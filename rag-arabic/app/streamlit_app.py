import streamlit as st
import os
from typing import Dict
from src.config import load_config
from src.rag_system import ArabicRAGSystem
from ui_strings import UI_STRINGS

def init_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'language' not in st.session_state:
        st.session_state.language = 'en'

def get_string(key: str) -> str:
    """Get UI string in current language."""
    return UI_STRINGS[st.session_state.language][key]

def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Language selector
    language = st.selectbox(
        "Language/اللغة",
        options=['en', 'ar'],
        index=0 if st.session_state.language == 'en' else 1
    )
    if language != st.session_state.language:
        st.session_state.language = language
        st.experimental_rerun()
    
    st.title(get_string("title"))
    
    # Document upload
    uploaded_file = st.file_uploader(
        get_string("upload_prompt"),
        type=['docx']
    )
    
    if uploaded_file:
        try:
            # Save uploaded file temporarily
            with open("temp_doc.docx", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process document
            with st.spinner(get_string("processing")):
                config = load_config()
                rag_system = ArabicRAGSystem(config)
                rag_system.process_document("temp_doc.docx")
                st.session_state.rag_system = rag_system
            
            # Clean up
            os.remove("temp_doc.docx")
            
        except Exception as e:
            st.error(f"{get_string('error_upload')}: {str(e)}")
            return
    
    # Query input
    query = st.text_input(
        "",
        placeholder=get_string("query_placeholder")
    )
    
    if st.button(get_string("submit_button")):
        if not st.session_state.rag_system:
            st.warning(get_string("no_document"))
            return
        
        try:
            response = st.session_state.rag_system.query(
                query,
                st.session_state.chat_history
            )
            
            # Display answer
            st.subheader(get_string("answer_header"))
            st.write(response["answer"])
            
            # Display sources
            st.subheader(get_string("sources_header"))
            for i, doc in enumerate(response["source_documents"], 1):
                with st.expander(f"Source {i}"):
                    st.write(doc.page_content)
            
            # Update chat history
            st.session_state.chat_history.append((query, response["answer"]))
            
        except Exception as e:
            st.error(f"{get_string('error_query')}: {str(e)}")

if __name__ == "__main__":
    main()
