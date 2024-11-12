import streamlit as st
import os
import sys
from ui_strings import UI_STRINGS
import streamlit_addons as st_addons
from src.config import load_config
from src.rag_system import ArabicRAGSystem

def init_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        config = load_config()
        st.session_state.rag_system = ArabicRAGSystem(config)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'language' not in st.session_state:
        st.session_state.language = 'ar'

def get_string(key: str) -> str:
    """Get UI string in current language."""
    return UI_STRINGS[st.session_state.language][key]

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Arabic Document Q&A", layout="wide")
    init_session_state()

    # Language selector
    col1, col2 = st.columns([1, 4])
    with col1:
        language = st.selectbox(
            get_string("language_selector"),
            options=['en', 'ar'],
            index=0 if st.session_state.language == 'en' else 1
        )
    if language != st.session_state.language:
        st.session_state.language = language

    st.title(get_string("title"))

    # Query interface
    with st.container():
        query = st.text_area(
            "",
            placeholder=get_string("query_placeholder"),
            height=100,
            key="query_input"
        )
        query_len = len(query.split())
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            pass
        with col2:
            if st.button(get_string("submit_button")):
                try:
                    with st.spinner(get_string("processing")):
                        response = st.session_state.rag_system.query(
                            query,
                            st.session_state.chat_history
                        )

                    # Display answer
                    st.subheader(get_string("answer_header"))
                    st_addons.annotated_text(response["answer"])

                    # Display sources
                    st.subheader(get_string("sources_header"))
                    for i, doc in enumerate(response["source_documents"], 1):
                        with st.expander(f"Source {i}"):
                            st_addons.annotated_text(doc.page_content)

                    # Update chat history
                    st.session_state.chat_history.append((query, response["answer"]))

                except Exception as e:
                    st.error(f"{get_string('error_query')}: {str(e)}")
        with col3:
            st.write(f"Words: {query_len}")

    # Chat history
    if st.session_state.chat_history:
        st.subheader(get_string("chat_history_header"))
        for i, (q, a) in enumerate(st.session_state.chat_history[::-1], 1):
            with st.expander(f"Query {i}"):
                st.write(f"**{get_string('query')}**: {q}")
                st.write(f"**{get_string('answer')}**: {a}")

if __name__ == "__main__":
    main()
