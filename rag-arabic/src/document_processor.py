from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class DocumentProcessor:
    """Handles document loading and text processing."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "؟", "،", " "]
        )

    def load_word_document(self, file_path: str) -> str:
        """Load and extract text from a Word document."""
        doc = Document(file_path)
        text = "\n".join(
            paragraph.text.strip()
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )
        return text

    def process_text(self, text: str) -> List[str]:
        """Split text into chunks for processing."""
        chunks = self.text_splitter.split_text(text)
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
