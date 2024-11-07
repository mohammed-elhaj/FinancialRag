# Arabic RAG System

A Retrieval-Augmented Generation (RAG) system designed for querying Arabic-language documents using Google's Gemini model and ChromaDB for vector storage.

## Features

- Process Arabic Word documents
- Vector-based semantic search
- Bilingual interface (Arabic/English)
- Conversational memory
- Source document tracking
- User-friendly Streamlit interface

## Prerequisites

- Python 3.8+
- Google API key for Gemini
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-arabic.git
cd rag-arabic
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and add your Google API key:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

2. Upload an Arabic Word document through the web interface
3. Enter questions about the document content
4. View answers and source references

## Project Structure

```
rag-arabic/
├── src/               # Core RAG system components
├── app/               # Streamlit application
├── tests/             # Test files
├── requirements.txt   # Project dependencies
└── .env              # Environment variables
```

## Common Issues

1. **ChromaDB Collection Errors**
   - The system automatically handles existing collections
   - If you encounter persistent errors, delete the `chroma_db` directory

2. **Memory Issues**
   - Large documents may require more RAM
   - Reduce chunk_size in config.py if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
