from langchain.prompts import PromptTemplate

TEMPLATES = {
    "ar": {
        "qa_template": """أنت مساعد متخصص في الإجابة على الأسئلة المتعلقة بنظام المنافسات والمشتريات الحكومية في المملكة العربية السعودية.
استخدم المعلومات التالية للإجابة على السؤال. إذا لم تكن المعلومات كافية، قل ذلك بوضوح.
المعلومات المتوفرة:
{context}

السؤال: {question}

قم بتقديم إجابة دقيقة ومباشرة مع ذكر رقم المادة والفصل ذي الصلة:""",
        "collection_name": "arabic_docs",
        "persist_directory": "./app/chroma_db_ar"
    },
    "en": {
        "qa_template": """You are a specialized assistant for answering questions about the Government Tenders and Procurement Law in Saudi Arabia.
Use the following information to answer the question. If the information is not sufficient, say so clearly.
Available Information:
{context}

Question: {question}

Provide a precise and direct answer, citing the relevant article and chapter numbers:""",
        "collection_name": "english_docs",
        "persist_directory": "./app/chroma_db_en"
    }
}

def get_prompt_template(lang: str) -> PromptTemplate:
    """Get the appropriate prompt template for the specified language."""
    if lang not in TEMPLATES:
        raise ValueError(f"Unsupported language: {lang}. Supported languages are: {list(TEMPLATES.keys())}")
    
    return PromptTemplate(
        template=TEMPLATES[lang]["qa_template"],
        input_variables=["context", "question"]
    )

def get_collection_name(lang: str) -> str:
    """Get the appropriate collection name for the specified language."""
    return TEMPLATES[lang]["collection_name"]

def get_persist_directory(lang: str) -> str:
    """Get the appropriate persist directory for the specified language."""
    return TEMPLATES[lang]["persist_directory"]
