"""Question embedding and context retrieval from vector database"""

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def embed_question(question: str, api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    return embeddings.embed_query(question)


def retrieve_context_from_vectordb(question_embedding: list[float], vectordb: Chroma):
    """Retrieve relevant context from the vector database using the question embedding."""
    results = vectordb.similarity_search_by_vector(question_embedding, k=5)

    context = "\n\n".join(
        f"[Score {result.score}]\n"
        f"Document ID: {result.metadata['document_id']}"
        f"[Page {result.metadata['page_number']}, "
        f"block {result.metadata['block_uuid']}]\n"
        f"{result.page_content}"
        for result in results
    )
    return context
