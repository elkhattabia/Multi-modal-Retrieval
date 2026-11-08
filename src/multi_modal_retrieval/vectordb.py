"""Build the development split's persistent text vector index."""

import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from multi_modal_retrieval.setup import Paths
from multi_modal_retrieval.vectorize.embedder import index_documents
from multi_modal_retrieval.vectorize.loader import load_documents_from_directory


def run_db(api_key: str) -> None:
    """Load, embed, and index the development dataset.

    Raises:
        SystemExit: If no Google embedding API key is configured or the loader
            returns no text blocks.
    """
    paths = Paths()

    blocks = load_documents_from_directory(paths)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    persist_directory = paths.base_path / ".chroma" / "tatdqa_dev"
    store = index_documents(
        documents=blocks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print(f"Loaded {len(blocks)} blocks and indexed {len(blocks)} chunks.")
    print(f"Chroma collection stored at: {persist_directory}")
    return store


if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in the .env file.")

    run_db(api_key)
