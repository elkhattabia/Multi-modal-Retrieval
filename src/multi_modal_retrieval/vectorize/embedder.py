"""Create and populate a persistent Chroma vector store."""

from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def index_documents(
    documents: Sequence[Document],
    embedding: Embeddings,
    persist_directory: Path,
    collection_name: str = "tatdqa_dev",
) -> Chroma:
    """Add documents to a persistent Chroma collection.

    Document IDs are taken from each document's ``block_uuid`` metadata.

    Args:
        documents: Chunks to embed and store.
        embedding: Embedding implementation used to vectorize the documents.
        persist_directory: Directory where Chroma stores the collection.
        collection_name: Name of the collection to populate.

    Returns:
        The Chroma vector store containing the indexed documents.
    """
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(persist_directory),
    )

    documents = list(documents)
    if documents:
        ids = [str(document.metadata["block_uuid"]) for document in documents]
        store.add_documents(documents=documents, ids=ids)

    return store
