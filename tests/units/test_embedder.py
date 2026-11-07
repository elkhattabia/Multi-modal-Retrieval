from pathlib import Path

from langchain_core.documents import Document

from multi_modal_retrieval import embedder


class FakeChroma:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.added_documents = None
        self.added_ids = None

    def add_documents(self, documents, ids):
        self.added_documents = documents
        self.added_ids = ids


def test_index_documents_uses_block_uuids_as_chroma_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(embedder, "Chroma", FakeChroma)
    documents = [
        Document(
            page_content="First block",
            metadata={"block_uuid": "block-1"},
        ),
        Document(
            page_content="Second block",
            metadata={"block_uuid": "block-2"},
        ),
    ]
    embedding = object()

    store = embedder.index_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=tmp_path / "chroma",
    )

    assert isinstance(store, FakeChroma)
    assert store.init_kwargs == {
        "collection_name": "tatdqa_dev",
        "embedding_function": embedding,
        "persist_directory": str(tmp_path / "chroma"),
    }
    assert store.added_documents == documents
    assert store.added_ids == ["block-1", "block-2"]


def test_index_documents_skips_add_when_input_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(embedder, "Chroma", FakeChroma)

    store = embedder.index_documents(
        documents=[],
        embedding=object(),
        persist_directory=Path(tmp_path / "chroma"),
    )

    assert isinstance(store, FakeChroma)
    assert store.added_documents is None
    assert store.added_ids is None
