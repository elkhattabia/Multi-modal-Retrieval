from multi_modal_retrieval import vectordb


def test_main_stops_when_gemini_api_key_is_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(vectordb, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(
        vectordb, "Paths", lambda: type("TestPaths", (), {"base_path": tmp_path})()
    )

    try:
        vectordb.run_db()
    except SystemExit as error:
        assert "GEMINI_API_KEY" in str(error)
    else:
        raise AssertionError("run_db() should stop when the API key is missing")


def test_main_loads_blocks_and_indexes_them(monkeypatch, tmp_path, capsys):
    api_key = "test-api-key"
    blocks = [object(), object()]
    embedding = object()
    calls = {}

    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    monkeypatch.setattr(vectordb, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(
        vectordb,
        "Paths",
        lambda: type("TestPaths", (), {"base_path": tmp_path})(),
    )
    monkeypatch.setattr(
        vectordb,
        "load_documents_from_directory",
        lambda paths: blocks,
    )

    def create_embeddings(**kwargs):
        calls["embedding_kwargs"] = kwargs
        return embedding

    def index_documents(**kwargs):
        calls["index_kwargs"] = kwargs

    monkeypatch.setattr(vectordb, "GoogleGenerativeAIEmbeddings", create_embeddings)
    monkeypatch.setattr(vectordb, "index_documents", index_documents)

    vectordb.run_db()

    assert calls["embedding_kwargs"] == {
        "model": "models/gemini-embedding-001",
        "google_api_key": api_key,
    }
    assert calls["index_kwargs"] == {
        "documents": blocks,
        "embedding": embedding,
        "persist_directory": tmp_path / ".chroma" / "tatdqa_dev",
    }
    assert "Loaded 2 blocks" in capsys.readouterr().out
