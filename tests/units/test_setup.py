from multi_modal_retrieval.setup import Paths


def test_paths_build_dataset_paths(tmp_path):
    paths = Paths()
    paths.base_path = tmp_path

    assert paths.data() == tmp_path / "data" / "TAT-DQA"
    assert paths.file_path("dev", "document-id") == (
        tmp_path / "data" / "TAT-DQA" / "dev" / "document-id.json"
    )
    assert paths.qa_data("dev") == (
        tmp_path / "data" / "TAT-DQA" / "tatdqa_dataset_dev.json"
    )
