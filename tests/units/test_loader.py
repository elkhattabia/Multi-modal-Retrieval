import json

import pytest

from multi_modal_retrieval.loader import load_documents_from_directory
from multi_modal_retrieval.setup import Paths


def _write_dataset(tmp_path, qa_records, document_files):
    paths = Paths()
    paths.base_path = tmp_path
    dataset_dir = paths.data()
    dev_dir = dataset_dir / "dev"
    dev_dir.mkdir(parents=True)

    with paths.qa_data("dev").open("w", encoding="utf-8") as file:
        json.dump(qa_records, file)

    for document_id, document_data in document_files.items():
        with paths.file_path("dev", document_id).open("w", encoding="utf-8") as file:
            json.dump(document_data, file)

    return paths


def test_load_documents_sorts_blocks_and_preserves_metadata(tmp_path):
    paths = _write_dataset(
        tmp_path,
        [
            {"doc": {"uid": "document-1"}},
            {"doc": {"uid": "document-1"}},
        ],
        {
            "document-1": {
                "pages": [
                    {
                        "blocks": [
                            {
                                "order": 2,
                                "text": " second ",
                                "uuid": "block-2",
                                "bbox": [2, 3, 4, 5],
                            },
                            {
                                "order": 1,
                                "text": " first ",
                                "uuid": "block-1",
                                "bbox": [1, 2, 3, 4],
                            },
                            {
                                "order": 3,
                                "text": "   ",
                                "uuid": "blank-block",
                                "bbox": [0, 0, 0, 0],
                            },
                        ]
                    }
                ]
            }
        },
    )

    documents = load_documents_from_directory(paths)

    assert [document.page_content for document in documents] == ["first", "second"]
    assert documents[0].metadata == {
        "document_id": "document-1",
        "page_number": 1,
        "block_uuid": "block-1",
        "block_order": 1,
        "bbox_left": 1,
        "bbox_top": 2,
        "bbox_right": 3,
        "bbox_bottom": 4,
    }


def test_load_documents_fails_when_there_are_no_text_blocks(tmp_path):
    paths = _write_dataset(
        tmp_path,
        [{"doc": {"uid": "document-1"}}],
        {
            "document-1": {
                "pages": [
                    {
                        "blocks": [
                            {
                                "order": 1,
                                "text": "  ",
                                "uuid": "blank-block",
                                "bbox": [0, 0, 0, 0],
                            }
                        ]
                    }
                ],
            }
        },
    )

    with pytest.raises(SystemExit, match="No text blocks"):
        load_documents_from_directory(paths)
