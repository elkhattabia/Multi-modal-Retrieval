# I want to load the files content and metadata into a vector database for retrieval. I will use the `vectordb.py` module to handle the vectorization and storage of the document content and metadata.

"""Load page text blocks and metadata as LangChain documents."""

import json

from langchain_core.documents import Document

from multi_modal_retrieval.setup import Paths


def load_documents_from_directory(base_directory: Paths):
    """Load nonempty text blocks from the development split.

    Each block becomes a LangChain document with its document ID, page number,
    block order, UUID, and bounding box stored as metadata.

    Args:
        base_directory: Path resolver for the repository's TAT-DQA files.

    Returns:
        LangChain documents for the text blocks in the development split.
    """

    qa_path = base_directory.qa_data(split="dev")
    # tat_dqa_path =
    documents = []
    filenames = set()

    with open(qa_path, "r") as openfile:
        data = json.load(openfile)
    openfile.close()

    for dictionnary in data:
        filename = dictionnary["doc"]["uid"]
        filenames.add(filename)

    for filename in filenames:
        file_path = base_directory.file_path(split="dev", filename=filename)
        with open(file_path, "r") as f:
            file_content = json.load(f)

        for page_number, page in enumerate(file_content["pages"], start=1):
            blocks = sorted(page["blocks"], key=lambda block: block["order"])

            for block in blocks:
                text = block["text"].strip()
                if not text:
                    continue

                left, top, right, bottom = block["bbox"]

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "document_id": filename,
                            "page_number": page_number,
                            "block_uuid": block["uuid"],
                            "block_order": block["order"],
                            "bbox_left": left,
                            "bbox_top": top,
                            "bbox_right": right,
                            "bbox_bottom": bottom,
                        },
                    )
                )

    if not documents:
        raise SystemExit("No text blocks were loaded from the development split.")

    return documents
