"""Resolve repository paths used by the retrieval data pipeline."""

from pathlib import Path


class Paths:
    """Build paths to the dataset and its split files."""

    def __init__(self) -> None:
        """Initialize paths relative to the repository root."""
        self.base_path = Path(__file__).resolve().parents[2]

    def data(self) -> Path:
        """Return the root directory for the TAT-DQA dataset.

        Returns:
            The path to the dataset directory.
        """
        return self.base_path / "data" / "TAT-DQA"

    def file_path(self, split: str, filename: str) -> Path:
        """Build the path to one document JSON file.

        Args:
            split: Dataset split containing the document.
            filename: Document identifier without the ``.json`` suffix.

        Returns:
            The path to the document JSON file.
        """
        return self.data() / split / f"{filename}.json"

    def qa_data(self, split: str) -> Path:
        """Build the path to a split's question-and-answer dataset file.

        Args:
            split: Dataset split, such as ``train`` or ``dev``.

        Returns:
            The path to the split's dataset JSON file.
        """
        return self.data() / f"tatdqa_dataset_{split}.json"
