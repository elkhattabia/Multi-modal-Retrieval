# Multi-Modal Retrieval

Retrieve relevant pages from a PDF using SigLIP image and text embeddings.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Put a PDF at:

```text
data/
```

Run the example:

```bash
uv run python -m multi_modal_retrieval.smoke
```

The first run downloads the SigLIP model from Hugging Face. The script renders
the PDF pages, embeds them, embeds the search query, and prints the five most
similar pages.

Change the query in `src/multi_modal_retrieval/smoke.py` to search for another
topic.

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
```

This project currently ranks pages. It does not yet generate answers from the
retrieved content.
