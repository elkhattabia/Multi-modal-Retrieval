from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def render_pdf_to_images(pdf_path: str | Path, dpi: int = 150) -> list[Image.Image]:
    """Renders every page of a PDF document into a list of PIL Images.

    Args:
        pdf_path: Path to the target PDF file.
        dpi: Target dots per inch for rendering resolution (default: 150).

    Returns:
        List of PIL.Image.Image objects corresponding to each page in order.

    Raises:
        FileNotFoundError: If the specified pdf_path does not exist.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found at: {path.resolve()}")

    pdf = pdfium.PdfDocument(path)
    # PDF points are 1/72 inch apart; scale determines rendered image resolution
    scale = dpi / 72.0

    images: list[Image.Image] = []
    for page in pdf:
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil())

    pdf.close()
    return images
