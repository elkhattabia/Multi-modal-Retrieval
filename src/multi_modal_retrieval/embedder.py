from collections.abc import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SigLIPEmbedder:
    """Computes joint L2-normalized metric embeddings for text queries and page images."""

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def embed_images(
        self, images: Sequence[Image.Image], batch_size: int = 8
    ) -> torch.Tensor:
        """Encodes PIL images into L2-normalized embeddings of shape (N, D).

        Processes images in batches to prevent CUDA/system OOM on large documents.
        """
        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(images), batch_size):
            batch = list(images[i : i + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            features = getattr(
                self.model.get_image_features(**inputs), "pooler_output", None
            )
            norm_features = F.normalize(features, p=2, dim=-1)
            all_embeddings.append(norm_features.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def embed_text(self, text: str | Sequence[str]) -> torch.Tensor:
        """Encodes text queries into L2-normalized embeddings of shape (N, D)."""
        if isinstance(text, str):
            text = [text]

        inputs = self.processor(
            text=list(text), padding="max_length", return_tensors="pt"
        ).to(self.device)
        features = self.model.get_text_features(**inputs).pooler_output
        return F.normalize(features, p=2, dim=-1).cpu()
