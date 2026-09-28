from pathlib import Path

from multi_modal_retrieval.embedder import SigLIPEmbedder
from multi_modal_retrieval.pdf_loader import render_pdf_to_images
from multi_modal_retrieval.search import rank_pages

# 1. Render pages
pdf_path = Path("/home/anas/Multi-modal-Retrieval/data/pharma.pdf")
print(f"Loading and rasterizing: {pdf_path}...")
images = render_pdf_to_images(pdf_path, dpi=150)
print(f"Rendered {len(images)} pages.")

# 2. Embed
print("Initializing SigLIP...")
embedder = SigLIPEmbedder()

print("Embedding document pages...")
page_embeddings = embedder.embed_images(images, batch_size=4)

query = "Recommendation on target blood pressure"
print(f"Embedding query: '{query}'...")
query_embedding = embedder.embed_text(query)

# 3. Search and rank
results = rank_pages(query_embedding, page_embeddings)

print("\n--- Retrieval Results ---")
for rank, (page_idx, score) in enumerate(results[:5], start=1):
    print(f"Rank {rank}: Page {page_idx + 1} | Cosine Similarity: {score:.4f}")
