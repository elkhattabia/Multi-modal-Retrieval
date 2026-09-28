import torch


def rank_pages(
    query_embedding: torch.Tensor, doc_embeddings: torch.Tensor
) -> list[tuple[int, float]]:
    """Ranks document pages by cosine similarity against a query.

    Args:
        query_embedding: Normalized vector of shape (1, D) or (D,).
        doc_embeddings: Normalized matrix of shape (N, D) for N pages.

    Returns:
        List of (page_index, cosine_similarity_score) sorted in descending order.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)

    # Inner product of L2-normalized tensors equals cosine similarity
    scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze(0)
    sorted_indices = torch.argsort(scores, descending=True)

    return [(idx.item(), scores[idx].item()) for idx in sorted_indices]
