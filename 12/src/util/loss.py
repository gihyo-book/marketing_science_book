import torch
import torch.nn.functional as F


def bpr_loss(positive_scores, negative_scores):
    """Return the mean BPR loss for positive and negative item scores."""
    return F.softplus(negative_scores - positive_scores).mean()


def squared_embedding_penalty(*embeddings):
    """Return the sum of squared embedding values per training example."""
    squared_sums = [embedding.square().sum() for embedding in embeddings]
    return torch.stack(squared_sums).sum() / embeddings[0].shape[0]
