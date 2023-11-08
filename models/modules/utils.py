import torch


def sinusodial(t, embedding_dim, device="cuda"):
    emb = 1 / (
        10000
        ** (torch.arange(1, embedding_dim, 2, device=device).float() / embedding_dim)
    )
    emb = emb[:, None]
    sin = torch.sin(emb * t).T
    cos = torch.cos(emb * t).T
    emb = torch.cat((sin, cos), dim=1)

    return emb
