import torch
from sklearn.decomposition import TruncatedSVD
import os
import numpy as np

def update_buffer_svd(buffer, new_embed, window_size=4, out_dim=64):
    new_embed = new_embed.unsqueeze(0)  
    buffer = torch.cat([buffer, new_embed], 0)
    if buffer.size(0) > window_size:
        buffer = buffer[-window_size:] 

    if buffer.size(0) >= 2:
        buffer_np = buffer.cpu().numpy()
        svd = TruncatedSVD(n_components=min(out_dim, buffer_np.shape[0]), random_state=42)
        cache = svd.fit_transform(buffer_np)
        cache = torch.from_numpy(cache)
    else:
        cache = torch.zeros(4, 32).cuda()

    return buffer, cache


def random_projection(embed, out_dim=64, file="random_projection.npy", seed=42):

    if embed.ndim == 1:
        single = True
        embed = embed.unsqueeze(0)
    else:
        single = False

    in_dim = embed.shape[1]

    if os.path.exists(file):
        R = np.load(file)
    else:
        np.random.seed(seed)
        R = np.random.randn(in_dim, out_dim) / np.sqrt(out_dim)
        np.save(file, R)

    embed_np = embed.cpu().numpy()
    embed_reduced = embed_np @ R
    embed_reduced = torch.from_numpy(embed_reduced).to(embed.device)

    if single:
        return embed_reduced[0]
    return embed_reduced