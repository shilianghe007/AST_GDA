"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


def visualize(features: torch.Tensor, domains: torch.Tensor, filename: str):
    """
    Visualize features from different domains using t-SNE.

    Args:
        features (tensor): features in shape :math:`(minibatch, F)`
        domains (tensor): the domain ids of these features
        filename (str): the file name to save t-SNE

    """
    features = features.numpy()
    domains = domains.numpy()

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap(['r', 'g', 'b']), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
