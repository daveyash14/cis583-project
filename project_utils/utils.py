import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def visualize_latent_space(latent_representations, epoch):
    flattened_latent_representations = np.concatenate(latent_representations, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    tsne_results = tsne.fit_transform(flattened_latent_representations)

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title(f"t-SNE Visualization of Latent Space (Epoch {epoch})")
    plt.show()