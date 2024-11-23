import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from utils_model import get_V, discretize_y

def compute_K(X, k):
    D = squareform(pdist(X.copy()))
    sig = np.sort(D, axis=1)[:, (k+1)]
    sig_mat = np.outer(sig, sig)
    K = np.exp(-(D**2) / sig_mat)
    return K

def plot_eigs(X, y, num_evrs=4, with_borders=True, to_color=True):
    ys = np.argsort(y)
    X = X[ys]
    y = y[ys]
    W = compute_K(X, 2)
    V = get_V(W, num_evrs)

    subplot_width = 8
    subplot_height = 4
    font_size = 20
    alpha = 0.8
    s = 15
    start_idx_title = 1
    title_font_size_inc = 10

    fig, ax = plt.subplots(2, 2, figsize=(subplot_width, subplot_height))

    for i, ax in enumerate(ax.flatten()):
        v = V[:, i]
        yq = discretize_y(v, 2).astype(int)
        c = yq if to_color else None
 
        _ = ax.set_title(f"$\mathbf{{v}}_{i + start_idx_title}$", fontsize=font_size + title_font_size_inc)
        _ = ax.scatter(np.arange(yq.size), v, c=c, cmap="tab10", s=s, alpha=alpha)

        ax.set_xlabel("Samples", fontsize=font_size)
        ax.set_ylabel("Values", fontsize=font_size)

        ax.set_xticklabels([]) 
        ax.set_yticklabels([])


        if with_borders:
            borders = np.where(np.diff(y))[0]
            for border in borders:
                _ = ax.axvline(x=border, color="black", linestyle="--", alpha=1)   

    plt.tight_layout()
