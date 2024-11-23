
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn_extra.cluster import KMedoids

from config_models import MODELS_DICT, XGBOOST_IMP_TYPES


def discretize_y(y, n_clusters=2):
    y = y.copy().reshape(-1, 1)
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    kmedoids.fit(y)
    yq = kmedoids.labels_
    return yq


def get_V(W, num_evrs):
    W = (W + W.T) / 2
    W_norm = np.diag(np.sqrt(1 / W.sum(1)))
    W = np.dot(W_norm, np.dot(W, W_norm))
    WT = W.T
    W[W < WT] = WT[W < WT]
    eigen_value, ul = scipy.linalg.eigh(a=W)
    V = np.dot(W_norm, ul[:, -1 * num_evrs - 1:-1])
    V = V[:, ::-1]
    V = V - np.mean(V, axis=0)
    return V


def get_fs_scores(model, num_orig_features, imp_type):
    if imp_type == "coef":
        if model.coef_.shape[0] == 1 or len(model.coef_.shape) == 1:
            fs_scores = np.abs(model.coef_.reshape(-1))
        else:
            fs_scores = np.linalg.norm(model.coef_, axis=0)
    elif imp_type == "feature_importances":
        fs_scores = np.abs(model.feature_importances_)
    elif imp_type in XGBOOST_IMP_TYPES:
        booster = model.get_booster()
        importance = booster.get_score(importance_type=imp_type)
        fs_scores = [0.0] * num_orig_features
        for f, importance in importance.items():
            idx = int(f[1:])
            fs_scores[idx] = importance
        fs_scores = np.abs(fs_scores)
    else:
        raise Exception("invalid importance type")

    return fs_scores


def get_model(model_name, args_model=None):
    if args_model is None:
        args_model = {}
    if model_name in MODELS_DICT:
        model_class = MODELS_DICT[model_name]["class"]
        args_model.update(MODELS_DICT[model_name]["args"])
        return model_class, args_model
    else:
        raise ValueError("invalid model name")


def compute_W(X, k):
    D = squareform(pdist(X.copy()))
    sig = np.sort(D, axis=1)[:, (k + 1)]
    sig_mat = np.outer(sig, sig)
    W = np.exp(-(D ** 2) / sig_mat)
    return W
