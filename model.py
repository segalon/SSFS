import numpy as np
from sklearn.model_selection import ShuffleSplit
from config_models import CLFS
from utils_model import get_model, get_fs_scores, discretize_y, \
                        get_V, compute_W


class SSFS:
    def __init__(self,
                 W,
                 num_clusters,
                 max_evrs=None,
                 selection="adaptive_vars",
                 prop_sample=0.95,
                 num_resamples=500,
                 model_selection="linear_clf",
                 s_type_selection="coef",
                 model_fs="xgboost_clf",
                 s_type_fs="gain"):
        """
        :param W: the adjacency matrix
        :param num_clusters: the number of eigenvectors to select for feature selection (after the selection phase),
                           typically this is the number of distinct categories searched for in the data,
                           but in some cases it might be preferable to choose a different number.
        :param: max_evrs: the maximum number of eigenvectors to consider select from. I
                            If None, then:
                                if selection="top_evrs", uses the top num_clusters eigenvectors.
                                if selection="adaptive_vars", uses the top 2*num_clusters eigenvectors.
        :param selection: the eigenvector selection method, "adaptive_vars" by model variance, or "top_evrs"
                            to select the top num_clusters eigenvectors
        :param model_selection: the name of the surrogate model to use for the variance estimation
                                (see config for the available models)
        :param s_type_selection: the feature score type for the eigenvector selection surrogate model,
                            e.g "coef" for linear models, "gain" for XGBoost, see get_model
        :param model_fs: the name of the surrogate model to use for the final feature scores, not for
                         the variance estimation, if None, use the same model as the one used
                          for the variance estimation
        :param s_type_fs: the importance type of the surrogate model, if None, use the same importance type as the
            one used for the variance estimation,
        """

        self.W = W
        self.n_clusters = num_clusters
        self.max_evrs = max_evrs
        self.selection = selection  # will be modified in the feature selection
        self.prop_sample = prop_sample
        self.num_resamples = num_resamples
        self.s_type_selection = s_type_selection
        self.model_selection = model_selection

        if max_evrs is None:
            if selection == "top_evrs":
                self.max_evrs = num_clusters
            else:
                self.max_evrs = 2 * num_clusters

        if model_fs is None:
            self.model_fs = self.model_selection
            self.s_type_fs = self.s_type_selection
        else:
            self.model_fs = model_fs
            self.s_type_fs = s_type_fs

        self.S = []
        self.V = None
        # A is a tensor for storing the feature scores for each split and eigenvector
        # (num_resamples, n_features, max_evrs)
        self.A = None
        self.idx_sel_eig = None

    def fit(self, X):
        if self.W is None:
            print("W is None, computing it")
            self.W = compute_W(X, k=2)

        self.max_evrs = min(self.max_evrs, X.shape[0] - 1)
        self.V = get_V(self.W, self.max_evrs)
        self.A = np.zeros((self.num_resamples, X.shape[1], self.max_evrs))

        S = []
        for i in range(self.max_evrs):
            model_class, args_model = get_model(self.model_selection)

            if model_class in CLFS:
                y_psdu = discretize_y(self.V[:, i], 2)
            else:
                y_psdu = self.V[:, i]

            if "adaptive" in self.selection:
                samples_scores = self.get_scores_resamples(X, y_psdu)
                self.A[:, :, i] = samples_scores

            model_class, args_model = get_model(self.model_fs)
            imp_type = self.s_type_fs

            model = model_class(**args_model).fit(X, y_psdu)
            s = get_fs_scores(model,
                              num_orig_features=X.shape[1],
                              imp_type=imp_type)
            S.append(s)

        self.S = np.array(S)
        return self

    def get_scores(self):
        if self.selection == "adaptive_vars":
            coef_vars = np.var(self.A[:, :, :self.max_evrs], axis=0)
            sum_coef_vars = np.sum(coef_vars, axis=0)
            idx_eig = np.argsort(sum_coef_vars)[:self.n_clusters]

        elif self.selection == "top_evrs":
            idx_eig = np.arange(self.max_evrs)
        else:
            raise ValueError("selection not recognized")

        self.idx_sel_eig = idx_eig

        print("[+] selected eigenvectors:", idx_eig)
        s = np.max(self.S[idx_eig, :], axis=0)
        return s

    def feature_ranking(self):
        """
        returns the indices of the features in descending order of importance
        """
        s = self.get_scores()
        idx = np.argsort(s)[::-1]
        return idx

    def get_scores_resamples(self,
                             X,
                             y_psdu):

        scores_samples = np.zeros((self.num_resamples, X.shape[1]))
        split_obj = ShuffleSplit(n_splits=self.num_resamples,
                                 test_size=1-self.prop_sample,
                                 random_state=0)

        for i, (train_index, _) in enumerate(split_obj.split(X, y_psdu)):
            model_class, args_model = get_model(self.model_selection)
            model = model_class(**args_model).fit(X[train_index], y_psdu[train_index])

            fs_scores = get_fs_scores(model, num_orig_features=X.shape[1], imp_type=self.s_type_selection)
            scores_samples[i, :] = fs_scores / np.sum(np.abs(fs_scores))

        return scores_samples
