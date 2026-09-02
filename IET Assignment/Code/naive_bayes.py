"""
naive_bayes.py
Gaussian Naive Bayes implemented from first principles (NumPy only).

Model:
    P(y | x_1..x_d) proportional to P(y) * prod_i P(x_i | y)
    with P(x_i | y) ~ Normal(mu_{i,y}, sigma_{i,y}^2)   (conditional independence
    assumption given the class -- the "naive" assumption)

We work in log-space for numerical stability:
    log P(y|x) = log P(y) + sum_i log N(x_i; mu_{i,y}, sigma_{i,y}^2) + const
"""
import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-6):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.mean_ = np.zeros((len(self.classes_), n_features))
        self.var_ = np.zeros((len(self.classes_), n_features))
        self.priors_ = np.zeros(len(self.classes_))
        global_var = X.var(axis=0)
        eps = self.var_smoothing * global_var.max()
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.mean_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + eps
            self.priors_[i] = Xc.shape[0] / X.shape[0]
        return self

    def _log_gaussian(self, X, mean, var):
        # log N(x; mu, sigma^2) for every feature, summed over features
        return -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(
            ((X - mean) ** 2) / var, axis=1
        )

    def joint_log_likelihood(self, X):
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            jll[:, i] = np.log(self.priors_[i]) + self._log_gaussian(
                X, self.mean_[i], self.var_[i]
            )
        return jll

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)
        # softmax-style normalisation in log-space for numerical stability
        max_jll = jll.max(axis=1, keepdims=True)
        exp_jll = np.exp(jll - max_jll)
        probs = exp_jll / exp_jll.sum(axis=1, keepdims=True)
        return probs  # columns correspond to self.classes_

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba_positive(self, X):
        """Probability of the positive class (Outcome == 1)."""
        probs = self.predict_proba(X)
        pos_idx = list(self.classes_).index(1)
        return probs[:, pos_idx]

    def manual_posterior_derivation(self, x, feature_names=None):
        """Return a step-by-step dictionary showing the manual computation
        of the posterior for a single patient record x, for reporting."""
        steps = {"priors": {}, "likelihoods": {}, "unnormalised": {}, "posterior": {}}
        for i, c in enumerate(self.classes_):
            steps["priors"][int(c)] = float(self.priors_[i])
            per_feature = []
            log_lik = 0.0
            for f in range(len(x)):
                mu, var = self.mean_[i, f], self.var_[i, f]
                dens = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((x[f] - mu) ** 2) / (2 * var))
                name = feature_names[f] if feature_names else f"x{f}"
                per_feature.append((name, float(x[f]), float(mu), float(var), float(dens)))
                log_lik += np.log(dens + 1e-300)
            steps["likelihoods"][int(c)] = per_feature
            steps["unnormalised"][int(c)] = float(np.exp(np.log(self.priors_[i]) + log_lik))
        total = sum(steps["unnormalised"].values())
        for c in steps["unnormalised"]:
            steps["posterior"][c] = steps["unnormalised"][c] / total if total > 0 else float("nan")
        return steps
