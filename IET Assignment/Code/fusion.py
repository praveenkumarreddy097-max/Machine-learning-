"""
fusion.py
Decision-fusion mechanism combining the MLP(+GA) prediction with the
Naive Bayes prediction.

Method chosen: Bayesian Model Averaging with per-model weights derived
from each model's own validation accuracy (a plug-in estimate of the
posterior model probability under a uniform model prior):

    w_MLP = Acc_MLP / (Acc_MLP + Acc_NB)
    w_NB  = Acc_NB  / (Acc_MLP + Acc_NB)

    P_fused(y=1 | x) = w_MLP * P_MLP(y=1|x) + w_NB * P_NB(y=1|x)

Justification: under BMA, the posterior predictive distribution is a
weighted mixture of each model's predictive distribution, weighted by the
model's posterior probability, P(M_k | Data) ∝ P(Data | M_k) P(M_k). With a
uniform prior over the two models, validation accuracy is a simple, direct
proxy for P(Data | M_k) on held-out data, so normalising the two accuracies
to sum to 1 gives an approximate, empirically-grounded BMA weighting
without requiring a full marginal-likelihood computation (which is
intractable for the MLP in closed form).
"""
import numpy as np


def bma_weights(acc_mlp: float, acc_nb: float, eps=1e-9):
    total = acc_mlp + acc_nb + eps
    return acc_mlp / total, acc_nb / total


def fuse_probabilities(p_mlp: np.ndarray, p_nb: np.ndarray, w_mlp: float, w_nb: float):
    return w_mlp * p_mlp + w_nb * p_nb


def fused_predict(p_mlp, p_nb, w_mlp, w_nb, threshold=0.5):
    p_fused = fuse_probabilities(p_mlp, p_nb, w_mlp, w_nb)
    return (p_fused >= threshold).astype(int), p_fused
