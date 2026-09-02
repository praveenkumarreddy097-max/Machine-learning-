"""
evaluate.py
Metrics (accuracy, precision, recall, F1, ROC-AUC) and k-fold
cross-validation implemented from first principles (NumPy only).
Also implements a paired t-test for statistical significance testing,
using scipy only as an independent cross-check, never as core logic
(the manual computation is included alongside it).
"""
import numpy as np


def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_auc(y_true, scores):
    """Rank-based ROC-AUC (Mann-Whitney U statistic), no sklearn."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = np.mean(ranks[order[i:j + 1]])
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = np.sum(ranks[y_true == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def all_metrics(y_true, y_pred, scores):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc(y_true, scores),
    }


def k_fold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx


def paired_t_test(diffs: np.ndarray):
    """Manual paired t-test on a vector of per-fold differences
    (model_A_metric - model_B_metric). Returns (t_stat, dof)."""
    n = len(diffs)
    mean_d = diffs.mean()
    sd_d = diffs.std(ddof=1)
    if sd_d == 0:
        return float("inf") if mean_d != 0 else 0.0, n - 1
    t_stat = mean_d / (sd_d / np.sqrt(n))
    return float(t_stat), n - 1


def t_to_p_two_sided(t_stat, dof):
    """p-value via scipy as an independent cross-check of the manual
    t-statistic above (scipy is NOT used to compute the t-statistic itself)."""
    from scipy import stats
    return float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
