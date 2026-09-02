"""
run_experiments.py
End-to-end, no-manual-intervention pipeline:
  1. Load + clean Pima Indians Diabetes data
  2. k-fold CV over 3 configurations: MLP-only, MLP+GA, Fused (MLP+GA + NB)
  3. Compute accuracy/precision/recall/F1/ROC-AUC per fold
  4. Statistical significance testing (paired t-test) between configurations
  5. Manual Naive-Bayes posterior derivation for one test patient
  6. PAC / sample-complexity analysis
  7. Save all logs, tables and plots into /results
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import (load_raw, impute_train_apply_test, standardize_train_apply_test,
                       to_xy, class_balance_report, oversample_minority, COLUMNS)
from src.mlp import MLP
from src.ga import GeneticWeightSearch
from src.naive_bayes import GaussianNaiveBayes
from src.fusion import bma_weights, fuse_probabilities
from src.evaluate import all_metrics, k_fold_indices, paired_t_test, t_to_p_two_sided
from src.pac_theory import finite_hypothesis_bound, vc_dim_estimate_mlp, vc_bound_sample_complexity

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SEED = 42
K = 5

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def run():
    rng = np.random.default_rng(SEED)
    df = load_raw(os.path.join(os.path.dirname(__file__), "data", "pima.csv"))
    log(f"Loaded dataset with shape {df.shape}")

    balance = class_balance_report(df["Outcome"].values)
    log(f"Class balance: {balance}")

    X_full = df.drop(columns=["Outcome"]).values.astype(float)
    y_full = df["Outcome"].values.astype(int)

    fold_records = []
    per_fold_configs = {"mlp_only": [], "mlp_ga": [], "fused": []}
    ga_convergence_example = None
    mlp_only_loss_curve = None
    mlp_ga_test_scores_for_roc = None
    y_test_for_roc = None

    for fold_i, (train_idx, test_idx) in enumerate(k_fold_indices(len(df), K, seed=SEED)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_df, test_df, medians = impute_train_apply_test(train_df, test_df)
        X_train, y_train = to_xy(train_df)
        X_test, y_test = to_xy(test_df)

        X_train, X_test, mu, sigma = standardize_train_apply_test(X_train, X_test)

        # Address class imbalance via training-fold-only oversampling
        X_train_bal, y_train_bal = oversample_minority(X_train, y_train, rng)

        # Split a validation slice out of the (balanced) training fold for the GA
        n_val = max(20, int(0.2 * len(X_train_bal)))
        val_idx = rng.choice(len(X_train_bal), size=n_val, replace=False)
        mask = np.ones(len(X_train_bal), dtype=bool)
        mask[val_idx] = False
        X_tr_sub, y_tr_sub = X_train_bal[mask], y_train_bal[mask]
        X_val_sub, y_val_sub = X_train_bal[val_idx], y_train_bal[val_idx]

        # ---------- Configuration 1: MLP alone (random init) ----------
        mlp_plain = MLP(n_input=X_train.shape[1], n_hidden=8, seed=fold_i)
        history = mlp_plain.fit(X_train_bal, y_train_bal, epochs=150, lr=0.05,
                                 batch_size=32, l2=1e-3, seed=fold_i, track_history=True)
        if fold_i == 0:
            mlp_only_loss_curve = history
        proba_mlp_only = mlp_plain.predict_proba(X_test)
        pred_mlp_only = (proba_mlp_only >= 0.5).astype(int)
        m1 = all_metrics(y_test, pred_mlp_only, proba_mlp_only)
        per_fold_configs["mlp_only"].append(m1)

        # ---------- Configuration 2: MLP + GA-optimised init/hyperparams ----------
        ga = GeneticWeightSearch(n_input=X_train.shape[1], pop_size=20,
                                  generations=20, seed=fold_i)
        best_ind, best_fit = ga.run(X_tr_sub, y_tr_sub, X_val_sub, y_val_sub)
        if fold_i == 0:
            ga_convergence_example = (ga.history_best, ga.history_mean)
        mlp_ga, lr_star, n_hidden_star = ga.build_best_mlp(best_ind)
        mlp_ga.fit(X_train_bal, y_train_bal, epochs=150, lr=lr_star,
                   batch_size=32, l2=1e-3, seed=fold_i)
        proba_mlp_ga = mlp_ga.predict_proba(X_test)
        pred_mlp_ga = (proba_mlp_ga >= 0.5).astype(int)
        m2 = all_metrics(y_test, pred_mlp_ga, proba_mlp_ga)
        m2["n_hidden"] = n_hidden_star
        m2["lr"] = lr_star
        per_fold_configs["mlp_ga"].append(m2)

        if fold_i == 0:
            mlp_ga_test_scores_for_roc = {"mlp_only": proba_mlp_only, "mlp_ga": proba_mlp_ga}
            y_test_for_roc = y_test.copy()

        # ---------- Naive Bayes (independent baseline) ----------
        nb = GaussianNaiveBayes().fit(X_train_bal, y_train_bal)
        proba_nb = nb.predict_proba_positive(X_test)

        # ---------- Fusion (Bayesian Model Averaging by validation accuracy) ----------
        val_pred_mlp_ga = mlp_ga.predict(X_val_sub)
        acc_mlp_ga_val = float(np.mean(val_pred_mlp_ga == y_val_sub))
        val_pred_nb = nb.predict(X_val_sub)
        acc_nb_val = float(np.mean(val_pred_nb == y_val_sub))
        w_mlp, w_nb = bma_weights(acc_mlp_ga_val, acc_nb_val)

        proba_fused = fuse_probabilities(proba_mlp_ga, proba_nb, w_mlp, w_nb)
        pred_fused = (proba_fused >= 0.5).astype(int)
        m3 = all_metrics(y_test, pred_fused, proba_fused)
        m3["w_mlp"] = w_mlp
        m3["w_nb"] = w_nb
        per_fold_configs["fused"].append(m3)

        if fold_i == 0:
            mlp_ga_test_scores_for_roc["fused"] = proba_fused
            mlp_ga_test_scores_for_roc["nb"] = proba_nb

        log(f"Fold {fold_i}: MLP acc={m1['accuracy']:.3f} | MLP+GA acc={m2['accuracy']:.3f} "
            f"(h={n_hidden_star}, lr={lr_star}) | Fused acc={m3['accuracy']:.3f} "
            f"(w_mlp={w_mlp:.2f}, w_nb={w_nb:.2f})")

        fold_records.append({
            "fold": fold_i, "mlp_only": m1, "mlp_ga": m2, "fused": m3
        })

    # ---------------- Aggregate results ----------------
    summary = {}
    for cfg in ["mlp_only", "mlp_ga", "fused"]:
        arr = per_fold_configs[cfg]
        summary[cfg] = {}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            vals = np.array([f[metric] for f in arr])
            summary[cfg][metric] = {"mean": float(vals.mean()), "std": float(vals.std())}

    # Statistical significance: MLP+GA vs MLP-only, and Fused vs MLP+GA
    def diffs(metric, cfg_a, cfg_b):
        a = np.array([f[metric] for f in per_fold_configs[cfg_a]])
        b = np.array([f[metric] for f in per_fold_configs[cfg_b]])
        return a - b

    sig_tests = {}
    for (cfg_a, cfg_b, label) in [
        ("mlp_ga", "mlp_only", "MLP+GA_vs_MLP"),
        ("fused", "mlp_ga", "Fused_vs_MLP+GA"),
        ("fused", "mlp_only", "Fused_vs_MLP"),
    ]:
        d = diffs("accuracy", cfg_a, cfg_b)
        t_stat, dof = paired_t_test(d)
        p_val = t_to_p_two_sided(t_stat, dof) if np.isfinite(t_stat) else 0.0
        sig_tests[label] = {"mean_diff_accuracy": float(d.mean()), "t_stat": t_stat,
                             "dof": dof, "p_value": p_val}

    log(f"Significance tests: {json.dumps(sig_tests, indent=2)}")

    # ---------------- Manual Naive Bayes posterior derivation ----------------
    # Refit NB on the full (imputed, standardised) dataset for a clean, reproducible
    # illustrative derivation independent of any single CV fold.
    df_clean, df_clean_test, _ = impute_train_apply_test(df, df.iloc[[0]])
    X_all, y_all = to_xy(df_clean)
    X_all_std, _, _, _ = standardize_train_apply_test(X_all, X_all[:1])
    nb_full = GaussianNaiveBayes().fit(X_all_std, y_all)
    sample_patient_idx = 5
    x_sample = X_all_std[sample_patient_idx]
    y_sample_true = int(y_all[sample_patient_idx])
    feature_names = COLUMNS[:-1]
    manual_steps = nb_full.manual_posterior_derivation(x_sample, feature_names)
    manual_steps["true_label"] = y_sample_true
    manual_steps["patient_index"] = sample_patient_idx

    with open(os.path.join(RESULTS_DIR, "manual_nb_derivation.json"), "w") as f:
        json.dump(manual_steps, f, indent=2)

    # ---------------- PAC / sample complexity ----------------
    n_input = X_full.shape[1]
    n_hidden_typical = 8
    n_weights = MLP.n_params(n_input, n_hidden_typical)
    h_size_finite = 4 * 4  # |hidden choices| * |lr choices| combinatorial GA search grid
    m_finite = finite_hypothesis_bound(h_size_finite, epsilon=0.1, delta=0.05)
    vc_dim = vc_dim_estimate_mlp(n_weights, n_layers=2)
    m_vc = vc_bound_sample_complexity(vc_dim, epsilon=0.1, delta=0.05)
    pac_summary = {
        "n_weights_mlp": int(n_weights),
        "finite_hypothesis_grid_size": h_size_finite,
        "m_required_finite_bound_eps0.1_delta0.05": float(m_finite),
        "vc_dim_estimate": float(vc_dim),
        "m_required_vc_bound_eps0.1_delta0.05": float(m_vc),
        "dataset_size": int(len(df)),
    }
    log(f"PAC summary: {json.dumps(pac_summary, indent=2)}")

    # ---------------- Save all results ----------------
    with open(os.path.join(RESULTS_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "per_fold_results.json"), "w") as f:
        json.dump(fold_records, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "significance_tests.json"), "w") as f:
        json.dump(sig_tests, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "pac_summary.json"), "w") as f:
        json.dump(pac_summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "class_balance.json"), "w") as f:
        json.dump(balance, f, indent=2)

    # Results table as CSV
    rows = []
    for cfg in ["mlp_only", "mlp_ga", "fused"]:
        row = {"config": cfg}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            row[f"{metric}_mean"] = summary[cfg][metric]["mean"]
            row[f"{metric}_std"] = summary[cfg][metric]["std"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "results_table.csv"), index=False)

    # ---------------- Plots ----------------
    plt.rcParams.update({"font.size": 10})

    # 1. GA convergence
    plt.figure(figsize=(6, 4))
    gens = np.arange(1, len(ga_convergence_example[0]) + 1)
    plt.plot(gens, ga_convergence_example[0], label="Best fitness")
    plt.plot(gens, ga_convergence_example[1], label="Mean fitness", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (validation accuracy - complexity penalty)")
    plt.title("GA Convergence (Fold 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ga_convergence.png"), dpi=150)
    plt.close()

    # 2. MLP training loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(mlp_only_loss_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("MLP (random init) Training Loss (Fold 0)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mlp_loss_curve.png"), dpi=150)
    plt.close()

    # 3. Bar chart comparing metrics across configurations
    metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics_list))
    width = 0.25
    plt.figure(figsize=(8, 5))
    for i, cfg in enumerate(["mlp_only", "mlp_ga", "fused"]):
        means = [summary[cfg][m]["mean"] for m in metrics_list]
        stds = [summary[cfg][m]["std"] for m in metrics_list]
        plt.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
                label={"mlp_only": "MLP alone", "mlp_ga": "MLP + GA", "fused": "Fused"}[cfg])
    plt.xticks(x, [m.upper() for m in metrics_list])
    plt.ylabel("Score")
    plt.title(f"{K}-Fold Cross-Validation Metric Comparison")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metric_comparison.png"), dpi=150)
    plt.close()

    # 4. ROC curves (fold 0) for all 4 scorers
    def roc_curve_points(y_true, scores):
        thresholds = np.sort(np.unique(scores))[::-1]
        tprs, fprs = [1.0], [1.0]
        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)
        for th in thresholds:
            pred = (scores >= th).astype(int)
            tp = np.sum((y_true == 1) & (pred == 1))
            fp = np.sum((y_true == 0) & (pred == 1))
            tprs.append(tp / P if P > 0 else 0)
            fprs.append(fp / N if N > 0 else 0)
        tprs.append(0.0); fprs.append(0.0)
        return np.array(fprs), np.array(tprs)

    plt.figure(figsize=(6, 6))
    for name, scores in mlp_ga_test_scores_for_roc.items():
        fprs, tprs = roc_curve_points(y_test_for_roc, scores)
        order = np.argsort(fprs)
        plt.plot(fprs[order], tprs[order], label=name)
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves, Fold 0 Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # 5. Class balance pie/bar
    plt.figure(figsize=(5, 4))
    plt.bar(["No Diabetes (0)", "Diabetes (1)"], [balance["negative"], balance["positive"]],
            color=["0.6", "0.3"])
    plt.ylabel("Number of Patients")
    plt.title("Class Distribution, Pima Indians Diabetes Dataset")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "class_balance.png"), dpi=150)
    plt.close()

    log("All results and plots saved to /results")
    return summary, sig_tests, pac_summary, manual_steps, balance


if __name__ == "__main__":
    run()
