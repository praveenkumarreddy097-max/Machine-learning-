"""
data.py
Loading, cleaning and preprocessing of the Pima Indians Diabetes dataset.
No scikit-learn is used for the core logic; only NumPy/Pandas.
"""
import numpy as np
import pandas as pd

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# In this dataset a physiological reading of 0 is not a real value,
# it denotes a missing measurement.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically impossible zeros with NaN, then median-impute
    per class so imputation does not leak label information across folds
    is handled at the fold level in evaluate.py. Here we do a simple global
    median impute for exploratory use only."""
    df = df.copy()
    for col in ZERO_AS_MISSING:
        df.loc[df[col] == 0, col] = np.nan
    for col in ZERO_AS_MISSING:
        df[col] = df[col].fillna(df[col].median())
    return df


def impute_train_apply_test(train: pd.DataFrame, test: pd.DataFrame):
    """Fit medians on the training fold only, apply to both train and test.
    Prevents test-set information leaking into imputation (data leakage)."""
    train = train.copy()
    test = test.copy()
    for col in ZERO_AS_MISSING:
        train.loc[train[col] == 0, col] = np.nan
        test.loc[test[col] == 0, col] = np.nan
    medians = {}
    for col in ZERO_AS_MISSING:
        med = train[col].median()
        medians[col] = med
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)
    return train, test, medians


def standardize_train_apply_test(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def to_xy(df: pd.DataFrame):
    X = df.drop(columns=["Outcome"]).values.astype(float)
    y = df["Outcome"].values.astype(int)
    return X, y


def class_balance_report(y: np.ndarray) -> dict:
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    return {"n": n, "positive": pos, "negative": neg,
            "positive_rate": pos / n, "imbalance_ratio": neg / max(pos, 1)}


def oversample_minority(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Simple random oversampling of the minority class on the TRAINING
    fold only, to address class imbalance without any external library."""
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()
    X_list, y_list = [X], [y]
    for c, cnt in zip(classes, counts):
        if cnt < majority_count:
            idx = np.where(y == c)[0]
            n_needed = majority_count - cnt
            extra_idx = rng.choice(idx, size=n_needed, replace=True)
            X_list.append(X[extra_idx])
            y_list.append(y[extra_idx])
    X_bal = np.vstack(X_list)
    y_bal = np.concatenate(y_list)
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]
