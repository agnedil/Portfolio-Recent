"""Shared helpers for BoW multi-class classifier scripts."""
from __future__ import annotations

import itertools
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

# ---------- Config ---------------------------------------------------------

RANDOM_STATE = 47
DATA_PATH = os.path.join("data", "file_name.pkl")
STOPWORDS_PATH = os.path.join("data", "stopwords_no_lemmas.txt")
RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

LABEL_TO_KEY: Dict[str, int] = {f"class{i}": i for i in range(6)}
KEY_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_KEY.items()}
ML_CATEGORIES: List[str] = list(LABEL_TO_KEY.keys())
LABELS: List[str] = [KEY_TO_LABEL[k] for k in sorted(KEY_TO_LABEL)]


# ---------- Data loading / prep -------------------------------------------

def load_stopwords(path: str = STOPWORDS_PATH) -> List[str]:
    """Load newline-separated stopwords; return [] if file missing."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_dataframe(path: str = DATA_PATH) -> pd.DataFrame:
    """Load pickled DataFrame and filter to known ML categories."""
    df = pd.read_pickle(path)
    df = df[df["label"].isin(ML_CATEGORIES)].copy()
    df["target"] = df["label"].map(LABEL_TO_KEY).astype(int)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates on (sentence, label)."""
    return df.drop_duplicates(subset=["sentence", "label"], keep="first").reset_index(drop=True)


def upsample_all(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Upsample each class to the majority-class size (with replacement) and shuffle the result."""
    labels = df["target"].unique()
    max_len = df["target"].value_counts().max()
    parts = []
    for lbl in labels:
        sub = df[df["target"] == lbl]
        if len(sub) < max_len:
            extra = sub.sample(max_len - len(sub), replace=True, random_state=random_state)
            sub = pd.concat([sub, extra])
        parts.append(sub)
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def get_xy(
    df: pd.DataFrame,
    feature_col: str = "sentence",
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split df by the `subset` column and return (X_train, X_test, y_train, y_test).

    The train split is upsampled to balance classes; the test split is left untouched.
    Both splits are shuffled with `random_state`.
    """
    df_train = df[df["subset"] == "train"].copy()
    df_test = df[df["subset"] == "test"].copy()
    df_train = upsample_all(df_train, random_state)

    X_train = df_train[feature_col].values
    y_train = df_train["target"].values
    X_test = df_test[feature_col].values
    y_test = df_test["target"].values

    X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = sklearn.utils.shuffle(X_test, y_test, random_state=random_state)
    return X_train, X_test, y_train, y_test


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-shot data prep: load → deduplicate → split by `subset` → upsample train only."""
    df = deduplicate(load_dataframe())
    return get_xy(df)


# ---------- CV / Grid search ----------------------------------------------

def kfold_cv(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Stratified k-fold CV on a fixed pipeline; returns mean accuracy/precision/recall/F1.

    Reporting-only: this evaluates the pipeline as-passed (no tuning). Model selection
    is done separately by `grid_search`, which runs its own CV over `param_grid`.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }
    results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return {metric: float(np.mean(results[f"test_{metric}"])) for metric in scoring}


def grid_search(
    pipeline: Pipeline,
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    random_state: int = RANDOM_STATE,
) -> GridSearchCV:
    """Grid-search `pipeline` over `param_grid` with stratified k-fold CV (scoring=macro F1).

    This is where model selection happens: the configuration with the highest mean CV
    F1 is chosen and refit on the full training set (`gs.best_estimator_`). Include
    the pipeline's default hyperparameters as points in `param_grid` (Option B) so the
    default competes against tuned candidates in the same CV comparison.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_macro",
        cv=skf,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X, y)
    return gs


# ---------- Metrics / plotting / IO ---------------------------------------

def test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy + macro precision/recall/f1 on the test set."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    out_path: str,
    title: str = "Confusion matrix",
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """Render a confusion-matrix heatmap to `out_path`."""
    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.PuBu)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.05)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"), ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def ensure_dirs() -> None:
    """Create results/ and results/models/ if missing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model: Any, name: str) -> str:
    """Pickle `model` to results/models/<name>.pkl and return the path."""
    ensure_dirs()
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def save_text(text: str, path: str) -> None:
    """Write `text` to `path` (parent dirs created)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def save_json(obj: Dict[str, Any], path: str) -> None:
    """Write `obj` to `path` as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def run_and_report(
    model_name: str,
    pipeline: Pipeline,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Run baseline CV + grid-search tuning, evaluate on test, persist artifacts, return summary.

    Two CV passes are done for different purposes:
      1. `kfold_cv(pipeline, ...)` — REPORTING ONLY. Scores the untuned `pipeline`
         (as returned by each model's `build_pipeline()`) to give a baseline. Its
         results are written to the summary as `cv_means` but do NOT drive selection.
      2. `grid_search(pipeline, param_grid, ...)` — SELECTION. Runs its own stratified
         5-fold CV over every combination in `param_grid` and refits the best configuration
         on the full train set. This refit estimator is what is tested and pickled.

    The model scripts include their pipeline defaults as points in `param_grid` (Option B),
    so the default configuration competes against tuned candidates inside the grid search.
    """
    ensure_dirs()

    # 1) Baseline k-fold CV on the untuned pipeline — reporting only, not used for selection.
    cv = kfold_cv(pipeline, X_train, y_train)

    # 2) Model selection: grid search picks the best hyperparameters by CV macro-F1
    #    and refits on the full train set. `best_model` is what gets saved.
    gs = grid_search(pipeline, param_grid, X_train, y_train)
    best_model = gs.best_estimator_

    # 3) evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    metrics = test_metrics(y_test, y_pred)

    # 4) persist artifacts
    save_text(report, os.path.join(RESULTS_DIR, f"{model_name}_classification_report.txt"))
    np.savetxt(os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.csv"), cm,
               fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, LABELS,
                          os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png"),
                          title=f"{model_name} confusion matrix")
    save_model(best_model, model_name)

    summary = {
        "model": model_name,
        "cv_means": cv,
        "best_params": gs.best_params_,
        "cv_best_f1": float(gs.best_score_),
        "test_metrics": metrics,
    }
    save_json(summary, os.path.join(RESULTS_DIR, f"{model_name}_summary.json"))

    print(f"\n=== {model_name} ===")
    print("CV means:", cv)
    print("Best params:", gs.best_params_)
    print("Test metrics:", metrics)
    print(report)
    return summary
