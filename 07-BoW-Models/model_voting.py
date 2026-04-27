"""Voting (soft) ensemble over LR(word), LR(char), SVM, XGBoost."""
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils import RANDOM_STATE, prepare_data, run_and_report

MODEL_NAME = "voting"


def _build_base_estimators():
    """Construct the four heterogeneous base pipelines used by the voter."""
    lr_word = Pipeline([
        ("vect", CountVectorizer(max_df=1.0, min_df=2, analyzer="word",
                                 ngram_range=(1, 2), binary=True)),
        ("clf", LogisticRegression(solver="liblinear", penalty="l1",
                                   max_iter=500, random_state=RANDOM_STATE)),
    ])
    lr_char = Pipeline([
        ("vect", CountVectorizer(max_df=1.0, min_df=2, analyzer="char",
                                 ngram_range=(1, 8), binary=False)),
        ("clf", LogisticRegression(solver="liblinear", penalty="l2",
                                   max_iter=500, random_state=RANDOM_STATE)),
    ])
    svm = Pipeline([
        ("vect", TfidfVectorizer(max_df=1.0, min_df=2, analyzer="word",
                                 ngram_range=(1, 2))),
        ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
    ])
    xgb = Pipeline([
        ("vect", CountVectorizer(max_df=1.0, min_df=5, analyzer="char",
                                 ngram_range=(1, 8), binary=True)),
        ("clf", XGBClassifier(n_estimators=150, learning_rate=0.3,
                              objective="multi:softmax", eval_metric="mlogloss",
                              use_label_encoder=False,
                              random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    return [("lr", lr_word), ("lr2", lr_char), ("svm", svm), ("xgb", xgb)]


def build_pipeline() -> VotingClassifier:
    """Soft-voting ensemble with equal weights across base estimators."""
    return VotingClassifier(estimators=_build_base_estimators(),
                            voting="soft", weights=[0.25, 0.25, 0.25, 0.25],
                            flatten_transform=True, n_jobs=-1)


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_pipeline()
    # tune only voting strategy + weights (hyperparam search on base estimators
    # would be prohibitively expensive inside nested CV)
    param_grid = {
        "voting": ["soft", "hard"],
        "weights": [[0.25, 0.25, 0.25, 0.25], [0.2, 0.2, 0.3, 0.3]],
    }
    run_and_report(MODEL_NAME, pipe, param_grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
