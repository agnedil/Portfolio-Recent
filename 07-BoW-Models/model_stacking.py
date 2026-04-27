"""Stacking ensemble: LR(word) + SVM base, LR final estimator."""
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from utils import RANDOM_STATE, prepare_data, run_and_report

MODEL_NAME = "stacking"


def _build_base_estimators():
    """LR (word n-grams) + SVM (tf-idf word) as stacking base learners."""
    lr = Pipeline([
        ("vect", CountVectorizer(max_df=1.0, min_df=2, analyzer="word",
                                 ngram_range=(1, 2), binary=True)),
        ("clf", LogisticRegression(solver="liblinear", penalty="l1",
                                   max_iter=500, random_state=RANDOM_STATE)),
    ])
    svm = Pipeline([
        ("vect", TfidfVectorizer(max_df=1.0, min_df=2, analyzer="word",
                                 ngram_range=(1, 2))),
        ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
    ])
    return [("lr", lr), ("svm", svm)]


def build_pipeline() -> StackingClassifier:
    """Stacking with LR meta-learner over LR+SVM base learners."""
    return StackingClassifier(
        estimators=_build_base_estimators(),
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        stack_method="auto", passthrough=False, n_jobs=-1,
    )


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_pipeline()
    # tune the meta-learner + whether to pass raw features through
    param_grid = {
        "final_estimator__C": [0.5, 1.0, 2.0],
        "passthrough": [False, True],
    }
    run_and_report(MODEL_NAME, pipe, param_grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
