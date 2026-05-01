"""Logistic Regression multi-class BoW classifier: CV, grid search, test eval, save."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import RANDOM_STATE, prepare_data, run_and_report

MODEL_NAME = "logistic_regression"


def build_pipeline() -> Pipeline:
    """CountVectorizer (char n-grams) + Logistic Regression."""
    vect = CountVectorizer(max_df=1.0, min_df=2, analyzer="char",
                           ngram_range=(1, 8), binary=False)
    clf = LogisticRegression(solver="liblinear", penalty="l2", C=1.0,
                             max_iter=500, random_state=RANDOM_STATE)
    return Pipeline([("vect", vect), ("clf", clf)])


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_pipeline()
    param_grid = {
        "vect__ngram_range": [(1, 6), (1, 8)],
        "clf__C": [0.5, 1.0, 2.0],
        "clf__penalty": ["l1", "l2"],
    }
    run_and_report(MODEL_NAME, pipe, param_grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
