"""SVM multi-class BoW classifier: CV, grid search, test eval, save."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from utils import RANDOM_STATE, prepare_data, run_and_report

MODEL_NAME = "svm"


def build_pipeline() -> Pipeline:
    """TF-IDF (word n-grams) + linear SVC."""
    vect = TfidfVectorizer(max_df=1.0, min_df=2, analyzer="word",
                           ngram_range=(1, 2), binary=False)
    clf = SVC(C=1.0, kernel="linear", decision_function_shape="ovr",
              random_state=RANDOM_STATE)
    return Pipeline([("vect", vect), ("clf", clf)])


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_pipeline()
    param_grid = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0],
        "clf__kernel": ["linear", "rbf"],
    }
    run_and_report(MODEL_NAME, pipe, param_grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
