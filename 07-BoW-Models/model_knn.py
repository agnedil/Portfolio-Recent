"""kNN BoW classifier: CV, grid search, test eval, save."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from utils import prepare_data, run_and_report

MODEL_NAME = "knn"


def build_pipeline() -> Pipeline:
    """CountVectorizer (char n-grams) + kNN."""
    vect = CountVectorizer(max_df=1.0, min_df=5, analyzer="char",
                           ngram_range=(1, 5), binary=True)
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance",
                               metric="minkowski", p=2, n_jobs=-1)
    return Pipeline([("vect", vect), ("clf", clf)])


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_pipeline()
    param_grid = {
        "vect__ngram_range": [(1, 3), (1, 5)],
        "clf__n_neighbors": [3, 5, 9],
        "clf__weights": ["uniform", "distance"],
    }
    run_and_report(MODEL_NAME, pipe, param_grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
