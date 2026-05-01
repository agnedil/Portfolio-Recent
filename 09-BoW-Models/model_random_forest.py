"""Random Forest BoW classifier: CV, grid search, test eval, save."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from utils import RANDOM_STATE, prepare_data, run_and_report

MODEL_NAME = "random_forest"


def build_pipeline() -> Pipeline:
    """CountVectorizer (word n-grams) + Random Forest."""
    vect = CountVectorizer(max_df=1.0, min_df=2, analyzer="word",
                           ngram_range=(1, 3), binary=True)
    clf = RandomForestClassifier(n_estimators=100, criterion="gini",
                                 class_weight="balanced",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("vect", vect), ("clf", clf)])


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_pipeline()
    param_grid = {
        "vect__ngram_range": [(1, 2), (1, 3)],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 20],
    }
    run_and_report(MODEL_NAME, pipe, param_grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
