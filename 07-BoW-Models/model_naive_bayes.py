"""Multinomial Naive Bayes BoW classifier: CV, grid search, test eval, save."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils import prepare_data, run_and_report

MODEL_NAME = "naive_bayes"


def build_pipeline() -> Pipeline:
    """CountVectorizer (char n-grams) + Multinomial NB."""
    vect = CountVectorizer(max_df=1.0, min_df=4, analyzer="char",
                           ngram_range=(1, 8), binary=False)
    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    return Pipeline([("vect", vect), ("clf", clf)])


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_pipeline()
    param_grid = {
        "vect__ngram_range": [(1, 6), (1, 8)],
        "vect__min_df": [2, 4],
        "clf__alpha": [0.1, 0.5, 1.0],
    }
    run_and_report(MODEL_NAME, pipe, param_grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
