"""XGBoost multi-class BoW classifier: CV, grid search, test eval, save."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from utils import RANDOM_STATE, prepare_data, run_and_report

MODEL_NAME = "xgboost"


def build_pipeline() -> Pipeline:
    """CountVectorizer (char n-grams) + XGBoost."""
    vect = CountVectorizer(max_df=1.0, min_df=5, analyzer="char",
                           ngram_range=(1, 8), binary=True)
    clf = XGBClassifier(
        n_estimators=150, learning_rate=0.3, objective="multi:softmax",
        eval_metric="mlogloss", booster="gbtree", tree_method="approx",
        subsample=0.75, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=1.0,
        gamma=0.25, random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False,
    )
    return Pipeline([("vect", vect), ("clf", clf)])


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_pipeline()
    # small grid to keep runtime reasonable
    param_grid = {
        "vect__ngram_range": [(1, 6), (1, 8)],
        "clf__n_estimators": [100, 150],
        "clf__learning_rate": [0.1, 0.3],
    }
    run_and_report(MODEL_NAME, pipe, param_grid, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
