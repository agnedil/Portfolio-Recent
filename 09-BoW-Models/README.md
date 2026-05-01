# BoW Multi-class Classifiers

End-to-end bag-of-words pipelines for multi-class text classification across 8 model families (XGBoost, SVM, Logistic Regression, Naive Bayes, Random Forest, kNN, Voting, Stacking). Each model script is self-contained: it loads the data, runs 5-fold cross-validation, performs grid-search hyperparameter tuning, evaluates on the held-out test set, and saves the best-F1 model to disk.

## Repo structure

```
.
├── utils.py                         # shared helpers (data loading, CV, grid search, plotting, IO)
├── model_xgboost.py                 # one script per model
├── model_svm.py
├── model_logistic_regression.py
├── model_naive_bayes.py
├── model_random_forest.py
├── model_knn.py
├── model_voting.py                  # soft-voting ensemble (LR×2, SVM, XGBoost)
├── model_stacking.py                # stacking ensemble (LR + SVM → LR meta)
├── orchestrator.py                  # runs all models and aggregates results
├── requirements.txt
├── report.md                        # combined report (written by orchestrator)
├── data/
│   ├── file_name.pkl                # pickled DataFrame (see Data format)
│   └── stopwords_no_lemmas.txt      # optional stopwords, one per line
└── results/                         # produced on run
    ├── models/                      # <model>.pkl binaries of best estimators
    ├── <model>_classification_report.txt
    ├── <model>_confusion_matrix.csv
    ├── <model>_confusion_matrix.png
    ├── <model>_summary.json         # CV means, best params, test metrics
    └── report.md                    # mirror of the top-level report.md
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

`utils.prepare_data()` expects a pickled pandas DataFrame at `data/file_name.pkl` with these columns:

| column     | type   | description                                                     |
|------------|--------|-----------------------------------------------------------------|
| `sentence` | str    | the text input                                                  |
| `label`    | str    | one of `class0`…`class5` (see `LABEL_TO_KEY` in `utils.py`)     |
| `subset`   | str    | `train` or `test`                                               |

`data/stopwords_no_lemmas.txt` is optional and unused by the default pipelines; if you wire it into a vectorizer via `utils.load_stopwords()`, one stopword per line is expected.

If your label set differs, edit `LABEL_TO_KEY` in `utils.py`.

## Run

Run everything:

```bash
python orchestrator.py
```

The orchestrator imports each `model_*.py` module, calls its `main()`, captures run status and elapsed time, collects the per-model `*_summary.json` files, and writes a Markdown report to both `results/report.md` and `./report.md`.

Run a single model:

```bash
python model_xgboost.py
```

## What happens on each run

For every model the pipeline:

1. **Loads** `data/file_name.pkl`, filters to known classes, maps labels to ints.
2. **Deduplicates** on `(sentence, label)`.
3. **Upsamples** the train set so every class matches the majority class size.
4. **Cross-validates** (stratified 5-fold) the default pipeline and reports mean accuracy, precision, recall, and F1 (macro).
5. **Grid-searches** a small hyperparameter grid (5-fold CV, scoring = macro F1) and refits the best estimator on the full train set.
6. **Evaluates** the refit estimator on the test set and prints the classification report.
7. **Saves** artifacts into `results/`:
   - `<model>_classification_report.txt`
   - `<model>_confusion_matrix.csv` and `.png`
   - `<model>_summary.json` (CV means, best params, test metrics)
   - `models/<model>.pkl` (the best estimator, pickled)

The orchestrator then prints and writes a combined Markdown report covering all models, including run status, CV metrics, test metrics, best hyperparameters, and the overall winner by test F1.

## Reproducibility

All random seeds are controlled by `RANDOM_STATE = 47` in `utils.py`.

## Notes

- `model_voting.py` tunes only the voting strategy and weights; tuning every base learner inside nested CV would be prohibitively slow.
- `model_stacking.py` tunes the meta-learner's `C` and `passthrough`.
- `matplotlib` uses the `Agg` backend so the scripts work in headless environments.
