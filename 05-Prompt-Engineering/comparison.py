"""Compare emotion classifiers across OpenAI, Anthropic, Gemini, and HuggingFace
on the WASSA 2023 dev set. Per-API classification reports and an aggregated
summary are written to the data directory as CSVs."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from classifiers import (
    AnthropicEmotionClassifier,
    BaseEmotionClassifier,
    EMOTION_LABELS,
    GeminiEmotionClassifier,
    HuggingFaceEmotionClassifier,
    OpenAIEmotionClassifier,
)


TEXT_COL = "essay_clean_spellchecked"
LABEL_COL = "emotion"
DATA_DIR = Path(__file__).parent / "data"

_CANON = {label.lower(): label for label in EMOTION_LABELS}
_NON_ALPHA = re.compile(r"[^a-zA-Z]+")


def primary_label(value: object) -> str:
    """Reduce the WASSA `emotion` field (str / list / slash-joined) to a single label."""
    if isinstance(value, (list, tuple)):
        return _CANON.get(str(value[0]).lower(), str(value[0])) if value else "Neutral"
    if isinstance(value, str):
        head = value.split("/")[0].strip()
        return _CANON.get(head.lower(), head)
    return str(value)


def normalize_prediction(raw: str) -> str:
    """Best-effort parse of a raw model output into a canonical label."""
    if not raw:
        return "INVALID"
    for token in _NON_ALPHA.split(raw):
        if token and token.lower() in _CANON:
            return _CANON[token.lower()]
    return "INVALID"


def predict_all(
    classifier: BaseEmotionClassifier, texts: Iterable[str]
) -> list[str]:
    preds: list[str] = []
    iterator = tqdm(list(texts), desc=f"{classifier.name} ({classifier.model})", file=sys.stderr)
    for text in iterator:
        try:
            raw = classifier.classify(text)
        except Exception as exc:
            print(f"[{classifier.name}] error: {exc}", file=sys.stderr)
            raw = ""
        preds.append(normalize_prediction(raw))
    return preds


def report_dataframe(gold: list[str], preds: list[str]) -> tuple[pd.DataFrame, dict]:
    report = classification_report(
        gold,
        preds,
        labels=list(EMOTION_LABELS),
        output_dict=True,
        zero_division=0,
    )
    rows = {k: v for k, v in report.items() if isinstance(v, dict)}
    df = pd.DataFrame(rows).T
    df.index.name = "label"
    return df, report


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of dev examples to process (useful for smoke runs).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing df_dev.pkl; output CSVs are written here.",
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    dev_path = args.data_dir / "df_dev.pkl"
    if not dev_path.exists():
        print(f"error: {dev_path} not found", file=sys.stderr)
        sys.exit(2)

    df_dev = pd.read_pickle(dev_path)
    if args.limit:
        df_dev = df_dev.head(args.limit)

    texts = df_dev[TEXT_COL].tolist()
    gold = [primary_label(v) for v in df_dev[LABEL_COL].tolist()]
    print(f"Loaded {len(texts)} dev examples from {dev_path}", file=sys.stderr)

    classifiers: list[BaseEmotionClassifier] = [
        OpenAIEmotionClassifier(),
        AnthropicEmotionClassifier(),
        GeminiEmotionClassifier(),
        HuggingFaceEmotionClassifier(),
    ]

    summary_rows = []
    for clf in classifiers:
        preds = predict_all(clf, texts)
        df_report, report = report_dataframe(gold, preds)

        report_path = args.data_dir / f"classification_report_{clf.name}_{safe_filename(clf.model)}.csv"
        df_report.to_csv(report_path)

        print(f"\n=== {clf.name}  model={clf.model} ===")
        print(df_report.round(3).to_string())
        print(f"saved -> {report_path}")

        summary_rows.append(
            {
                "classifier": clf.name,
                "model": clf.model,
                "n_examples": len(gold),
                "accuracy": report.get("accuracy", 0.0),
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_path = args.data_dir / "classifier_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Comparison Summary (sorted by macro_f1) ===")
    print(summary_df.round(4).to_string(index=False))
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
