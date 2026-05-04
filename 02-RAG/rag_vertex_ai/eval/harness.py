from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_vertex.client import RagMemoryClient
from rag_vertex.settings import RagSettings


@dataclass
class EvalCase:
    query: str
    relevant_sources: set[str]


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text())
    return [
        EvalCase(query=c["query"], relevant_sources=set(c["relevant_sources"]))
        for c in raw
    ]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def run(client: RagMemoryClient, cases: list[EvalCase], k_max: int = 10) -> dict:
    per_query = []
    for case in cases:
        chunks = client.retrieve(case.query, top_k=k_max)
        sources = [c.source for c in chunks if c.source]
        per_query.append(
            {
                "query": case.query,
                "n_retrieved": len(sources),
                "recall@5": recall_at_k(sources, case.relevant_sources, 5),
                "recall@10": recall_at_k(sources, case.relevant_sources, 10),
            }
        )

    summary = {
        "n_queries": len(per_query),
        "mean_recall@5": statistics.mean(r["recall@5"] for r in per_query),
        "mean_recall@10": statistics.mean(r["recall@10"] for r in per_query),
    }
    return {"summary": summary, "per_query": per_query}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("eval/queries.yaml"))
    parser.add_argument("--out", type=Path, default=Path("eval/results.json"))
    args = parser.parse_args()

    client = RagMemoryClient.from_settings(RagSettings())
    cases = load_cases(args.queries)
    report = run(client, cases)

    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
