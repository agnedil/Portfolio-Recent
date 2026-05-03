"""
Interactive REPL for the RAG application.

Compiles the graph once and loops on stdin: ask a question, print the answer
and the number of chunks the retriever surfaced. Press Enter on an empty
line (or send Ctrl-D / Ctrl-C) to exit.

Usage:
    export OPENAI_API_KEY=...
    python serve.py
"""

from build_rag_graph import build_graph


def main() -> None:
    """Compile the graph once and serve queries in a loop."""
    # Build the graph upfront so the first question doesn't pay the setup cost.
    rag_app = build_graph()
    print("RAG app ready. Type a question (empty line to quit).")

    while True:
        # Graceful exit on EOF / Ctrl-C as well as empty input.
        try:
            q = input("\nAsk: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break

        # Single graph invocation per question; state is fresh each turn.
        result = rag_app.invoke({"question": q, "docs": [], "answer": ""})
        print("\nAnswer:\n", result["answer"])
        print("\nChunks used:", len(result["docs"]))


if __name__ == "__main__":
    main()
