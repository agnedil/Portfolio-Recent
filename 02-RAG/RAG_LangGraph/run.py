from rag_graph import rag_app

while True:
    q = input("\nAsk: ").strip()
    if not q: break
    result = rag_app.invoke( {"question": q, "docs": [], "answer": ""} )
    print("\nAnswer:\n", result["answer"])
    print("\nChunks used:", len(result["docs"]))


# NOTE: With LangGraph, it is easy to insert nodes and rewire
g.add_node("rewrite_query", rewrite)
g.add_node("rerank", rerank)
g.add_node("evaluate_response", evaluate)