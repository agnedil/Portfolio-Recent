# RAG with LangGraph
# Why LangGraph
# 
# 1. Explicit Control Flow
# * LangGraph: You can see exactly how data flows between steps (retrieve → generate) - makes debugging and understanding the pipeline much easier.
# * LangChain chains: Control flow is more implicit and hidden in chain internals.
# * With LangGraph, it is easy to insert nodes and rewire the tree.
# 
# 2. State Management
# * LangGraph: Explicit state object (RAGState) that you control
# * Chains: State is passed implicitly, harder to inspect or modify mid-pipeline
# 
# 3. Scalability for Complex Workflows
# * LangGraph gives you room to grow without refactoring. The code below is simple now, but LangGraph makes it trivial to add query rewriting, query routing, iterative retrieval, response evaluation, reranking, self-critique, multi-hop retrieval or other agentic behaviors later: just use `g.add_node()`


# INDEX.PY - load, chunk, and vectorize
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = Path("docs")
def load_docs():
    return [ doc for fp in DOCS_DIR.glob("*") if fp.is_file()
             for doc in TextLoader(str(fp), encoding="utf-8").load() ]

splitter   = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120,)
chunks     = splitter.split_documents(load_docs())
embeddings = OpenAIEmbeddings()

Chroma.from_documents( documents=chunks, embedding=embeddings,
                       persist_directory="chroma_db", collection_name="rag_docs", )
print("✅ Index built")