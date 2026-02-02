from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
        
# load a pre-indexed vector DB
vectorstore = Chroma( persist_directory="chroma_db",
                      collection_name="rag_docs",
                      embedding_function=OpenAIEmbeddings(), )

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
def retrieve(state: RAGState) -> RAGState:
    q = state["question"]
    docs = retriever.invoke(q)
    return {"question": q, "docs": docs, "answer": ""}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
def generate(state: RAGState) -> RAGState:
    q = state["question"]
    docs = state["docs"]
    context = "\n\n".join([d.page_content for d in docs])
    messages = [ SystemMessage( content=(
                "Answer using ONLY the context. If missing is missing,"
                "say you don't have that info in the documents." ) ),
                HumanMessage(content=f"Question:\n{q}\n\nContext:\n{context}") ]
    resp = llm.invoke(messages)
    return {**state, "answer": resp.content}

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()

rag_app = build_graph()