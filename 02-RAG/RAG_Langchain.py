# RAG with LangChain
# When Chains Are Fine
# * For simple linear RAG (retrieve → generate → done), chains work perfectly well. But LangGraph gives you room to grow without refactoring - you can add additional nodes easily or introduce cycles (see below)
# * The following is a condensed version of `https://github.com/agnedil/rag-demo-with-streamlit/blob/main/advanced_rag_history.py`


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.llms import Replicate
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains import ConversationalRetrievalChain

def create_rag_chain(pdf_links, chunk_size=1500, top_k=5):
    """Create an advanced RAG system with hybrid retrieval and reranking."""
    # Load and chunk documents
    docs = [OnlinePDFLoader(url).load()[0] for url in pdf_links]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # Create retrievers
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = top_k
    faiss = FAISS.from_documents(chunks, CohereEmbeddings(model="embed-english-light-v3.0")).as_retriever(search_kwargs={"k": top_k})
    
    # Combine with ensemble and reranker
    ensemble = EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.6, 0.4])
    retriever = ContextualCompressionRetriever(base_retriever=ensemble, base_compressor=CohereRerank(top_n=5))
    
    # Create chain
    llm = Replicate(model='meta/llama-2-70b-chat', model_kwargs={"temperature": 0.5, "top_p": 1, "max_new_tokens": 1000})
    return ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)

def query_with_history(chain, query, history=[]):
    """Query the chain and maintain 5 last utterances in chat history"""
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result['answer'], history[-5:]

# Usage
chain = create_rag_chain(['url1.pdf', 'url2.pdf'])
history = []
answer, history = query_with_history(chain, "What is...?", history)