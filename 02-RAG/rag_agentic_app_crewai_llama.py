#! /usr/bin/env python
"""
Agentic RAG Demo: AI Knowledge Assistant
=========================================

A multi-agent Retrieval-Augmented Generation app served as a Gradio web UI.
Built with CrewAI for agent orchestration, ``crewai_tools.PDFSearchTool`` for
RAG over a research paper (the "Attention Is All You Need" PDF), Tavily for
live web search, and Groq (via the OpenAI-compatible API) as the LLM
backend.

Architecture
------------
A small "crew" of five agents executes in sequence on every question:

    1. Router Agent     - decides between vectorstore (PDF) and web search.
    2. Retriever Agent  - calls the chosen tool to fetch context.
    3. Grader Agent     - filters out irrelevant retrievals.
    4. Hallucination Grader - verifies the answer is grounded in the context.
    5. Final Answer Agent  - synthesizes the response, with optional web
                             search top-up.

The PDF is downloaded once and indexed once at startup (with HuggingFace
embeddings via embedchain inside ``PDFSearchTool``); the crew object is
built once and reused across questions.

Setup
-----
1. Install dependencies::

       pip install gradio requests crewai crewai-tools langchain-openai \\
           langchain-community tavily-python sentence-transformers

2. Set your API keys in the shell *before* launching the app::

       export GROQ_API_KEY=...
       export TAVILY_API_KEY=...

3. Run the app::

       python rag_agentic_app_crewai_llama.py

   Open the URL printed by Gradio (default http://127.0.0.1:7860).

Notes
-----
- The app reads ``GROQ_API_KEY`` and ``TAVILY_API_KEY`` from the environment.
  It will refuse to start if either is missing.
- First-question latency includes a one-time PDF download and a HuggingFace
  embedding-model download. Subsequent questions reuse the cached index.
- ``Crew(verbose=True)`` prints agent traces to the terminal where you ran
  ``python ...``. Set it to ``False`` to silence it.
"""

import os
import sys
from typing import Tuple

import gradio as gr
import requests
from crewai import Agent, Crew, Task
from crewai_tools import PDFSearchTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# ----------------------------- Configuration ------------------------------- #

PDF_URL = (
    "https://proceedings.neurips.cc/paper_files/paper/2017/file/"
    "3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
)
PDF_PATH = "attention_is_all_you_need.pdf"

# Groq (OpenAI-compatible) chat model used by every agent.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama3-8b-8192"

# Embeddings used by PDFSearchTool's vector index.
EMBED_PROVIDER = "huggingface"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _require_env(*names: str) -> None:
    """Fail fast if any of the named environment variables is unset."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        # Print to stderr and exit so the user sees a clear, actionable error.
        sys.stderr.write(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + "\nSet them in your shell, e.g.\n"
            "    export GROQ_API_KEY=...\n"
            "    export TAVILY_API_KEY=...\n"
        )
        sys.exit(1)


# ------------------------------ LLM factory -------------------------------- #

def build_llm() -> ChatOpenAI:
    """Build the Groq-backed chat LLM used by every agent."""
    # Routes OpenAI SDK calls to Groq's OpenAI-compatible endpoint.
    return ChatOpenAI(
        openai_api_base=GROQ_BASE_URL,
        openai_api_key=os.environ["GROQ_API_KEY"],
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=1000,
    )


# ------------------------------- PDF setup --------------------------------- #

def download_pdf(url: str = PDF_URL, path: str = PDF_PATH) -> str:
    """Download the source PDF if it isn't already cached locally.

    Args:
        url: URL of the PDF to download.
        path: Local path where the PDF will be saved.

    Returns:
        The local path of the available PDF file.
    """
    # Skip the download if the file is already present.
    if not os.path.exists(path):
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
    return path


# ------------------------------- Tool setup -------------------------------- #

def setup_tools() -> Tuple[PDFSearchTool, TavilySearchResults]:
    """Build the RAG tool (PDF vector search) and the web-search tool.

    Returns:
        ``(rag_tool, web_search_tool)`` ready to be attached to agents.
    """
    # PDFSearchTool indexes the PDF with the configured embedder on first use.
    rag_tool = PDFSearchTool(
        pdf=PDF_PATH,
        config=dict(
            llm=dict(provider="groq", config=dict(model=LLM_MODEL)),
            embedder=dict(
                provider=EMBED_PROVIDER,
                config=dict(model=EMBED_MODEL),
            ),
        ),
    )
    # Top-k=3 keeps the prompt small.
    web_search_tool = TavilySearchResults(k=3)
    return rag_tool, web_search_tool


# ------------------------------ Router tool -------------------------------- #

# Note: crewai_tools' @tool decorator requires a name string.
@tool("Router Tool")
def router_tool(question: str) -> str:
    """Route a question to the vectorstore or to live web search.

    Args:
        question: The user's natural-language question.

    Returns:
        ``"vectorstore"`` for paper-related questions, otherwise
        ``"web_search"``.
    """
    # Simple keyword heuristic; the LLM router agent uses this as a hint.
    keywords = ["self-attention", "transformer", "attention", "language model"]
    if any(k in question.lower() for k in keywords):
        return "vectorstore"
    return "web_search"


# ------------------------------- Agents ------------------------------------ #

def create_agents(llm: ChatOpenAI):
    """Build the five-agent crew used to answer each question."""
    # 1. Router decides between vectorstore and web search.
    router_agent = Agent(
        role="Router",
        goal="Route user question to a vectorstore or web search",
        backstory=(
            "You are an expert at routing a user question to a vectorstore "
            "or web search. Use the vectorstore for questions related to "
            "Retrieval-Augmented Generation. Be flexible in interpreting "
            "keywords related to these topics."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # 2. Retriever invokes the chosen tool to gather context.
    retriever_agent = Agent(
        role="Retriever",
        goal="Use retrieved information to answer the question",
        backstory=(
            "You are an assistant for question-answering tasks. Provide "
            "clear, concise answers using retrieved context."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # 3. Grader filters out retrievals that don't address the question.
    grader_agent = Agent(
        role="Answer Grader",
        goal="Filter out irrelevant retrievals",
        backstory=(
            "You are a grader assessing the relevance of retrieved "
            "documents. Evaluate if the document contains keywords related "
            "to the user question."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # 4. Hallucination grader checks factual grounding in the context.
    hallucination_grader = Agent(
        role="Hallucination Grader",
        goal="Verify answer factuality",
        backstory=(
            "You are responsible for ensuring the answer is grounded in "
            "facts and directly addresses the user's question."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # 5. Final answer agent synthesizes a clean response.
    final_answer_agent = Agent(
        role="Final Answer Agent",
        goal="Provide a comprehensive and accurate response",
        backstory=(
            "You synthesize information from various sources to create a "
            "clear, concise, and informative answer to the user's question."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    return (
        router_agent,
        retriever_agent,
        grader_agent,
        hallucination_grader,
        final_answer_agent,
    )


# ------------------------------- Tasks ------------------------------------- #

def create_tasks(agents, tools):
    """Build the ordered task graph that the crew executes per question."""
    rag_tool, web_search_tool = tools
    (
        router_agent,
        retriever_agent,
        grader_agent,
        hallucination_grader,
        final_answer_agent,
    ) = agents

    # Routing decision (router_tool returns 'vectorstore' or 'web_search').
    router_task = Task(
        description=(
            "Analyze the keywords in the question {question}. "
            "Decide whether it requires a vectorstore search or web search."
        ),
        expected_output="Return 'web_search' or 'vectorstore'",
        agent=router_agent,
        tools=[router_tool],
    )

    # Retrieval driven by the router's verdict.
    retriever_task = Task(
        description=(
            "Retrieve information for the question {question} using either "
            "web search or vectorstore based on the router task."
        ),
        expected_output="Provide retrieved information",
        agent=retriever_agent,
        context=[router_task],
        tools=[rag_tool, web_search_tool],
    )

    # Relevance check on the retrieved content.
    grader_task = Task(
        description=(
            "Evaluate the relevance of retrieved content for the "
            "question {question}"
        ),
        expected_output="Return 'yes' or 'no' for relevance",
        agent=grader_agent,
        context=[retriever_task],
    )

    # Factual-grounding check.
    hallucination_task = Task(
        description="Verify if the retrieved answer is factually grounded",
        expected_output="Return 'yes' or 'no' for factual accuracy",
        agent=hallucination_grader,
        context=[grader_task],
    )

    # Final synthesis (may invoke web search again if the prior chain stalled).
    answer_task = Task(
        description=(
            "Generate a final answer based on retrieved and verified "
            "information. Perform additional search if needed."
        ),
        expected_output="Provide a clear, concise answer",
        agent=final_answer_agent,
        context=[hallucination_task],
        tools=[web_search_tool],
    )

    return [router_task, retriever_task, grader_task, hallucination_task, answer_task]


# ----------------------------- Crew lifecycle ------------------------------ #

# Module-level cache: the crew is expensive to build (downloads PDF, indexes
# embeddings on first use of PDFSearchTool) so we build it exactly once.
_crew_cache = {"crew": None}


def get_crew() -> Crew:
    """Lazily build and return a single Crew instance reused across questions."""
    if _crew_cache["crew"] is None:
        # One-time setup: PDF download, tool init, agents, tasks, crew assembly.
        download_pdf()
        llm = build_llm()
        tools = setup_tools()
        agents = create_agents(llm)
        tasks = create_tasks(agents, tools)
        _crew_cache["crew"] = Crew(
            agents=list(agents),
            tasks=tasks,
            verbose=True,
        )
    return _crew_cache["crew"]


def run_rag_pipeline(question: str) -> str:
    """Run the cached crew against a user question and return the answer."""
    try:
        crew = get_crew()
        result = crew.kickoff(inputs={"question": question})
        # crewai may return a CrewOutput object; coerce to string for Gradio.
        return str(result)
    except Exception as e:
        return f"An error occurred: {e}"


# ----------------------------- Gradio interface ---------------------------- #

def gradio_interface(query: str) -> str:
    """Gradio-facing wrapper around the crew pipeline."""
    if not query or not query.strip():
        return "Please enter a question."
    return run_rag_pipeline(query.strip())


def create_gradio_app() -> gr.Interface:
    """Build the Gradio Interface object."""
    return gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(
            label="Enter your question about AI, Language Models, or Self-Attention"
        ),
        outputs=gr.Textbox(label="Response"),
        title="Agentic RAG Demo: AI Knowledge Assistant",
        description=(
            "Ask questions about AI, Language Models, Transformers, and "
            "Self-Attention mechanisms. The system uses a multi-agent "
            "approach to retrieve and verify information."
        ),
        theme="soft",
        # Use the modern Gradio 4.x flagging API.
        flagging_mode="never",
    )


# --------------------------------- Main ------------------------------------ #

if __name__ == "__main__":
    # Refuse to start if required keys are missing - avoids cryptic 401s later.
    _require_env("GROQ_API_KEY", "TAVILY_API_KEY")

    # Warm the crew (PDF download + embedding model + index) before serving,
    # so the first user question doesn't pay the full setup cost.
    get_crew()

    app = create_gradio_app()
    app.launch()
