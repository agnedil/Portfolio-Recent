"""
LangGraph state schemas.

Defines the typed state containers passed between graph nodes:

    - ``State``:         top-level conversation state for the orchestrator graph
    - ``AgentState``:    per-question state for the inner retrieval subgraph
    - ``QueryAnalysis``: structured output schema for the query rewriter

State management:
    - ``accumulate_or_reset`` is a custom reducer for ``agent_answers`` that
      appends new entries unless an explicit ``__reset__`` marker is present.
    - Both state classes inherit from ``MessagesState`` so chat history is
      preserved across nodes via the standard messages reducer.

Replace this module if the data contract between nodes needs to change.

Documentation:
    - https://langchain-ai.github.io/langgraph/concepts/low_level/#state
"""

from typing import Annotated, List

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """Reducer that appends to a list unless a ``__reset__`` marker is present."""
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new


class State(MessagesState):
    """State for the main orchestrator graph."""
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []


class AgentState(MessagesState):
    """State for an individual retrieval agent subgraph."""
    question: str = ""
    question_index: int = 0
    final_answer: str = ""
    agent_answers: List[dict] = []


class QueryAnalysis(BaseModel):
    """Structured output schema for the query analyzer / rewriter."""
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )
