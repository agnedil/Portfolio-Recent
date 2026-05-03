"""
Graph node implementations.

Each function defined here is a LangGraph node responsible for a single step
of the agentic workflow:

    - analyze_chat_and_summarize: extracts conversation context for query rewriting
    - analyze_and_rewrite_query:  rewrites the user query, splits multi-part questions
    - human_input_node:           interrupt placeholder for human-in-the-loop
    - route_after_rewrite:        routes to agents (parallel via Send) or human input
    - agent_node:                 ReAct loop that runs retrieval tools
    - extract_final_answer:       extracts the agent's final non-tool message
    - aggregate_responses:        merges per-question agent answers into a final reply

Each node operates on a ``State`` / ``AgentState`` dict and returns a partial
state update. Replace any individual node to change one step of the pipeline
without touching the others.

Documentation:
    - https://docs.langchain.com/oss/python/langgraph/graph-api#nodes
    - https://docs.langchain.com/oss/python/langgraph/graph-api#edges
    - https://docs.langchain.com/oss/python/langgraph/graph-api#send
"""

from typing import Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langgraph.types import Send

from prompts import (
    get_aggregation_prompt,
    get_conversation_summary_prompt,
    get_query_analysis_prompt,
    get_rag_agent_prompt,
)
from state import AgentState, QueryAnalysis, State


def build_nodes(llm: BaseChatModel, llm_with_tools: Runnable) -> Dict[str, Callable]:
    """Bind nodes to the configured LLMs and return a name -> callable map.

    Args:
        llm: The chat model used for non-tool reasoning steps.
        llm_with_tools: The chat model with retrieval tools bound for the agent.

    Returns:
        A dict suitable for ``StateGraph.add_node`` lookups.
    """

    def analyze_chat_and_summarize(state: State):
        """Summarize prior conversation history for use during query rewriting."""
        # Need at least a few turns of history to bother summarizing.
        if len(state["messages"]) < 4:
            return {"conversation_summary": ""}

        # Keep only user / assistant content messages, exclude tool-call traces.
        relevant_msgs = [
            msg for msg in state["messages"][:-1]
            if isinstance(msg, (HumanMessage, AIMessage))
            and not getattr(msg, "tool_calls", None)
        ]
        if not relevant_msgs:
            return {"conversation_summary": ""}

        conversation = "Conversation history:\n"
        for msg in relevant_msgs[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conversation += f"{role}: {msg.content}\n"

        summary_response = llm.invoke([
            SystemMessage(content=get_conversation_summary_prompt()),
            HumanMessage(content=conversation),
        ])
        # Reset agent_answers on a new turn to avoid mixing with prior answers.
        return {
            "conversation_summary": summary_response.content,
            "agent_answers": [{"__reset__": True}],
        }

    def analyze_and_rewrite_query(state: State):
        """Rewrite the latest user query and optionally split it into sub-queries."""
        last_message = state["messages"][-1]
        conversation_summary = state.get("conversation_summary", "")

        # Build context block: include the summary only when non-empty.
        context_section = (
            f"Conversation Context:\n{conversation_summary}\n"
            if conversation_summary.strip() else ""
        ) + f"User Query:\n{last_message.content}\n"

        # Structured output ensures we always get back the QueryAnalysis schema.
        llm_struct = llm.with_structured_output(QueryAnalysis)
        response = llm_struct.invoke([
            SystemMessage(content=get_query_analysis_prompt()),
            HumanMessage(content=context_section),
        ])

        if response.questions and response.is_clear:
            # Drop in-flight messages so each new query starts the agent cleanly.
            delete_all = [
                RemoveMessage(id=m.id)
                for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]
            return {
                "questionIsClear": True,
                "messages": delete_all,
                "originalQuery": last_message.content,
                "rewrittenQuestions": response.questions,
            }

        # Unclear query: surface a clarification message and wait for the user.
        clarification = (
            response.clarification_needed
            if (response.clarification_needed
                and len(response.clarification_needed.strip()) > 10)
            else "I need more information to understand your question."
        )
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)],
        }

    def human_input_node(state: State):
        """Interrupt placeholder for human-in-the-loop clarification."""
        return {}

    def route_after_rewrite(state: State):
        """Route to per-question agents in parallel, or to human input if unclear."""
        if not state.get("questionIsClear", False):
            return "human_input"
        # Fan out: one agent invocation per rewritten sub-question.
        return [
            Send(
                "process_question",
                {"question": query, "question_index": idx, "messages": []},
            )
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]

    def agent_node(state: AgentState):
        """ReAct agent loop: searches first, then optionally retrieves parents."""
        sys_msg = SystemMessage(content=get_rag_agent_prompt())
        # First call: seed with the question. Subsequent calls: continue from history.
        if not state.get("messages"):
            human_msg = HumanMessage(content=state["question"])
            response = llm_with_tools.invoke([sys_msg, human_msg])
            return {"messages": [human_msg, response]}
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    def extract_final_answer(state: AgentState):
        """Extract the last non-tool AI message as the agent's final answer."""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return {
                    "final_answer": msg.content,
                    "agent_answers": [{
                        "index": state["question_index"],
                        "question": state["question"],
                        "answer": msg.content,
                    }],
                }
        # Fallback when the agent produced no usable content.
        return {
            "final_answer": "Unable to generate an answer.",
            "agent_answers": [{
                "index": state["question_index"],
                "question": state["question"],
                "answer": "Unable to generate an answer.",
            }],
        }

    def aggregate_responses(state: State):
        """Merge per-question agent answers into a single coherent response."""
        if not state.get("agent_answers"):
            return {"messages": [AIMessage(content="No answers were generated.")]}

        # Preserve original sub-question order in the final answer.
        sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])
        formatted = ""
        for i, ans in enumerate(sorted_answers, start=1):
            formatted += f"\nAnswer {i}:\n{ans['answer']}\n"

        user_message = HumanMessage(
            content=(
                f"Original user question: {state['originalQuery']}\n"
                f"Retrieved answers:{formatted}"
            )
        )
        synthesis = llm.invoke([
            SystemMessage(content=get_aggregation_prompt()),
            user_message,
        ])
        return {"messages": [AIMessage(content=synthesis.content)]}

    return {
        "analyze_chat_and_summarize": analyze_chat_and_summarize,
        "analyze_and_rewrite_query": analyze_and_rewrite_query,
        "human_input_node": human_input_node,
        "route_after_rewrite": route_after_rewrite,
        "agent_node": agent_node,
        "extract_final_answer": extract_final_answer,
        "aggregate_responses": aggregate_responses,
    }
