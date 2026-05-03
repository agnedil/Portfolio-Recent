"""
LangGraph construction.

Wires the nodes from ``nodes.py`` into a hierarchical graph:

    Outer (orchestration) graph:
        START -> summarize -> analyze_rewrite -> [human_input | parallel agents]
              -> aggregate -> END

    Inner (per-question) agent subgraph:
        START -> agent <-> tools, then extract_answer -> END

The compiled graph carries an ``InMemorySaver`` checkpointer so each
conversation thread persists across turns, and an ``interrupt_before`` on the
``human_input`` node so the application can prompt the user for clarification.

Replace this module to change the orchestration topology while keeping the
nodes, prompts, tools, and state contracts unchanged.

Documentation:
    - https://docs.langchain.com/oss/python/langgraph/graph-api#stategraph
    - https://docs.langchain.com/oss/python/langgraph/add-memory#add-short-term-memory
"""

from typing import Iterable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from state import AgentState, State


def build_agent_graph(nodes: dict, tools: Iterable):
    """Compile the full hierarchical LangGraph from nodes and tools.

    Args:
        nodes: Dict produced by ``nodes.build_nodes(...)``.
        tools: Iterable of LangChain tool callables exposed to the agent.

    Returns:
        Compiled ``StateGraph`` with checkpointer and ``human_input`` interrupt.
    """
    # --- Inner subgraph: per-question retrieval agent (ReAct loop) ---
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("agent", nodes["agent_node"])
    agent_builder.add_node("tools", ToolNode(list(tools)))
    agent_builder.add_node("extract_answer", nodes["extract_final_answer"])

    agent_builder.add_edge(START, "agent")
    # tools_condition routes to "tools" when the LLM returned a tool call,
    # otherwise to extract_answer.
    agent_builder.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: "extract_answer"}
    )
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("extract_answer", END)
    agent_subgraph = agent_builder.compile()

    # --- Outer graph: orchestration of summarize / rewrite / agents / aggregate ---
    g = StateGraph(State)
    g.add_node("summarize", nodes["analyze_chat_and_summarize"])
    g.add_node("analyze_rewrite", nodes["analyze_and_rewrite_query"])
    g.add_node("human_input", nodes["human_input_node"])
    g.add_node("process_question", agent_subgraph)
    g.add_node("aggregate", nodes["aggregate_responses"])

    g.add_edge(START, "summarize")
    g.add_edge("summarize", "analyze_rewrite")
    g.add_conditional_edges("analyze_rewrite", nodes["route_after_rewrite"])
    # Resume from human_input back into analyze_rewrite once the user clarifies.
    g.add_edge("human_input", "analyze_rewrite")
    # Fan-in: wait for all parallel agent invocations to finish, then aggregate.
    g.add_edge(["process_question"], "aggregate")
    g.add_edge("aggregate", END)

    # Short-term memory + human-in-the-loop interrupt.
    return g.compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["human_input"],
    )
