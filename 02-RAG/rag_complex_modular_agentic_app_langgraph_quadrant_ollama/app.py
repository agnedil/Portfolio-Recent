"""
Gradio chat interface.

Wraps a compiled LangGraph as a Gradio ``ChatInterface`` with:

    - per-conversation thread IDs persisted by the graph checkpointer
    - automatic resumption when the graph is interrupted at ``human_input``
    - a Clear button that drops checkpointer state for the current thread

Replace this module to swap UI (FastAPI, Streamlit, Slack bot, etc.). Only the
compiled graph object is required to serve queries.

Documentation:
    - https://www.gradio.app/docs/gradio/chatinterface
    - https://www.gradio.app/docs/blocks
"""

import logging
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage

from config import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT

logger = logging.getLogger(__name__)


def _new_thread_config() -> dict:
    """Generate a fresh thread config for the LangGraph checkpointer."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def build_demo(agent_graph) -> gr.Blocks:
    """Build a Gradio Blocks demo that talks to the compiled agent graph.

    Args:
        agent_graph: Compiled LangGraph (output of ``graph.build_agent_graph``).

    Returns:
        A ``gr.Blocks`` instance ready for ``.launch()``.
    """
    # Per-demo mutable holder so closures can swap thread IDs after Clear.
    state = {"config": _new_thread_config()}

    def clear_session():
        """Drop checkpointer state and start a new thread."""
        try:
            agent_graph.checkpointer.delete_thread(
                state["config"]["configurable"]["thread_id"]
            )
        except Exception as e:
            logger.warning("Could not delete thread state: %s", e)
        state["config"] = _new_thread_config()

    def chat_with_agent(message: str, history) -> str:
        """Single-turn chat handler with human-in-the-loop resumption."""
        cfg = state["config"]
        current_state = agent_graph.get_state(cfg)

        if current_state.next:
            # Graph is interrupted (waiting for clarification): inject and resume.
            agent_graph.update_state(
                cfg, {"messages": [HumanMessage(content=message.strip())]}
            )
            result = agent_graph.invoke(None, cfg)
        else:
            # Fresh turn: start a new run with the user message.
            result = agent_graph.invoke(
                {"messages": [HumanMessage(content=message.strip())]}, cfg
            )
        return result["messages"][-1].content

    with gr.Blocks(theme=gr.themes.Citrus()) as demo:
        chatbot = gr.Chatbot(
            height=600,
            placeholder=(
                "<strong>Ask me anything!</strong><br>"
                "<em>I'll search, reason, and act to give you the best answer.</em>"
            ),
            type="messages",
        )
        chatbot.clear(clear_session)
        gr.ChatInterface(fn=chat_with_agent, chatbot=chatbot, type="messages")

    return demo


def launch(agent_graph) -> None:
    """Build and launch the Gradio app on the configured host / port."""
    demo = build_demo(agent_graph)
    logger.info(
        "Launching Gradio app at http://%s:%d", GRADIO_SERVER_NAME, GRADIO_SERVER_PORT
    )
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
