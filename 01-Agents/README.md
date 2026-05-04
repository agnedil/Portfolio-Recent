# Agents

A reference collection of LLM-agent **patterns**, each implemented across the major orchestration frameworks (LangChain, LangGraph, Google ADK, CrewAI, OpenAI Agents) and the two interoperability protocols (Model Context Protocol and Agent-to-Agent). Subfolders are organised by *pattern* — tool use, ReAct, routing, planning, reasoning, self-reflection, parallelization, multi-agent collaboration, memory, MCP, A2A — and inside each folder the same pattern is implemented in several frameworks where it has an idiomatic form.

## Introduction: Why several frameworks and protocols?

"Build an agent" is not a single thing — it is a *family* of patterns (decide what tool to call, route between sub-agents, plan multi-step work, reflect on output, share memory across turns, coordinate with another agent over the network). Every major orchestration framework expresses these patterns differently, and the right choice depends on what you are optimising for. The folder shows the same patterns side-by-side so the trade-offs are concrete rather than abstract.

### Why several frameworks (LangChain, LangGraph, Google ADK, CrewAI, OpenAI Agents)?

- **LangChain** — the broadest ecosystem. Mature integrations for almost every LLM, vector store, and tool. Good for *linear* agent flows (tool-calling agents, ReAct). Hits its limits when you want explicit control flow or cycles.
- **LangGraph** — LangChain's successor for *non-linear* agent flows. Models the agent as an explicit state graph; control flow and state are first-class. The right answer for self-reflection loops, multi-step plans with cycles, and human-in-the-loop interrupts.
- **Google ADK (Agent Development Kit)** — Google's first-party agent framework, built around Gemini. Native primitives for `SequentialAgent`, `ParallelAgent`, `LoopAgent`, and tool-as-agent (`agent_tool.AgentTool`). The most ergonomic option when you are already on GCP / Vertex AI.
- **CrewAI** — opinionated multi-agent abstraction: agents have *roles*, *goals*, and *backstories*; tasks are assigned to agents and executed by a `Crew`. Shines when the work decomposes naturally into named roles (researcher, writer, reviewer).
- **OpenAI Agents Framework** — OpenAI's lightweight agent SDK; minimal surface, designed to pair tightly with the OpenAI Responses API and built-in tools.

In short: LangChain for breadth, LangGraph for control-flow, ADK on GCP, CrewAI for role-based teams, OpenAI Agents for OpenAI-native stacks.

### Why two interoperability protocols (MCP, A2A)?

These are not orchestration frameworks — they are *wire protocols* that let agents and tools from different stacks talk to each other.

- **MCP — Model Context Protocol** — a standard for exposing tools, resources, and prompts to an LLM client over stdio / HTTP / SSE. Solves "I built a tool in framework X; how does framework Y call it?" Every modern agent runtime is shipping MCP support.
- **A2A — Agent-to-Agent protocol** — a standard for one agent to discover and call another (capability cards, request / response, streaming). Solves "I have agent A in CrewAI and agent B in ADK; how do they collaborate without sharing a process?"

A reviewer can compare a single-process pattern (LangChain `AgentExecutor`) directly with a cross-process pattern (MCP toolset on the agent side, FastMCP server on the tool side) and see what changes.

### Why several LLMs (GPT, Gemini, Claude)?

Different patterns are easier to write or perform better on different model families. The folder uses whichever model is most idiomatic for each framework — Gemini in the Google ADK examples, GPT-4o in the LangChain / OpenAI Agents examples, with Claude swappable wherever a `ChatAnthropic` substitution makes sense. Most scripts read the LLM choice from constants near the top, so swapping is a one-line change.

## Repository layout

Files are grouped by **pattern**, with the framework appearing in the file name. The numbering is roughly a learning order: tool-use is the foundation that every other pattern builds on; multi-agent / MCP / A2A are the more advanced compositions at the end.

### `00-intro/` — long-form walkthrough

- `ai_agents_tools_and_websearch.py` — a 286-line end-to-end walkthrough that introduces tools and web search from scratch in LangChain. Good first read.

### `01-tool-use/` — let the LLM call functions

The foundation pattern: bind one or more tools to an LLM, let the model decide when to call them, and execute the tool calls. Five framework variants:

- `tool_use_langchain_search.py` — LangChain `create_tool_calling_agent` + `AgentExecutor` over a simulated search tool, async batch execution.
- `tool_use_google_search.py` / `google_search.py` — Google ADK `Agent` with the built-in `google_search` tool.
- `tool_use_google_adk_calculator.py` — ADK `FunctionTool` wrapper around a custom Python function.
- `tool_use_crew_ai_stocks.py` — CrewAI agent with a `@tool`-decorated stock-lookup function.
- `tool_use_vertex_ai_search.py` — ADK `Agent` calling a Vertex AI Search datastore (managed retrieval).

### `02-react/` — Reason → Act → Observe loop

The classic ReAct pattern: the model interleaves reasoning steps with tool calls until it has enough information to answer.

- `react_langchain.py` — `initialize_agent(..., agent="zero-shot-react-description")` over a Google-Serper + LLM-Math toolset.

### `03-routing/` — pick the right agent / tool for the question

Routing chooses between alternatives (different tools, different sub-agents, different prompts) before doing the work.

- `routing_langchain.py` — LangChain `RunnableBranch` switching on a classifier output.
- `routing_google_adk.py` — ADK coordinator agent dispatching to specialist sub-agents via `agent_tool.AgentTool`.

### `04-planning/` — decompose a goal into ordered steps

Planning produces an explicit plan (often as a structured list) before execution begins — useful when the goal is complex enough that step ordering matters.

- `planning_crew_ai.py` — CrewAI's built-in planner role producing a Task graph.
- `planning_openai_deep_research.py` — OpenAI Deep Research API: model produces a research plan, then executes it with web search.

### `05-reasoning/` — give the agent an explicit reasoning surface

A small ADK example showing how to combine a search agent with a code-execution agent under a coordinator so reasoning steps can run code rather than just generate text.

- `reasoning_google_adk.py`

### `06-self-reflection/` — let the agent critique and revise its own output

Six variants of the same idea — generate an answer, critique it, regenerate if the critique fails — across frameworks. Useful for comparing how each stack handles cycles.

- `self_reflection_langchain.py`, `self_reflection_langchain_2.py`, `self_reflection_langchain_iterative.py` — three LangChain variants (basic, structured-output, iterative).
- `self_reflection_langgraph.py` — same loop expressed as an explicit state graph (the cleanest of the six).
- `self_reflection_google_adk.py` — ADK `LoopAgent` with a critique sub-agent.
- `self_reflection_openai_agents_framework.py` — minimal OpenAI Agents version.

### `07-parallelization/` — fan out multiple agents on independent sub-tasks

Parallelization runs sub-tasks concurrently and aggregates the results — the right move when sub-tasks are independent and latency dominates.

- `parallelization_langchain.py` — `RunnableParallel` over multiple chains.
- `parallelization_google_adk.py` — ADK `ParallelAgent` with explicit aggregation.

### `08-multi-agent-collaboration/` — multiple agents working together

Coordination patterns when more than one agent is in the loop. The four `multi_agent_collaboration_gemini_*.py` files demonstrate ADK's four built-in topologies side-by-side; the remaining files show the same idea in CrewAI and LangChain.

- `multi_agent_collaboration_gemini_sequential.py` — `SequentialAgent` (output of step *n* feeds step *n+1*).
- `multi_agent_collaboration_gemini_parallel.py` — `ParallelAgent` (sibling agents run concurrently).
- `multi_agent_collaboration_gemini_loop.py` — `LoopAgent` (iterate until a stop condition).
- `multi_agent_collaboration_gemini_coordinator.py` — coordinator agent dispatching to sub-agents as tools.
- `multi_agent_collaboration_google_adk.py` — a more complete ADK example wiring all of the above.
- `multi_agent_collaboration_crewai_gemini.py` — CrewAI roles + tasks driven by Gemini.
- `multi_agent_collaboration_langchain.py` — LangChain equivalent for comparison.

### `09-memory-management/` — give the agent state across turns

Short-term (within-conversation) and longer-term (cross-session) memory.

- `ai_agents_chatbots_agents_memory.py` — LangChain `ConversationBufferMemory` + chatbot.
- `memory_management_langchain_and_langgraph.py` — `ChatMessageHistory` and LangGraph's `InMemoryStore` side-by-side.
- `memory_management_google_adk.py` / `memory_management_google_adk_2.py` — ADK `InMemorySessionService` and event-based session state.

### `10-mcp/` — Model Context Protocol

Both sides of the MCP wire: standing up a server, and consuming one.

- `model_context_protocol_fastmcp_server.py` — FastMCP server exposing tools.
- `model_context_protocol_consuming_fastmcp.py` — client that connects to a FastMCP server.
- `model_context_protocol_mcp_filesystem.py` — Google ADK `MCPToolset` using stdio to talk to the reference filesystem MCP server, so an ADK agent can list / read files via MCP.

Has its own `requirements.txt` (FastMCP + the MCP spec library + ADK).

### `11-a2a-protocol/` — Agent-to-Agent protocol

Cross-agent capability discovery and request / response.

- `a2a_protocol_agent_card.py` — defining an A2A agent card (capability advertisement).
- `a2a_protocol.py` — making A2A requests against another agent.

Has its own `requirements.txt` (just the `a2a-sdk`).

### `notebooks/`

Two Jupyter notebooks that mirror the introductory scripts in pretty-printed, runnable form:

- `AI_Agents_Tools_and_Websearch_pprint.ipynb`
- `LangChain_Chatbots_Agents_Memory_pprint.ipynb`

## Setup

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install dependencies

The root `requirements.txt` is the union of every dependency referenced anywhere in this folder. For most patterns this is overkill — pick the focused install instead:

```bash
# Everything
pip install -r requirements.txt

# Or, for a specific pattern with a heavier stack:
pip install -r 08-multi-agent-collaboration/requirements.txt
pip install -r 10-mcp/requirements.txt
pip install -r 11-a2a-protocol/requirements.txt
```

The other subfolders are covered by the root file (mostly LangChain + Google ADK + OpenAI).

### 3. API keys

Copy `.env.example` to `.env` and fill in the keys you need. Each script only requires the keys for the providers/tools it actually uses — see the per-pattern descriptions above. The minimum to run the LangChain examples is `OPENAI_API_KEY`; the minimum to run the ADK / Gemini examples is `GOOGLE_API_KEY`.

```bash
cp .env.example .env
# edit .env and fill in the keys you have
```

## Suggested reading order

1. **`00-intro/ai_agents_tools_and_websearch.py`** — the long-form walkthrough that grounds the rest.
2. **`01-tool-use/`** — start with the LangChain example, then compare against the ADK and CrewAI versions to see how the same pattern is expressed differently.
3. **`02-react/`** and **`03-routing/`** — the two simplest control-flow patterns on top of tool use.
4. **`06-self-reflection/`** — six implementations of the same loop is the best place to compare frameworks head-to-head; the LangGraph version is the cleanest.
5. **`07-parallelization/`** and **`08-multi-agent-collaboration/`** — composition patterns on top of the basics.
6. **`09-memory-management/`** — orthogonal axis: state across turns.
7. **`04-planning/`** and **`05-reasoning/`** — higher-level cognitive patterns.
8. **`10-mcp/`** and **`11-a2a-protocol/`** — interoperability between agents/tools across processes and across frameworks.
