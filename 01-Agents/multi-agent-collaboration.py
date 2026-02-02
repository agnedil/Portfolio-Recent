import asyncio
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from decouple import config

# Shared state that all agents will use
class AgentState(TypedDict):
    task: str
    research_notes: str
    blog_content: str
    final_output: str
llm = ChatOpenAI(api_key=config("OPENAI_API_KEY"), temperature=0.7)

async def researcher_agent(state: AgentState):
    prompt = f"""You are a research agent. Your task is to gather key facts,
    statistics, and important points about the following topic:
    Topic: {state['task']}
    Provide 5-7 bullet points with relevant information."""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {**state, "research_notes": response.content}

async def writer_agent(state: AgentState):
    prompt = f"""You are a content writer. Using the research notes below,
    create a well-structured blog post (3-4 paragraphs) about:
    Topic: {state['task']}
    Research Notes:
    {state['research_notes']}
    Write engaging, clear content."""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {**state, "blog_content": response.content}

async def editor_agent(state: AgentState):
    prompt = f"""You are an editor. Review and improve the following draft.
    Fix any issues, enhance clarity, and ensure professional quality.
    Original Topic: {state['task']}
    Draft:
    {state['draft_content']}
    Provide the polished final version."""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {**state, "final_output": response.content }

def create_workflow():
    builder = StateGraph(AgentState)
    # Add the three agents nodes
    builder.add_node("researcher", researcher_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("editor", editor_agent)
    # Define the sequential flow: researcher → writer → editor → END
    builder.set_entry_point("researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "editor")
    builder.add_edge("editor", END)
    return builder.compile()

# Execute the multi-agent collaboration
async def main():
    graph = create_workflow()
    # Initial task
    initial_state = { "task": "The benefits of meditation for mental health",
                      "research_notes": "", "draft_content": "", "final_output": "" }
    # Run the workflow
    result = await graph.ainvoke(initial_state)
    # Display results
    print("\n=# RESEARCH NOTES:")
    print(result["research_notes"])
    print("\n=# DRAFT CONTENT:")
    print(result["draft_content"])
    print("\n=# FINAL OUTPUT:")
    print(result["final_output"])

if __name__ == "__main__":
    asyncio.run(main())