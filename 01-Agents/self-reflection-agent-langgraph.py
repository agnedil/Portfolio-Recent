import asyncio
from langgraph.graph import END, MessageGraph
from typing import List, Sequence
from langgraph.graph import MessageGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from decouple import config
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [("system",
    """You are an AI assistant researcher tasked with researching on a variety of topics in a short summary of 5 pa
    " generate the best research possible as per user request.",
    " You have access to a senior researcher who can answer followup questions or your previous attempts."""),
    MessagesPlaceholder(variable_name="messages"), ])
llm = ChatOpenAI(api_key=config("OPENAI_API_KEY"))
gen_chain = prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(
    [("system",
    "You are a senior researcher"
    "Provide detailed recommendations to an assistant researcher to help improve this researches",),
    MessagesPlaceholder(variable_name="messages"),])
reflect = reflection_prompt | llm

async def generation_node(state: Sequence[BaseMessage]):
    return await gen_chain.ainvoke({"messages": state})

async def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Other choice: Critique the most recent generation
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [messages[0]] + [cls_map[msg.type](content=msg.content) for msg in messages[1:] ]
    res = await reflect.ainvoke({"messages": translated})
    # this will be treated as normal feedback in the next generation
    return HumanMessage(content=res.content)


builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")

def should_continue(state: List[BaseMessage]):
    if len(state) > 6: return END    # End after 5 iterations
    return "reflect"

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()

async def stream_responses():
    async for event in graph.astream(
        [HumanMessage(content="Research on climate change" )], ):
        print(event, "\n---")

asyncio.run(stream_responses())