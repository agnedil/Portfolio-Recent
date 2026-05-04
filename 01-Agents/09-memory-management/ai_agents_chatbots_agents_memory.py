#!/usr/bin/env python
# coding: utf-8

# # LangChain: Prompts, Chatbots, Agents, Memory

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# Chatbot

chat = ChatOpenAI(temperature=0)

# build prompt
template              ="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template        ="{text}"
human_message_prompt  = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt           = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion
chat(chat_prompt.format_prompt( input_language="English",
                                output_language="French",
                                text="I love programming.",
                              ).to_messages()
    )


# ## Chatbot using LLMChain

chat = ChatOpenAI(temperature=0)

# build prompt
template              ="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template        ="{text}"
human_message_prompt  = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt           = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# chaining it all together
chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(input_language="English", output_language="French", text="I love programming.")


# ## Using agents with chat models

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


# load LLM
chat = ChatOpenAI(temperature=0)

# load tools
os.environ('SERPAPI_API_KEY') = 'SERPAPI_API_KEY'             # provide this
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Initialize agent w/tools, LLM, and type of agent
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Test it
agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")


# ## Memory
# Keep previous messages as memory objects, rather than condensing them into a single string

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# ConversationChain - maintain the context of conversation across multiple interactions
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

conversation.predict(input="Hi there!")

conversation.predict(input="I'm doing well! Just having a conversation with an AI.")

conversation.predict(input="Tell me about yourself.")


# Reference
# https://hackernoon.com/a-comprehensive-guide-to-langchain-building-powerful-applications-with-large-language-models