#!/usr/bin/env python
# coding: utf-8

# AI Agents

# !pip install pydantic==2.5.3 --q;
# !pip install langchain-core==0.2.40 langchain==0.2.16 langchain-community --q ;
# !pip install langchain-google-genai==1.0.10 --q;
# !pip install chromadb faiss-cpu pypdf duckduckgo-search --q;


import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ['GOOGLE_API_KEY'] = user_secrets.get_secret("google_api_key")
import requests
import json
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish


tools = []
llm   = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.8)


# Tools


# SIMPLE CALCULATOR
def calculator(expr: str) -> str:
    try:
        allowed = set('0123456789+-*/()%. ')
        if not all(c in allowed for c in expr):
            return 'Error : Invalid character in expression'
        res = eval(expr)
        return f'The answer is {res}'
    except Exception as e:
        return f'Error Calculating "{expr}"'
    
    
calc_tool = Tool(
    name = 'Calculator',
    description = 'Use this for mathematical calculations with numbers, make sure input is a valid expression like "2+4/3-(42)" and "21%7" etc.',
    func = calculator, )


# TEXT ANALYZER
def text_analyze(text: str)-> str:
    stats = { 'word_count' : len(text.split()),        # ok for our demo
              'sentence_count' : len(text.split('.')),
              'character_count' : len(text), }
    return ''.join(f'\n\t{k} : {v}' for k, v in stats.items())

text_analysis_tool = Tool(
    name = 'Text Analyzer',
    description = 'Analyze text and provide statistics.',
    func = text_analyze, )


# CURRENT TIME
def get_time(timezone: str = 'UTC')-> str:
    dt = datetime.now()
    dt = dt.strftime('%d-%m-%Y %H:%M:%S')
    return f'Current timezone {timezone}, time : {dt}'

datetime_tool = Tool(
    name = 'Date and Time',
    description = 'Get the current date and time. Input can be a timezone (default in UTC)',
    func = get_time, )


# UNIT CONVERSION
def unit_conversion(conversion : str)-> str:
    conversion = conversion.lower().strip()

    conversions = {
            "celsius to fahrenheit": lambda c: f"{c}°C = {c * 9/5 + 32}°F",
            "fahrenheit to celsius": lambda f: f"{f}°F = {(f - 32) * 5/9}°C",
            "km to miles": lambda km: f"{km} km = {km * 0.621371} miles",
            "miles to km": lambda miles: f"{miles} miles = {miles * 1.60934} km",
            "kg to lbs": lambda kg: f"{kg} kg = {kg * 2.20462} lbs",
            "lbs to kg": lambda lbs: f"{lbs} lbs = {lbs * 0.453592} kg"
        }

    try:
        # standardizing str to : {source_val} {source_units} to {target_units}
        conv = conversion.split(' ')
        if len(conv) >= 4:
            src_value = float(conv[0])
            conv_command = " ".join(conv[1:])
            if conv_command in conversions:
                return conversion[conv_command](src_val)
        else:
            return 'use "{source_val} {source_units} to {target_units}" format for conversion'
    except Exception as e:
        return f'Error in Conversion {e}'
    
conversion_tool = Tool(
    name = 'Unit Converter',
    description = 'convert between common units. FORMAT : source_val source_units to target_units',
    func = unit_conversion, )

tools.append(calc_tool, text_analysis_tool, datetime_tool, conversion_tool)
for t in tools:
    print(f'{t.name} : {t.description}\n')


# Creating Agents

# LangChain agent's structure is designed for dynamic decision-making and interaction with the world around it.
# It's fundamentally different from a chain, which follows a predefined sequence of actions. 
# 
# Here are the major elemeents: 
# + **Language Model (LLM)**: This is the brain of the agent, responsible for understanding inputs, reasoning, and deciding the next action.
# + **Tools**: These are external resources, APIs, or functions that the agent can use to perform specific tasks, like searching the web, performing calculations, or interacting with a database.
# + **Agent Executor**: This component manages the agent's actions, executes them through the tools, and incorporates the results back into the agent's reasoning process


# Zero-shot React Agent (most common)

from langchain.agents import AgentType, initialize_agent

react_agent = initialize_agent(
    tools = tools,
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    max_iterations = 3,
    early_stopping_method = 'generate'
)

queries = [
    "What's 15% of 250?",
    "Convert 30 celsius to fahrenheit",
    "Analyze this text for word, character and sentence count: 'Machine learning is revolutionizing how we process data and make predictions.'",
    "What's the current time and what's 100 divided by 4?",
    "Convert 5 miles to km and then calculate 20% of that result"
]

for q in queries:
    if len(q)>150:
        q = q[:150]
    print(f'Ques : {q}')
    agent_res = react_agent.invoke(q)
    print('\n\n')


# Conversational Agent With Memory

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key = 'chat_history')

conv_agent = initialize_agent(
    tools = tools,
    llm = llm,
    memory = memory,
    agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose = True
)

res1 = conv_agent.invoke('What do you get from 2105+2120/3102? use the calculator tool.')
res2 = conv_agent.invoke('What is 20% of that value?')
for k, v in res2.items():
    print(f'{k} : {v}\n')


# Custom Agent with Structured Tools
# Instead of using pre defined tooling that takes in a single argument we can create a StructuredTool object to add flexibility and better structure to the agent tool

from langchain.tools import StructuredTool
from typing import Optional
from pydantic import BaseModel

class WeatherInp(BaseModel):
    loc: str
    units: Optional[str] = "metric"

def get_weather(loc: str, units: str = 'metric')-> str:
    data = {
        'bhopal' : {'temp': 38, "condition": 'sunny'},
        'berlin' : {'temp': 22, "condition": 'windy'},
        'austin' : {'temp': 19, "condition": 'cloudy'},
        'tokyo' : {'temp': 28, "condition": 'snowy'},
    }

    loc = loc.lower()
    if loc in data:
        t_data = data[loc]
        condition = t_data['condition']
        t_units = 'C' if units == 'metric' else 'F'
        temp = t_data['temp'] if units == 'metric' else t_data['temp']*9/5 + 32
        return f'Weather in {loc}: {temp}{t_units} - {condition}'
    else:
        return f'Weather for {loc} is not available.'

weather_tool = StructuredTool.from_function(
    func = get_weather,
    name = 'WeatherTool',
    description = 'Get Weather of a location, provide location and optionally the units (eg: metric or imperial)',
    args_schema = WeatherInp   # how to structure input (like the pydantic base model class)
)

s_tools = tools + [weather_tool]

s_agent = initialize_agent(
    tools = s_tools,
    llm = llm,
    agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)

res = s_agent.invoke('How is the weather in Bhopal?')


# Simple Websearch Agent
# Create an agent that can search the web for latest information

import os
from langchain.agents import AgentType, initialize_agent
from langchain.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

search_agent = initialize_agent(
    tools = [search_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)

res = search_agent.invoke("What is the current Internet sentiment on Takopi's original sin")


# Websearch Agent Enhanced with Custom Tools

from langchain.tools import Tool
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub

memory = ConversationBufferMemory()

class WebSearch:
    def __init__(self):
        self.engine = DuckDuckGoSearchRun()

    def seacrh(self, query : str)-> str:
        try:
            res = self.engine.invoke(query)
            return f'Search Results for query {query} : \n{res}'
        except Exception as e:
            return f'Seacrch Failed {str(e)}'

search = WebSearch()

search_tool = Tool(
    name = 'Search Web',
    description = 'Search the Internet for current information, used it in case information about current or recent events are needed.',
    func = search.seacrh
)

prompt_temp = hub.pull('hwchase17/structured-chat-agent')


agent = create_structured_chat_agent(
    llm = llm,
    tools = [search_tool],
    prompt = prompt_temp
)

agent_exec = AgentExecutor(
    agent = agent,
    tools = [search_tool],
    memory = memory,
    handle_parsing_error = True,
    verbose = True
)

res = agent_exec.invoke({'input':"What is the current Internet sentiment on Takopi's original sin"})

# Now that's how you make simple Agents - we can make the LLM run scripts programmatically, i.e., we can call APIs, make resources available and much more. Sky is the limit!