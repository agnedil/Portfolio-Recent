# OpenAI Agents framework
import json
from agents import Agent, Runner 
from pydantic import BaseModel, Field

class Critique(BaseModel):
    valid: bool = Field(description="True if orig. answer already correct, complete, and clear")
    rewrite: str = Field(description="An improved answer if valid is False. Empty string otherwise.")

# 2. Define the two agents
answer_agent = Agent(
    name="AnswerAgent",
    instructions=('You are a knowledgeable assistant, think step‑by‑step',
                   'return final concise answer', ), )

reflection_agent = Agent(
    name="ReflectionAgent",
    instructions=("You are a strict reviewer. Evaluate the answer ...'),
    output_type=Critique, )

def self_reflect(question: str, max_iters: int = 3) -> str:
    answer_result = Runner.run_sync(answer_agent, question)
    for _ in range(max_iters):
        critique_prompt = (f"QUESTION:\n{question}\n\nANSWER:\n{answer_result.final_output}")
        critique_result = Runner.run_sync(reflection_agent, critique_prompt)
        critique = critique_result.final_output_as(Critique)
        if critique.valid:
            return answer_result.final_output
        improvement_prompt = (
            f"The reviewer says the answer needs improvements:\n{critique.rewrite}\n\n"
            f"Write a **better** answer that resolves all issues." )
        answer_result = Runner.run_sync(answer_agent, improvement_prompt)
    return answer_result.final_output    # fallback – the best

user_question = "Explain why the sky appears blue during the day?"
final_answer = self_reflect(user_question)
print(final_answer)