from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Schema for reviewer output
class Critique(BaseModel):
    valid: bool = Field(description="True if orig. answer already correct, complete, and clear")
    rewrite: str = Field(description="An improved answer if valid is False. Empty string otherwise.")

critique_parser = PydanticOutputParser(pydantic_object=Critique)

# Build the two chains
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
answer_prompt = PromptTemplate.from_template( 'You are ...' )
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

reflection_prompt = PromptTemplate(
    template=(
        "You are a strict reviewer. Evaluate the answer ...'
        f"Respond ONLY with JSON matching this schema:\n{critique_parser.get_format_instructions()}"),
    input_variables=["question", "answer"],
    output_parser=critique_parser, )
reflection_chain = LLMChain(llm=llm, prompt=reflection_prompt, output_parser=critique_parser)

def self_reflect(question: str, max_iters: int = 3) -> str:
    answer = answer_chain.run(question=question).strip()
    for _ in range(max_iters):
        critique = reflection_chain.run(question=question, answer=answer)
        if critique.valid:
            return answer  
        answer = answer_chain.run(
            question=f"The reviewer says to improve the answer like this:\n{critique.rewrite}"
        ).strip()
    return answer

q = "Explain why the sky appears blue during the day?"
final = self_reflect(q)
print("\nFinal answer after reflection:\n", final)