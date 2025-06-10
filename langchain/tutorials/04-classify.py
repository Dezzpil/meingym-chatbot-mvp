"""
Classify Text into Labels

Tagging means labeling a document with classes such as:

    Sentiment
    Language
    Style (formal, informal etc.)
    Covered topics
    Political tendency

Overview

Tagging has a few components:

    function: Like extraction, tagging uses functions to specify how the model should tag a document
    schema: defines how we want to tag the document

"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(base_url="http://192.168.0.17:11434", model="qwen2.5:3b", temperature=0.5)

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template("""
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

# class Classification(BaseModel):
#     sentiment: str = Field(description='The sentiment of the text')
#     aggressiveness: int = Field(description='How aggressive the text is on a scale from 1 to 10')
#     language: str = Field(description='The language the text is written in')
#
# structured_llm = model.with_structured_output(Classification)
#
# inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = structured_llm.invoke(prompt)
#
# print(response)
#
# inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = structured_llm.invoke(prompt)
#
# print(response.model_dump())

class FinerClassification(BaseModel):
    sentiment: str = Field(description='The sentiment of the text', enum=['happy', 'neutral', 'sad'])
    aggressiveness: int = Field(description='describes how aggressive the statement is, the higher the number the more aggressive', enum=[1,2,3,4,5])
    language: str = Field(description='The language the text is written in', enum=["spanish", "english", "french", "german", "italian", 'russian'])

model = ChatOllama(base_url="http://192.168.0.17:11434", model="qwen2.5:3b", temperature=0.1)
structured_llm = model.with_structured_output(FinerClassification)

inp = "Weather is ok here, I can go outside without much more than a coat"
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

print(response.model_dump())