"""
Build an Extraction Chain

In this tutorial, we will use tool-calling features of chat models to extract structured
information from unstructured text. We will also demonstrate how to use few-shot prompting
in this context to improve performance.

"""
from typing import Optional, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

class Person(BaseModel):
    """Information about a person."""

    name: Optional[str] = Field(default=None, description="The name of person")
    hair_color: Optional[str] = Field(default=None, description="The color of the person's hair if known")
    height_in_meters: Optional[str] = Field(default=None, description="Height measured in meters")


class Data(BaseModel):
    """Extracted data about people."""

    people: List[Person]


llm = ChatOllama(base_url="http://192.168.20.38:11434", model="qwen2.5:7b", temperature=0.5)
structured_llm = llm.with_structured_output(schema=Data)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert extraction algorithm.
Only extract relevant information from the text.
If you do not know the value of an attribute asket to extract, return null for the attribute's value.
    """),
    # Please see the how-to about improving performance with
    # reference examples.
    # MessagesPlaceholder('examples'),
    ("human", "{text}")
])

text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
result = structured_llm.invoke(prompt)

print(result)
