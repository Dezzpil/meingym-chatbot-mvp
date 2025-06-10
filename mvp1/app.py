from pprint import pprint
from typing import List

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from os import environ
import re

from mvp1.templates import template1, template2, template3

load_dotenv()

class App:
    def __init__(self, translater = None):
        self.translater = translater
        self.model = ChatOllama(base_url=environ.get('MODEL_BASE_URL'), model=environ.get('MODEL_NAME'), temperature=environ.get('MODEL_T'))

        self.templates = [template1, template2, template3]
        if translater:
            for i in range(len(self.templates)):
                self.templates[i] = translater.translate_pipe_ru_to_en(self.templates[i])

        pprint(self.templates)

    def generate_search_queries(self, query: str) -> List[str]:
        queries = []

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.templates[0]),
            ("system", self.templates[1]), ## получать динамически
            ("system", self.templates[2]),
            ("human", '{query}')
        ])
        prompt = prompt.format(query=query)
        response = self.model.invoke(prompt)
        pprint(response)

        match = re.match(r'</think>(.+)$', response.content, flags=re.M)
        if match:
            queries = match.group(1).strip().split('\n')
            if self.translater:
                for i in range(len(queries)):
                    queries[i] = self.translater.translate_pipe_en_to_ru(queries[i])

        return queries


    def search_and_collect_documents(self, queries: List[str]) -> List[str]:
        docs = []



        return docs

