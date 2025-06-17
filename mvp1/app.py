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
        self.model = ChatOllama(
            base_url=environ.get('MODEL_BASE_URL'),
            model=environ.get('MODEL_NAME'),
            temperature=environ.get('MODEL_T'),
            extract_reasoning=True,
            request_timeout=60.0  # Timeout in seconds
        )

        self.templates = [template1, template2, template3]
        if self.translater:
            for i in range(len(self.templates)):
                self.templates[i] = translater.from_ru_to_en(self.templates[i])

        pprint(self.templates)

    def generate_search_queries(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.templates[0]),
            ("system", self.templates[1]), ## получать динамически
            ("system", self.templates[2]),
            ("system", 'Write in english!' if self.translater else 'Пиши по-русски!'),
            ("human", '{query}')
        ])
        prompt = prompt.format(query=query)
        response = self.model.invoke(prompt)
        pprint(response)

        match = re.match(r'.*<\/think>(.+)$', response.content, flags=re.M)
        if match:
            queries = match.group(1).strip().split('\n')
        else:
            queries = response.content.strip().split('\n')

        if self.translater:
            for i in range(len(queries)):
                queries[i] = self.translater.from_en_to_ru(queries[i])

        return queries


    def search_and_collect_documents(self, queries: List[str]) -> List[str]:
        docs = []
        search_tool = DuckDuckGoSearchResults()

        for query in queries:
            try:
                search_results = search_tool.invoke(query)
                if search_results:
                    # Extract URLs from the search results
                    urls = []
                    for result in search_results.split('\n'):
                        if result.startswith('['):
                            # Extract URL from format like "[title](url)"
                            url_match = re.search(r'\[(.*?)\]\((.*?)\)', result)
                            if url_match and url_match.group(2):
                                urls.append(url_match.group(2))

                    docs.extend(urls)
            except Exception as e:
                print(f"Error searching for query '{query}': {e}")

        # Remove duplicates while preserving order
        unique_docs = []
        for doc in docs:
            if doc not in unique_docs:
                unique_docs.append(doc)

        return unique_docs
