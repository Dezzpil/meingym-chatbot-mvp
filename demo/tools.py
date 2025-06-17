from typing import List, Tuple, Set

from duckduckgo_search import DDGS
from langchain_core.prompts import ChatPromptTemplate
from unstructured.partition.html import partition_html

from demo.structures import Training, Result
from demo.templates import template_role, template_action_queries, user_data_example, template_extract, \
    template_sum_and_extract, template_report


def generate_search_queries(llm, query: str) -> List[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", template_role),
        ("system", template_action_queries),
        ("human", user_data_example), ## получать динамически
        ("human", query),
    ])

    prompt = prompt.format(query=query)
    response = llm.invoke(prompt)

    queries = []
    for q in response.content.strip().split(';'):
        queries.append(q.strip() + ' в зале')

    return queries


def search(queries: List[str], max_results_each_q=5) -> Set[Tuple[str, str]]:
    urls = set()
    searcher = DDGS()
    for query in queries:
        try:
            search_results = searcher.text(query, max_results=max_results_each_q)
            if search_results:
                for result in search_results:
                    urls.add((result['title'], result['href']))

        except Exception as e:
            print(f"Error searching for query '{query}': {e}")

    return urls


def load_documents(urls) -> List[Tuple[str, str, int, str]]:
    docs = []
    for title, href in urls:
        doc = ['#' + title]
        elements = None
        try:
            elements = partition_html(url=href)
        except Exception as e:
            #print('Exception while parsing and partition:' + str(e))
            continue

        if elements is not None:
            for elem in elements:
                el = elem.to_dict()
                if el['type'] == 'NarrativeText':
                    doc.append(el['text'])
                elif el['type'] == 'ListItem':
                    doc.append('*' + el['text'])
                elif el['type'] == 'Title':
                    doc.append('##' + el['text'])
            if len(doc) == 0:
                continue
            text = '\n'.join(doc)
            docs.append((title, href, len(text), text))
    return docs


def extract_training_from_documents(llm, docs) -> List[Training]:
    structured_llm = llm.with_structured_output(schema=Training)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", template_role),
        ("system", template_extract),
        ("human", "{doc}")
    ])

    skipped = []
    lens = []
    trainings = []
    for doc in docs:
        prompt = prompt_template.invoke({"doc": doc[3]})
        result = structured_llm.invoke(prompt)

        # отфильтруем результаты с пустым списком упражнений
        dump = result.model_dump()
        if dump['exercises'] and len(dump['exercises']):
            trainings.append(result)
            lens.append(len(str(dump)))
        else:
            skipped.append(doc)
            continue

    return trainings


def dump_trainings(trainings) -> List[str]:
    descs = []
    for t in trainings:
        desc = ''
        t = t.model_dump()
        if t['exercises'] is None or len(t['exercises']) == 0:
            continue

        if t['title']: desc += '##' + t['title'] + '\n'
        if t['description']: desc += 'Комментарий: ' + t['description'] + '\n'

        desc += 'Упражнения:\n'
        for e in t['exercises']:
            desc += ' - "' + e['name'] + '"'
            if e['muscles']:
                desc += ': ' + ', '.join(e['muscles']) + '; '
            if e['comments']:
                desc += '; ' + e['comments']
            desc += '\n'

        descs.append(desc)

    return descs


def generate_result(llm, trainings_dumps: List[str]) -> Result:
    structured_result_llm = llm.with_structured_output(schema=Result)
    prompt = ChatPromptTemplate.from_messages([
        ("system", template_sum_and_extract),
        ("human", "\n\n".join(trainings_dumps)),
    ])

    prompt = prompt.format()
    response = structured_result_llm.invoke(prompt)
    return response


def summarize(llm, query: str, result: Result):
    prompt = ChatPromptTemplate.from_messages([
        ("system", template_report),
        ("human", query),
    ])

    prompt = prompt.format(user=user_data_example, context=str(result.model_dump()))
    response = llm.invoke(prompt)
    return response

