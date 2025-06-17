from langchain_ollama import ChatOllama

from demo.client import ApiClient
from demo.tools import generate_search_queries, search, load_documents, extract_training_from_documents, dump_trainings, \
    generate_result, summarize


class App:
    def __init__(self):
        base_url = 'http://192.168.0.24:11434'
        model = 'qwen2.5:7b'
        temperature = 0.5

        self.user_id = 'clux8gey10000wiqegntxp3y3'
        self.llm = ChatOllama(base_url=base_url, model=model, temperature=temperature, extract_reasoning=True, request_timeout=20.0)

    def case1(self, query: str):
        queries = generate_search_queries(self.llm, query)
        if len(queries) == 0:
            print(f'Не удалось сформулировать поисковые запросы')
            exit(0)

        urls = search(queries, 5)
        if len(urls) == 0:
            print(f'Не удалось найти ни одной страницы по запросам')
            exit(0)

        docs = load_documents(urls)
        if len(docs) == 0:
            print(f'Не удалось собрать данные ни с одной из страниц')
            exit(0)

        trainings = extract_training_from_documents(self.llm, docs)

        dumps = dump_trainings(trainings)
        if len(dumps) == 0:
            print(f'Не удалось собрать данные ни об одной тренировке')
            exit(0)

        result = generate_result(self.llm, dumps)

        client = ApiClient()
        response = client.create_training_from_llm(self.user_id, result)
        if response.success is False:
            print(f'Возникла ошибка обращения к приложению. {response.message}')
            exit(1)

        training_path = response.data['path']
        training_url = client.base_url + training_path

        report = summarize(self.llm, query, result)
        print(f'{report}\nТренировка создана: {training_url}')




