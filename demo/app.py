import re
from langchain_ollama import ChatOllama

from demo.client import ApiClient
from demo.logging_utils import logger
from demo.tools import generate_search_queries, search, load_documents, extract_training_from_documents, dump_trainings, \
    generate_result, summarize, generate_muscles_list

from utils import ChatOpenAI
course_api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2N2EwNmU0OGU5OTI3Yjg1N2Q1YTYzOGEiLCJleHAiOjE3NTQwOTI4MDB9.xexTktg1chsl6X9BnA56ACd3b9kYa4qGQorp0CDOWBA'

class App:
    def __init__(self):
        base_url = 'http://192.168.0.24:11434'
        model = 'qwen2.5:7b'
        # model = 'qwen2.5:3b'
        temperature = 0.3

        self.user_id = 'clux8gey10000wiqegntxp3y3'
        self.llm = ChatOllama(base_url=base_url, model=model, temperature=temperature, extract_reasoning=True, request_timeout=20.0)
        # self.llm = ChatOpenAI(
        #     model='gpt-o4-mini',
        #     temperature=0.3,
        #     course_api_key=course_api_key
        # )

    def case1(self, query: str):
        logger.processing(f"Начало обработки запроса: '{query}'")


        muscles = generate_muscles_list(self.llm, query)

        queries = generate_search_queries(self.llm, query)
        if len(queries) == 0:
            logger.error('Не удалось сформулировать поисковые запросы')
            exit(0)

        urls = search(queries, 3, 4)
        if len(urls) == 0:
            logger.error('Не удалось найти ни одной страницы по запросам')
            exit(0)

        docs = load_documents(urls, 5, 8000)
        if len(docs) == 0:
            logger.error('Не удалось собрать данные ни с одной из страниц')
            exit(0)

        trainings = extract_training_from_documents(self.llm, docs, query, 0.5)

        if len(trainings) == 0:
            logger.error('Не удалось обнаружить списка упражнений ни в одном из отобранных документов')
            exit(0)

        dumps = dump_trainings(trainings)
        if len(dumps) == 0:
            logger.error('Во всех описаниях тренировок не указанны упражнения для выполнения')
            exit(0)

        result = generate_result(self.llm, dumps)

        logger.processing("Создание тренировки в приложении...")
        client = ApiClient()
        response = client.create_training_from_llm(self.user_id, result, muscles)
        if response['success'] is False:
            logger.error(f'Возникла ошибка обращения к приложению. {response['message']}')
            exit(1)

        training_path = response['data']['path']
        training_url = client.base_url + training_path
        logger.success(f"Тренировка успешно создана в приложении")

        report = summarize(self.llm, query, result)

        # Выводим финальный отчет
        print(re.sub(r'/(\n)+/', '\n', report.content, re.M))
        logger.success(f'Ссылка на тренировку: {training_url}')
