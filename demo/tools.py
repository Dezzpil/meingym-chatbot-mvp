from typing import List, Tuple, Set
import requests
import random

from duckduckgo_search import DDGS
from langchain_core.prompts import ChatPromptTemplate
from unstructured.partition.html import partition_html

from demo.structures import Training, Result
from demo.templates import template_role, template_action_queries, user_data_example, template_extract, \
    template_sum_and_extract, template_report, template_check, template_define_muscles
from demo.logging_utils import logger


def generate_muscles_list(llm, query) -> List[str]:
    logger.processing(f"Определение целевых мышечных групп по запросу пользователя")

    prompt = ChatPromptTemplate.from_messages([
        ("system", template_role),
        ("system", template_define_muscles),
        ("human", '{query}'),
    ])

    prompt = prompt.format(query=query)
    logger.info("Отправка запроса к языковой модели...")
    response = llm.invoke(prompt)
    logger.success("Ответ от языковой модели получен")
    muscles = []
    for m in response.content.strip().split(','):
        muscles.append(m.strip())
    logger.success(f"Целевые мышцы определены: {", ".join(muscles)}")
    return muscles


def generate_search_queries(llm, query: str) -> List[str]:
    logger.processing(f"Генерация поисковых запросов на основе запроса: '{query}'")

    prompt = ChatPromptTemplate.from_messages([
        ("system", template_role),
        ("system", template_action_queries),
        ("human", user_data_example), ## получать динамически
        ("human", query),
    ])

    prompt = prompt.format(query=query)
    logger.info("Отправка запроса к языковой модели...")
    response = llm.invoke(prompt)
    logger.success("Ответ от языковой модели получен")

    queries = []
    for q in response.content.strip().split(';'):
        if len(q.strip()) > 3:
            queries.append(q.strip() + ' в зале')

    if queries:
        logger.highlight(f"Сформированы поисковые запросы ({len(queries)}):")
        for i, q in enumerate(queries, 1):
            logger.info(f"  {i}. {q}")
    else:
        logger.warning("Не удалось сформировать поисковые запросы")

    return queries


def search(queries: List[str], max_results_each_q=5, max_urls=5) -> List[Tuple[str, str]]:
    logger.processing(f"Поиск информации по {len(queries)} запросам (максимум {max_results_each_q} результатов на запрос)")

    urls = set()
    searcher = DDGS()

    for i, query in enumerate(queries, 1):
        logger.info(f"Поиск по запросу {i}/{len(queries)}: '{query}'")
        try:
            search_results = searcher.text(query, max_results=max_results_each_q)
            if search_results:
                count_before = len(urls)
                for result in search_results:
                    urls.add((result['title'], result['href']))
                count_added = len(urls) - count_before
                logger.success(f"Найдено {count_added} новых результатов")
            else:
                logger.warning(f"По запросу '{query}' ничего не найдено")

        except Exception as e:
            logger.error(f"Ошибка при поиске по запросу '{query}': {e}")

    if urls:
        logger.highlight(f"Всего найдено уникальных ссылок: {len(urls)}")
        random_urls = list(urls)
        random.shuffle(random_urls)
        logger.highlight(f"Перемешиваем и берем до {max_urls} из найденных")
        return random_urls[:max_urls]
    else:
        logger.warning("Не удалось найти ни одной страницы по всем запросам")
        return []


def load_documents(urls: List[Tuple[str, str]], timeout_seconds = 10, max_len = 5000) -> List[Tuple[str, str, int, str]]:
    logger.processing(f"Загрузка и обработка {len(urls)} страниц")

    docs = []
    successful = 0
    failed = 0

    def load_html_with_timeout(url, timeout=timeout_seconds):
        response = requests.get(url, timeout=timeout)
        html_content = response.text
        return partition_html(text=html_content)

    for i, (title, href) in enumerate(urls, 1):
        logger.info(f"Обработка страницы {i}/{len(urls)}: {title}")
        logger.info(f"  URL: {href}")

        doc = ['#' + title]
        elements = None
        try:
            logger.info(f"  Загрузка HTML (таймаут: {timeout_seconds} сек)...")

            try:
                elements = load_html_with_timeout(href, timeout=timeout_seconds)
                logger.success(f"  HTML успешно загружен и разобран")
            except requests.Timeout:
                logger.warning(f"  Превышен таймаут ({timeout_seconds} сек) при загрузке страницы")
                failed += 1
                continue
        except Exception as e:
            logger.error(f"  Ошибка при загрузке страницы: {str(e)}")
            failed += 1
            continue

        if elements is not None:
            text_elements = 0
            logger.info(f"  Извлечение текста из {len(elements)} элементов...")

            for elem in elements:
                el = elem.to_dict()
                if el['type'] == 'NarrativeText':
                    doc.append(el['text'][:300])
                    text_elements += 1
                elif el['type'] == 'ListItem':
                    doc.append('*' + el['text'][:200])
                    text_elements += 1
                elif el['type'] == 'Title':
                    doc.append('##' + el['text'][:100])
                    text_elements += 1

            if len(doc) <= 1:  # Only title, no content
                logger.warning(f"  Не удалось извлечь полезный текст из страницы")
                failed += 1
                continue

            text = '\n'.join(doc)

            # Ограничиваем длину документа до 6000 символов
            if len(text) > max_len:
                logger.info(f"  Обрезаем текст документа с {len(text)} до {max_len} символов")
                text = text[:max_len]

            docs.append((title, href, len(text), text))
            logger.success(f"  Извлечено {text_elements} текстовых элементов ({len(text)} символов)")
            successful += 1

    if docs:
        logger.highlight(f"Успешно обработано {successful} из {len(urls)} страниц")
        logger.highlight(f"Общий объем собранных данных: {sum(doc[2] for doc in docs)} символов")
    else:
        logger.warning("Не удалось собрать данные ни с одной из страниц")

    return docs


def extract_training_from_documents(llm, docs, query, threshold) -> List[Training]:
    logger.processing(f"Извлечение информации о тренировках из {len(docs)} документов")

    structured_llm = llm.with_structured_output(schema=Training)
    logger.info("Настроена модель для структурированного вывода")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", template_role),
        ("system", template_extract),
        ("human", "{doc}")
    ])

    skipped = []
    lens = []
    trainings = []
    filtered = []

    for i, doc in enumerate(docs, 1):
        title = doc[0]
        logger.info(f"Анализ документа {i}/{len(docs)}: {title}")
        logger.info(f"  Размер документа: {doc[2]} символов")

        prompt = prompt_template.invoke({"doc": doc[3]})
        logger.info(f"  Отправка запроса к языковой модели...")

        result = structured_llm.invoke(prompt)
        logger.info(f"  Ответ от языковой модели получен")

        # отфильтруем результаты с пустым списком упражнений
        dump = result.model_dump()
        if dump['exercises'] and len(dump['exercises']):
            trainings.append(result)
            lens.append(len(str(dump)))

            exercise_count = len(dump['exercises'])
            logger.success(f"  Извлечено упражнений: {exercise_count}")

            if dump['title']:
                logger.info(f"  Название тренировки: {dump['title']}")

            true_ex = 0
            for j, ex in enumerate(dump['exercises'], 1):
                if ex['name'] == 'None': continue

                logger.info(f"    {j}. {ex['name']}")
                true_ex += 1

            if true_ex == 0:
                skipped.append(doc)
                logger.warning(f"  Не удалось извлечь упражнения из документа")
                continue

            if filter_trainings(llm, result, query, threshold):
                filtered.append(result)
                if len(filtered) == 2:
                    break # TODO demo!!
        else:
            skipped.append(doc)
            logger.warning(f"  Не удалось извлечь упражнения из документа")
            continue

    if trainings:
        logger.highlight(f"Успешно извлечены данные о тренировках из {len(trainings)} из {len(docs)} документов")
        total_exercises = sum(len(t.model_dump()['exercises']) for t in trainings)
        logger.highlight(f"Всего извлечено {total_exercises} упражнений")
    else:
        logger.warning("Не удалось извлечь данные о тренировках ни из одного документа")

    diff = len(trainings) - len(filtered)
    if diff > 0:
        logger.info(f"  Отфильтровано {diff} документов, которые плохо соответствуют запросу")
    return filtered


def filter_trainings(llm, training: Training, query: str, threshold: float):
    logger.processing(f"Проверка качества содержания обработанных документов")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", template_role),
        ("system", template_check),
        ("human", "{query}")
    ])

    score = 0
    prompt = prompt_template.invoke({"training": str(training), "query": query })
    result = llm.invoke(prompt)
    r = result.content.strip()
    try:
        score = int(r)
    except Exception:
        pass

    logger.success(f"  Оценка упражнений из документа: {score/10}")
    if score/10 >= threshold:
        return True
    else:
        logger.warning(f"  Упражнения из документа плохо соответствуют запросу")
        return False


def dump_trainings(trainings) -> List[str]:
    logger.processing(f"Форматирование данных о {len(trainings)} тренировках")

    descs = []
    for i, t in enumerate(trainings, 1):
        desc = ''
        t = t.model_dump()

        if t['exercises'] is None or len(t['exercises']) == 0:
            logger.warning(f"Тренировка {i} не содержит упражнений, пропускаем")
            continue

        logger.info(f"Форматирование тренировки {i}/{len(trainings)}")

        if t['title']: 
            desc += '##' + t['title'] + '\n'
            logger.info(f"  Название: {t['title']}")

        if t['description']: 
            desc += 'Комментарий: ' + t['description'] + '\n'
            logger.info(f"  Описание: {t['description'][:50]}..." if len(t['description']) > 50 else f"  Описание: {t['description']}")

        desc += 'Упражнения:\n'
        logger.info(f"  Упражнения ({len(t['exercises'])}):")

        for j, e in enumerate(t['exercises'], 1):
            exercise_line = f" - \"{e['name']}\""

            if e['muscles']:
                exercise_line += f": {', '.join(e['muscles'])}"

            if e['comments']:
                exercise_line += f"; {e['comments']}"

            desc += exercise_line + '\n'

            # Логируем каждое упражнение
            log_line = f"    {j}. {e['name']}"
            if e['muscles']:
                log_line += f" (мышцы: {', '.join(e['muscles'])})"
            logger.info(log_line)

        descs.append(desc)
        logger.success(f"  Тренировка {i} успешно отформатирована ({len(desc)} символов)")

    if descs:
        logger.highlight(f"Успешно отформатировано {len(descs)} тренировок")
        logger.highlight(f"Общий объем данных: {sum(len(d) for d in descs)} символов")
    else:
        logger.warning("Не удалось отформатировать ни одну тренировку")

    return descs

def generate_result(llm, trainings_dumps: List[str]) -> Result:
    logger.processing("Генерация итогового результата на основе собранных данных")

    logger.info(f"Объединение данных из {len(trainings_dumps)} тренировок")
    combined_data_size = len("\n\n".join(trainings_dumps))
    logger.info(f"Общий размер данных для анализа: {combined_data_size} символов")

    structured_result_llm = llm.with_structured_output(schema=Result)

    prompt = ChatPromptTemplate.from_messages([
        ("system", template_sum_and_extract),
        ("human", "\n\n".join(trainings_dumps)),
    ])

    prompt = prompt.format()
    logger.info("Отправка запроса к языковой модели для обобщения данных...")
    response = structured_result_llm.invoke(prompt)
    logger.success("Ответ от языковой модели получен")

    result = response.model_dump()

    if result['exercises'] and len(result['exercises']):
        logger.highlight(f"Сформирована тренировка с {len(result['exercises'])} упражнениями")

        if result['muscles']:
            logger.info(f"Целевые мышцы: {result['muscles']}")

        logger.info("Упражнения в итоговой тренировке:")
        for i, exercise in enumerate(result['exercises'], 1):
            logger.info(f"  {i}. {exercise}")

        if result['comments']:
            comment_preview = result['comments'][:100] + "..." if len(result['comments']) > 100 else result['comments']
            logger.info(f"Комментарий: {comment_preview}")
    else:
        logger.warning("Не удалось сформировать упражнения для итоговой тренировки")

    return response


def summarize(llm, query: str, result: Result):
    logger.processing("Создание итогового отчета для пользователя")

    # Подготовка данных для отчета
    result_data = result.model_dump()
    exercise_count = len(result_data['exercises']) if result_data['exercises'] else 0
    logger.info(f"Количество упражнений в тренировке: {exercise_count}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", template_report),
        ("human", query),
    ])

    prompt = prompt.format(user=user_data_example, context=str(result_data))
    logger.info("Отправка запроса к языковой модели для создания отчета...")

    response = llm.invoke(prompt)
    logger.success("Ответ от языковой модели получен")

    report_length = len(response.content)
    logger.highlight(f"Отчет создан ({report_length} символов)")

    return response
