from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import RetrievalQA

from helpers import user_data_str, get_retriever

url = 'http://obione.archive.systems:11434'
model = 'qwen2.5:7b'
temperature = 1.0
llm = Ollama(base_url=url, model=model, temperature=temperature)
query = 'Собери мне тренировку ног на сегодня'


# Step 1
def generate_user_data(user_data: str):
    template = """
    Ты - опытный действующий спортивный тренер, который очень хочет помочь мне стать сильным, здоровым и красивым.
    Твоя задача оценить меня в спортивном плане и сделать небольшое резюме,
    подчеркнув важные моменты в моих данных для составление программы тренировок и выбора упражнений.
    Мои данные: ```{user_data}```
    ###
    Учти важные моменты при составлении ответа:
    * Не делай вступления и заключения, только перечисли важные моменты.
    * В конце добавь вывод одним предложением.
    * Не используй форматирование в ответе.
    * Отвечай только по-русски. 
    * Не используй в ответе китайский, пожалуйста!
    ###
    Важные моменты: 
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["user_data"]
    )
    formatted = prompt.format(user_data=user_data)
    result = llm.invoke(formatted)
    print(f'Характеристика пользователя:\n{result}',end='\n\n')
    return result

user_example = generate_user_data(user_data_str)


# Step 2
def generate_search_phrases(user: str, query: str):
    template = """
    Ты - опытный действующий спортивный тренер, который очень хочет помочь мне стать сильным, здоровым и красивым.
    Твоя задача - сформулируй список из предложений, которые хорошо дополняют и расширяют мой запрос.
    Не выполняй мой запрос, а только прими его во внимание.
    Мой запрос: ```{query}```
    ###
    Пример запроса: "составь мне тренировку"
    Пример ответа: "программа тренировок для общего развития, как правильно тренироваться, примеры тренировок, программа тренировок, упражнения"
    ###
    Учти мои данные в ответе:
    Мои данные: ```{user}```
    ###
    Учти важные моменты при составлении ответа:
    * Не делай вступления и заключения, оставь только перечисление важных моментов.
    * Не используй форматирование в ответе.
    * Отвечай только по-русски. 
    * Не используй в ответе китайский, пожалуйста!
    ###
    Фразы для дальнейшего поиска:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["user", "query"]
    )
    formatted = prompt.format(user=user, query=query)
    result = llm.invoke(formatted)
    print(f'Ключевые фразы:\n{result}', end='\n\n')
    return result


# Step 3 ??
def generate_text_for_rag(phrases):
    template_prompt = """
    Ты - опытный действующий спортивный тренер, который очень хочет помочь мне стать сильным, здоровым и красивым.
    Твоя задача - суммаризировать список фраз в 1-3 предложения, расширив мой запрос.
    Мой запрос: ```{query}```
    Фразы: ```{phrases}```
    ###
    Учти важные моменты при составлении ответа:
    * Не делай вступления и заключения, оставь только перечисление важных моментов.
    * Не используй форматирование в ответе.
    * Отвечай только по-русски. 
    * Не используй в ответе китайский, пожалуйста!
    ###
    Результат:
    """
    prompt = PromptTemplate(
        template=template_prompt,
        input_variables=["phrases", "query"]
    )
    formatted_prompt = prompt.format(phrases=phrases, query=query)
    result = llm.invoke(formatted_prompt)
    print(f'Запрос для RAG:\n{result}')


phrases_example = """
тренировка ног
силовая тренировка
упражнения на ноги
разминка перед тренировкой
профилактика остеохондроза и грыжи в поясничном отделе
выбор упражнений с учетом панкреатита
развитие мышц ног для силовой тренировки
сплит-тренировка
программа тренировок на неделю
регламентирование нагрузок
исключения травмирующих упражнений
"""

phrases_example = """
тренировка ног
силовая тренировка
упражнения на ноги
разминка перед тренировкой
профилактика остеохондроза и грыжи в поясничном отделе
выбор упражнений с учетом панкреатита
развитие мышц ног для силовой тренировки
"""

# перевернул список
phrases_example = """
развитие мышц ног для силовой тренировки
выбор упражнений с учетом панкреатита
профилактика остеохондроза и грыжи в поясничном отделе
разминка перед тренировкой
упражнения на ноги
тренировка ног
силовая тренировка
"""

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Step 4
def generate_exercises_from_rag(phrases: str, user: str):
    # Определяем шаблон для системного сообщения
    system_template = """
    Ты - опытный действующий спортивный тренер, который очень хочет помочь мне стать сильным, здоровым и красивым.
    ###
    Учти важные моменты при составлении ответа:
    * If you don't know the answer, say you don't know.
    * Keep the answer concise.
    * Не делай вступления и заключения.
    * Не используй форматирование в ответе.
    * Отвечай только по-русски. 
    * Не используй в ответе китайский, пожалуйста!
    """

    # Определяем шаблон для человеческого сообщения
    human_template = """
    Собери мне тренировку по ключевым фразам, учитывая контекст и мои данные.
    В ответе укажи только список упражнений.
    ###
    Контекст: ```{context}```
    ###
    Мои данные: ```{user}```
    ###
    Ключевые фразы: ```{input}```
    ###
    Список упражнений:
    """

    # Создаем ChatPromptTemplate с системным и человеческим сообщениями
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # при использовании эластика возникает проблема:
    # последний документ полностью занимает всю память, поэтому
    # даже при обнаружении релевантных документов - в учет принимается только последний
    # и все остальные инструкции размываются
    #retriever = gather_retrievers_ensemble()

    retriever = get_retriever()

    chain = create_retrieval_chain(retriever, question_answer_chain)

    # ИМЕННО input ОБЯЗАТЕЛЬНО, иначе будет ошибка
    result = chain.invoke({"input": phrases, "user": user})
    # input, context: [{metadata, page_content}], answer
    print(f"Ответ: {result['answer']}\n\n")
    print(f"Источники:\n")
    for i, doc in enumerate(result['context']):
        print(f"{i+1}. {doc.metadata}")
    return result


generate_exercises_from_rag(phrases_example, user_example)