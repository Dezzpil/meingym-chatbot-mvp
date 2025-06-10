def bm25_query(search_query: str):
    return {
        "size": 3,
        "query": {
            "multi_match": {
                "query": search_query,
                "fields": ["title", "content"]
            }
        },
    }


user_data_str = """
* Мужчина 30 лет, 175 см рост, 180 кг вес.
* Указал следующие проблемы: "лишний вес, панкреатит, остеохондроз, грыжа в поясничном отделе"
* Описал свой спортивный опыт и физическую форму: "в детстве играл в футбол, катаюсь на велосипеде"
* Начал заниматься в приложении 1 месяц и 5 дней назад и за это время выполнил 10 тренировок.
* Вернулся к активным занятиям 5 дней назад.
* Указал развитие силы (strength) как главную цель тренировок.
* Предпочитает заниматься по формату сплит (split).

История тренировок в текущем тренировочном периоде:
Выполнил 2 тренировки в рамках активного плана.

1. Тренировка 1 день назад. 
Выполнил 5 упражнений: .... 
В рамках выполнения были задействованы мышцы разных групп в качестве мышц-агонистов: 
* Руки: 6 раз
* Поясницы: 2 раз 
* Плеч: 2 раза
* Ноги: 1 раз

2. Тренировка 3 день назад. 
Выполнил 7 упражнений: .... 
В рамках выполнения были задействованы мышцы разных групп в качестве мышц-агонистов: 
* Ноги: 6 раз
* Спина: 3 раз
"""

user_data_str_empty = """
* Женщина 35 лет, 164 см рост, 80 кг вес.
* Указала следующие проблемы: "..." # можно заменить на "Не указала".
* Описала свой спортивный опыт и физическую форму: "..." # можно заменить на нет опыта
* Еще не начала заниматься в приложении.
* Указала снижение веса (weight loss) как главную цель тренировок. # default
* Предпочитает заниматься по формату фулл-бади (full-body). # default

История тренировок в текущем тренировочном периоде:
Еще не начала заниматься.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchRetriever
from langchain.retrievers import EnsembleRetriever


def gather_retrievers_ensemble():
    elasticsearch_retriever = ElasticsearchRetriever.from_es_params(
        index_name="docs",
        body_func=bm25_query,
        content_field="content",
        url="http://localhost:9200",
    )

    embedding_model = HuggingFaceEmbeddings(model_name="sergeyzh/rubert-mini-frida")
    faiss_vectorstore = FAISS.load_local(
        "../../faiss_index_rubert-mini-frida",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 50})

    ensemble = EnsembleRetriever(
        retrievers=[elasticsearch_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    return ensemble

def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sergeyzh/rubert-mini-frida")
    faiss_vectorstore = FAISS.load_local(
        "../../faiss_index_rubert-mini-frida",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 50})
    return faiss_retriever