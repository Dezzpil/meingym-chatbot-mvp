from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pprint import pprint

# Инициализация модели
llm = Ollama(base_url='http://obione.archive.systems:11434', model="qwen2.5:7b", temperature=0.3)

# Загрузка векторной БД
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_index_all-MiniLM-L6-v2", embedding_model, allow_dangerous_deserialization=True)

texts = [
    'Подскажи какие существуют базовые упражнения и почему они так называются?',
    'Расскажи, как избавиться от лишнего веса?',
    'Посоветуй программу тренировок для роста мышечной массы'
]

# for doc, score in vector_db.similarity_search_with_relevance_scores(texts[0], k=15):
#     print(doc.metadata)
#     print(doc.page_content)
#     print(score)
#     print('\n')

# for doc, score in vector_db.similarity_search_with_score(texts[2], k=5, fetch_k=50):
#     pprint(doc.metadata)
#     pprint(doc.page_content)
#     print(score)

for doc in vector_db.similarity_search('занятия с гирями', k=5, fetch_k=30):
    print(doc)