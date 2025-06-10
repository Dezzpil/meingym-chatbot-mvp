from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchRetriever
from langchain.retrievers import EnsembleRetriever

from helpers import bm25_query

query = 'Упражнение для ног, как накачать ноги, тренировка ног'

print('ElasticSearchBM25Retriever:')

elasticsearch_retriever = ElasticsearchRetriever.from_es_params(
    index_name="docs",
    body_func=bm25_query,
    content_field="content",
    url="http://localhost:9200",
)
for doc in elasticsearch_retriever.invoke(query):
    print(doc)

print('\nFAISS:')

embedding_model = HuggingFaceEmbeddings(model_name="sergeyzh/rubert-mini-frida")
faiss_vectorstore = FAISS.load_local("../faiss_index_rubert-mini-frida", embedding_model, allow_dangerous_deserialization=True)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={ "k": 5, "fetch_k": 50 })

for doc in faiss_retriever.invoke(query):
    print(doc.metadata)


# ensemble
print('\nEnsemble:')

ensemble_retriever = EnsembleRetriever(
    retrievers=[elasticsearch_retriever, faiss_retriever],
    weights=[0.3, 0.7]
)

for doc in ensemble_retriever.invoke(query):
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-----")

