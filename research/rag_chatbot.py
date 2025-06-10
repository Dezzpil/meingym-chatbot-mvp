from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

from prompt import PROMPT

# Инициализация модели
llm = Ollama(base_url='http://obione.archive.systems:11434', model="qwen2.5:7b", temperature=0.3)

# Загрузка векторной БД
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_index_all-MiniLM-L6-v2", embedding_model, allow_dangerous_deserialization=True)

# Создание RAG цепочки
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Интерактивный чат
print("RAG Chatbot запущен. Введите 'exit' для выхода.")
while True:
    query = input("\nВопрос: ")
    if query.lower() == 'exit':
        break

    result = rag_chain.invoke({"query": query})

    print(f"\nОтвет: {result['result']}")
    print("\nИсточники:")
    for i, doc in enumerate(result['source_documents']):
        print(f"{i+1}. {doc.metadata['title']} | {doc.metadata.get('url', 'N/A')}")