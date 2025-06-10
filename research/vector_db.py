import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Загрузка данных
df = pd.read_csv('../meingym_sport_20250530_prepared.csv')
df = df[df['wordCount'] > 20]
df['full_text'] = df['title'] + ". " + df['content']

# Разбиение текста на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1250,
    chunk_overlap=250,
    length_function=len,
    separators=["\n\n", "\n", ".", ""]
)

chunks = []
for _, row in tqdm(df.iterrows()):
    try:
        text_chunks = text_splitter.split_text(row['full_text'])
    except Exception as e:
        print(row['full_text'], row['title'], row['content'])
        break

    for chunk in text_chunks:
        chunks.append({
            "text": chunk,
            "metadata": {
                "title": row['title'],
                "url": row['url'],
                "category": row['category']
            }
        })

# Создание эмбеддингов
embedding_model = HuggingFaceEmbeddings(
    model_name="sergeyzh/rubert-mini-frida",
)

# Создание векторной БД
vector_db = FAISS.from_texts(
    texts=[chunk["text"] for chunk in chunks],
    embedding=embedding_model,
    metadatas=[chunk["metadata"] for chunk in chunks],

)

# Сохранение индекса
vector_db.save_local("faiss_index_rubert-mini-frida")