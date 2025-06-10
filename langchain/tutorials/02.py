# Build a semantic search engine

# This tutorial will familiarize you with LangChain's document loader, embedding,
# and vector store abstractions. These abstractions are designed to support retrieval of data--
# from (vector) databases and other sources-- for integration with LLM workflows.
# They are important for applications that fetch data to be reasoned over as part
# of model inference, as in the case of retrieval-augmented generation,
# or RAG (see our RAG tutorial here).

# Here we will build a search engine over a PDF document. This will allow us to retrieve
# passages in the PDF that are similar to an input query.

# Concepts

# This guide focuses on retrieval of text data. We will cover the following concepts:
# Documents and document loaders;
# Text splitters;
# Embeddings;
# Vector stores and retrievers.

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

load_dotenv()

documents = [
    Document(
        page_content='Dogs are great companions, known for their loyalty and friendliness.',
        metadata={'source': 'pets-docs'}
    ),
    Document(
        page_content='Cats are independent pets that often enjoy their own space.',
        metadata={'source': 'pets-docs'}
    )
]

loader = PyPDFLoader(file_path='./nke-10k-2023.pdf')
docs = loader.load()
len(docs)

text_splitter = RecursiveCharacterTextSplitter(
    add_start_index=True,
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(docs)
len(all_splits)

print(all_splits[100])

embeddings = OllamaEmbeddings(
    model='qwen2.5:3b',
    base_url='http://192.168.0.17:11434',
)

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(all_splits)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

from typing import List, Tuple
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> List[Tuple[Document, float]]:
    return vector_store.similarity_search_with_score(query, k=1)


results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print('Custom retriever')
print(results)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.8},
)

results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print('Build-in retriever')
print(results)

# Look further
# https://python.langchain.com/docs/tutorials/retrievers/