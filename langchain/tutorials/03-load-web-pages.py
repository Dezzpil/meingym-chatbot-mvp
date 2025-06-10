# How to load web pages
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_unstructured import UnstructuredLoader

load_dotenv()

url = 'https://fitseven.ru/myschtsy/atlas-uprajneyniy/kak-nakachat-myshtsy-nog'

loader = UnstructuredLoader(web_url=url)

docs = []
for doc in loader.lazy_load():
    if doc.metadata['category'] in ['NarrativeText', 'Title']:
        print(doc)
    docs.append(doc)

print(len(docs))

