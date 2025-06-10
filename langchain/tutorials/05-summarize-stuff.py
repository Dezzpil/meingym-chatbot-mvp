from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

llm = ChatOllama(base_url='http://192.168.0.18:11434', model='qwen2.5:3b', temperature=0.5)

loader = WebBaseLoader(web_path="https://musclefit.info/programma-trenirovok-na-nogi/")
docs = loader.load()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "write a concise summary of the following: \\n\\n{context}")
# ])
#
# chain = create_stuff_documents_chain(llm, prompt)
# result = chain.invoke({ 'context': docs })
# print(result)

# https://fitseven.ru/myschtsy/atlas-uprajneyniy/kak-nakachat-myshtsy-nog
# https://musclefit.info/programma-trenirovok-na-nogi/

from langchain import hub

map_prompt = hub.pull("rlm/map-prompt")
print(map_prompt)

reduce_prompt = hub.pull("rlm/reduce-prompt")
print(reduce_prompt)

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500, strip_whitespace=True)
split_docs = text_splitter.split_documents(docs)
print(len(split_docs))


