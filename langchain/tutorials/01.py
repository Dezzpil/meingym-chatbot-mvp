# load environment variables from .env file (requires `python-dotenv`)
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOllama(
    model='qwen2.5:3b',
    base_url='http://192.168.0.17:11434',
    temperature=0.5,
)

system_prompt = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt), ("user", "{text}")
])
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
response = model.invoke(prompt)
print(response.content)