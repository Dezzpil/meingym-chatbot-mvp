from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from research.prompt import prompt_template_case1

case1_query = 'Собери мне тренировку ног на сегодня'

for model in ['qwen2.5:7b', 'llama3.2:latest']:
    llm = Ollama(base_url='http://obione.archive.systems:11434', model=model, temperature=1)

    prompt = ChatPromptTemplate(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(prompt_template_case1),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

    # Example usage: first input
    print(f'Model {model}')
    result = conversation_chain.invoke({"query": case1_query})
    print(result['text'], end='\n\n')
