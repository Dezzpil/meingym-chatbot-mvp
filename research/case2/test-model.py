from langchain_community.llms import Ollama

url = 'http://192.168.0.17:11434'
model = 'qwen3:8b'
temperature = 1.0
llm = Ollama(base_url=url, model=model, temperature=temperature)

query1 = """
Что такое мышцы-агонисты, синергисты и антагонисты. Расскажи коротко
"""
query2 = """
Расскажи какие мышцы работают в процессе выполнения упражнения становая тяга? 
Какие агонисты, синергисты, антогонисты?
"""
result = llm.invoke(query2)
print(result)
