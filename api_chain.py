##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=hI2BY7yl_Ac

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.llms import OpenAI
from langchain.chains.api.prompt import API_RESPONSE_PROMPT

from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.api import open_meteo_docs

llm = OpenAI(temperature = 0, max_tokens = 100)
chain_new = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)

temperature = chain_new.run("Qual é a temperatura agora na cidade de São Paulo em graus Celsius?")
print(temperature)

time = chain_new.run("Que horas são agora na cidade de São Paulo?")
print(time)

lat_long = chain_new.run("Qual é a latitude e a longitude da cidade de São Paulo? Responda em português.")
print(lat_long)
