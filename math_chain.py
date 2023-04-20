##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=hI2BY7yl_Ac

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.llms import OpenAI
from langchain.chains import PALChain

llm = OpenAI(model_name = "text-davinci-003", temperature = 0, max_tokens = 512)

pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question = "João tem três vezes mais o número de animais do que Márcia. Márcia tem dois animais a mais que Maria. Se Maria tem 2 animais, quantos animais tem João?"
question_02 = "A cafeteria tem 23 maçãs. Se eles usarem 20 para o almoço e comprarem mais 6, quantas maçãs eles tem?"

result = pal_chain.run(question)
result_02 = pal_chain.run(question_02)
print(result)
print(result_02)
