##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=J_0qvRt4LNk

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(model_name = "text-davinci-003", temperature = 0.9, max_tokens = 256)

restaurant_template = """
Eu quero atuar como um consultor de nomes para novos restaurantes.
Retorne uma lista de nomes de restaurantes. Cada nome deve ser curto, cativante e fácil de lembrar. Ele deve ser relacionado com o tipo de restaurante que se está nomeando.
Quais são bons nomes para um restaurante que é {restaurant_description}? 
"""

#Um prompt de exemplo com uma variável de entrada
prompt_template = PromptTemplate(
    input_variables=["restaurant_description"],
    template = restaurant_template
)

description = "Um lugar grego que serve souvlakis de cordeiro fresco e outras comidas gregas."
description_02 = "Um lugar de hamburguer que é decorado com recordações de baseball."
description_03 = "Um café que tem hard rock ao vivo e decoração nessa temática."

##consultando o modelo com o prompt template
chain = LLMChain(llm=llm, prompt=prompt_template)

##rodando o chain especificando apenas as variáveis de entrada
print(description + "\n" + chain.run(description) + "\n")
print(description_02 + "\n" + chain.run(description_02) + "\n")
print(description_03 + "\n" + chain.run(description_03) + "\n")

