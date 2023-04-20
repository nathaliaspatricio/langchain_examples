##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=J_0qvRt4LNk

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate 
from langchain.chains import LLMChain

llm = OpenAI(model_name = "text-davinci-003", temperature = 0.9, max_tokens = 256)

##Primeiramente, criamos uma lista com pequenos exemplos
examples = [
    {"word": "feliz", "antonym": "triste"},
    {"word": "alto", "antonym": "baixo"}
] 

##Depois, especificamos o template para formatar os exemplos que fornecemos
##Usamos a classe "PrompTemplate" para isso
example_formatter_template = """
Palavra: {word}
Antônimo: {antonym}\n
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

##Finalmente, criamos o objeto "FewShotPromptTemplate"
few_shot_prompt = FewShotPromptTemplate(
    ##Esses são os exemplos que desejamos inserir no prompt.
    examples=examples,
    ##Essa é a forma como desejamos formatar os exemplos quando os inserimos no prompt.
    example_prompt=example_prompt,
    ##O prefixo é algum texto que vem antes dos exemplos no prompt.
    ##Normalmente, consiste em instruções. 
    prefix="Dê o antônimo de cada entrada",
    ##O sufixo é algum texto que vem depois dos exemplos no.
    ##Normalmente, é onde a entrada do usuário virá.
    suffix="Palavra: {input}\nAntônimo:",
    ##As variáveis de entrada são as variáveis que, em geral, o prompt espera.
    input_variables=["input"],
    ##O example_separator é a string que usamos para juntar o prefixo, exemplos e o sufixo.
    example_separator="\n"
)

##Agora podemos gerar um prompt usando o método "format"
print(few_shot_prompt.format(input="grande"))

##consultando o modelo com o Few Shot Prompt Template
chain = LLMChain(llm=llm, prompt=few_shot_prompt)
print("\nPequeno é o antônimo de" + chain.run("Pequeno:"))
