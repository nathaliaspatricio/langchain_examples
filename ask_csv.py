##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=xQ3mZhw69bc

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.llms import OpenAI 
from langchain.agents import create_csv_agent

filepath = "database.csv"
agent = create_csv_agent(OpenAI(temperature=0), filepath, verbose=True)

print(agent)

agent.agent.llm_chain.prompt.template
"""
Você está trabalhando com um dataframe chamado pandas em Python. O nome do dataframe é `df`.
Você deve usar as ferramentas abaixo para responder à pergunta feita por você:

python_repl_ast: Use isso para executar comandos python. A entrada deve ser um comando python válido. Ao usar esta ferramenta, às vezes a saída é abreviada - certifique-se de que não pareça abreviada antes de usá-la em sua resposta.

Use o seguinte formato:

Questão: a questão de entrada que você deve responder
Pensamento: você deve sempre pensar no que fazer
Ação: a ação a ser executada deve ser uma das [python_repl_ast]
Entrada para Ação: a entrada para a ação
Observação: o resultado da ação
... (este Pensamento/Ação/Entrada para Ação/Observação pode repetir N vezes)
Pensamento: agora sei a resposta final
Resposta final: a resposta final para a pergunta de entrada original

Esse é o resultado de `print(df.head())`:
{df}

Comece!
Questão: {input}
{agent_scratchpad}
"""

answer = agent.run("Quantas linhas tem no dataframe?")
answer = agent.run("Quantas universidades são nos Estados Unidos?")
answer = agent.run("Qual a posição e o nome da primeira universidade da lista que fica na Suíça?")
answer = agent.run("Há mais universidades dos Estados Unidos nessa lista do que do Reino Unido?")
answer = agent.run("Quais países no continente americano que estão na lista de universidades? Considere países das Américas do Norte, Central e do Sul. Agora mostre o total de universidades que estão nos países listados.")
print(answer)
