##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=phHqvLHCwH4&t=58s

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

messages=[
    SystemMessage(content="Você é uma professora de história muito didática chamada Maria."),
]

llm = ChatOpenAI(
    model_name = "gpt-3.5-turbo", 
    temperature = 0, 
    max_tokens = 256
)

print(llm(messages).content)

template = """
    Receba a seguinte questão: {user_input}
    Responda a questão de uma forma informativa e interessante, mas concisa para alguém que é novo nesse tópico.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["user_input"]
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

user_input = "O que é inteligência artificial?"

response = llm_chain(user_input)

print(response["text"])
