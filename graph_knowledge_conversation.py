##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=X550Zbz_ROE

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.chains.conversation.memory import ConversationKGMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

##NetworkX Entity Graph
import networkx as nx
import matplotlib.pyplot as plt

llm = OpenAI(
    model_name = "text-davinci-003", 
    temperature = 0, 
    max_tokens = 256
) 

template = """
    O que vem a seguir é uma conversa amigável entre um humano e uma IA. A IA é falante e fornece muitos detalhes específicos do seu contexto. Se a IA não sabe a resposta para uma questão, ela diz verdadeiramente que não sabe. A IA SOMENTE usa informações contidas na seção "Informações Relevantes" e não alucina.

    Informações Relevantes:

    {history}

    Conversa:
    Humano: {input}
    IA:
"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)

##Para usar o ConversationKGMemory precisa do networkx instalado
memory = ConversationKGMemory(llm=OpenAI())

conversation_with_kg = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
    prompt=prompt
)

conversation_with_kg.predict(input="Oi! Eu sou a Maria.")
conversation_with_kg.predict(input="Eu gosto de ler livros e você?")
conversation_with_kg.predict(input="Eu adoro os livros do Harry Potter.")
conversation_with_kg.predict(input="Meu livro favorito é o quarto, o Cálice de Fogo.")

print(conversation_with_kg.memory.kg)
print(conversation_with_kg.memory.kg.get_triples())

