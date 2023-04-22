##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=X550Zbz_ROE

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain import OpenAI, ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any

## O prompt
ENTITY_MEMORY_CONVERSATION_TEMPLATE.template
"""
Você é o assistente de um humano, alimentado por um grande modelo de linguagem treinado pela OpenAI.

Você foi projetado para ajudar em uma ampla gama de tarefas, desde responder a perguntas simples até fornecer explicações e discussões detalhadas sobre uma ampla variedade de tópicos. Como um modelo de linguagem, você é capaz de gerar texto semelhante ao humano com base na entrada que recebe, permitindo que você se envolva em conversas que soam naturais e forneça respostas coerentes e relevantes para o tópico em questão.

Você está constantemente aprendendo e melhorando, e suas capacidades estão em constante evolução. Você é capaz de processar e entender grandes quantidades de texto e pode usar esse conhecimento para fornecer respostas precisas e informativas a uma ampla gama de perguntas. Você tem acesso a algumas informações personalizadas fornecidas pelo humano na seção Contexto abaixo. Além disso, você pode gerar seu próprio texto com base nas informações recebidas, permitindo que você participe de discussões e forneça explicações e descrições sobre uma ampla gama de tópicos.

No geral, você é uma ferramenta poderosa que pode ajudar em uma ampla gama de tarefas e fornecer percepções e informações valiosas sobre uma ampla variedade de tópicos. Se o ser humano precisa de ajuda com uma pergunta específica ou apenas deseja conversar sobre um tópico específico, você está aqui para ajudar.

Contexto:
{entities}

Conversa atual:
{history}
Última linha:
Humano: {input}
Você:
"""

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256
)

conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm)
)

conversation.predict(input="Oi! Eu sou a Maria. Eu gostaria de ler algum livro agora. Você teria alguma indicação para mim?")
conversation.predict(input="Eu prefiro não ficção")
conversation.predict(input="Gostaria de ler algum livro em inglês.")
answer = conversation.predict(input="Onde posso encontrá-los em Karlsruhe?")
print(answer)

print(conversation.memory.entity_cache)
print(conversation.memory.entity_store.store)
