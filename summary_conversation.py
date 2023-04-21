##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=X550Zbz_ROE

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(
    model_name = "text-davinci-003", 
    temperature = 0, 
    max_tokens = 256
) 

summary_memory = ConversationSummaryMemory(llm=OpenAI())

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=summary_memory
)

conversation.predict(input="Oi! Eu sou a Maria.")
conversation.predict(input="Eu gosto de ler livros e você?")
conversation.predict(input="Eu adoro os livros do Harry Potter.")

print(conversation.memory.buffer)
