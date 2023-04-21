##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=X550Zbz_ROE

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(
    model_name = "text-davinci-003", 
    temperature = 0, 
    max_tokens = 512
) 

##max_token_limit=40 - limites de tokens precisa de transformers e tiktoken instalado
memory = ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)

conversation_with_summary = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)

conversation_with_summary.predict(input="Oi! Eu sou a Maria.")
conversation_with_summary.predict(input="Eu gosto de ler livros e você?")
conversation_with_summary.predict(input="Eu adoro os livros do Harry Potter.")
conversation_with_summary.predict(input="Meu livro favorito é o quarto, o Cálice de Fogo.")

print(conversation_with_summary.memory.moving_summary_buffer)
