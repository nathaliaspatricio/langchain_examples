##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=phHqvLHCwH4&t=58s

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

import openai

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Você é um assistente útil."},
        {"role": "user", "content": "Oi, que tipo de assistente você é?"}
    ]
)

print(response)

messages=[
    {"role": "system", "content": "Você é um assistente útil."},
    {"role": "user", "content": "Oi, que tipo de assistente você é?"}
]

conversation_total_tokens = 0
while True:
    message = input("Usuário: ")
    if message == "exit":
        print(f"{conversation_total_tokens} tokens foram usados no total nessa conversa")
        break
    if message:
        messages.append(
            {"role": "user", "content": message},        
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
    reply = response.choices[0].message.content
    total_tokens = response.usage["total_tokens"]
    conversation_total_tokens += total_tokens
    print(f"ChatGPT: {reply} \n {total_tokens} tokens usados")
    messages.append({"role": "assistant", "content": reply}) 
