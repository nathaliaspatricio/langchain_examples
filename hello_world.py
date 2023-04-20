##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=J_0qvRt4LNk

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
##A HUGGINGFACEHUB_API_TOKEN deve ser gerado na sua conta no HuggingFace: https://huggingface.co/settings/tokens
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub

#Primeiro exemplo com OpenAI GPT3
llm = OpenAI(model_name = "text-davinci-003", temperature = 0.9, max_tokens = 256)
text = "Quem foi Santos Dumont?"
print(text)
print("GPT 3 responde:" + llm(text))

#Primeiro exemplo com T5-Flan-XL
llm_hf = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0.9})
print("T5-Flan-XL responde: " + llm_hf(text))
