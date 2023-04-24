##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=LNq_2s_H01Y

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap

llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter()

# carregar o doc
with open('ia.txt') as f:
    ia = f.read()
texts = text_splitter.split_text(ia)

print("Tamanho do texto: " + str(len(texts)))

docs = [Document(page_content=t) for t in texts[:4]]

print(docs)
print("\n\n")

prompt_template = """Escreva um resumo suscinto em formato de bullet points do seguinte:
{text}
RESUMO SUSCINTO EM BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, 
                        input_variables=["text"])

#Usando Map Reduce para resumir o texto e Template de prompt
chain = load_summarize_chain(llm, 
                        chain_type="map_reduce", 
                        return_intermediate_steps=True, 
                        map_prompt=BULLET_POINT_PROMPT, 
                        combine_prompt=BULLET_POINT_PROMPT)

#retorna não só o resumo final, mas os resumos intermediários
output_summary = chain({"input_documents": docs}, return_only_outputs=True)

#imprime os resumos intermediários 
i=0
while i <= 2: 
    wrapped_text = textwrap.fill(output_summary["intermediate_steps"][i], 
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)
    print(wrapped_text)
    i=i+1

#imprime o resumo final 
wrapped_text = textwrap.fill(output_summary["output_text"], 
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)
print(wrapped_text)





