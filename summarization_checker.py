##Código Teste sobre o LangChain
##Tutorial completo em: https://www.youtube.com/watch?v=d-yeHDLgKHw

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import LLMSummarizationCheckerChain
import textwrap

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)

article = """
Brasil, oficialmente República Federativa do Brasil, é o maior país da América do Sul e da região da América Latina, sendo o quinto maior do mundo em área territorial (equivalente a 47,3% do território sul-americano), com 8 510 417,771 km², e o sexto em população (com mais de 207,8 milhões de habitantes). É o único país na América onde se fala majoritariamente a língua portuguesa e o maior país lusófono do planeta, além de ser uma das nações mais multiculturais e etnicamente diversas, em decorrência da forte imigração oriunda de variados locais do mundo. Sua atual Constituição, promulgada em 1988, concebe o Brasil como uma república federativa presidencialista, formada pela união dos 26 estados, do Distrito Federal e dos 5 570 municípios. 
Banhado pelo Oceano Atlântico, o Brasil tem um litoral de 7 491 km e faz fronteira com todos os outros países sul-americanos, exceto Chile e Equador, sendo limitado a norte pela Venezuela, Guiana, Suriname e pelo departamento ultramarino francês da Guiana Francesa; a noroeste pela Colômbia; a oeste pela Bolívia e Peru; a sudoeste pela Argentina e Paraguai e ao sul pelo Uruguai. Vários arquipélagos formam parte do território brasileiro, como o Atol das Rocas, o Arquipélago de São Pedro e São Paulo, Fernando de Noronha (o único destes habitado por civis) e Trindade e Martim Vaz. O Brasil também é o lar de uma diversidade de animais selvagens, ecossistemas e de vastos recursos naturais em uma grande variedade de habitats protegidos. Particularmente, acho o Brasil o melhor país do mundo. As comidas do Brasil também são as mais deliciosas.
"""

print(len(article))

fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extraia os fatores chave desse texto. Não inclua opiniões. Dê a cada fato um número e mantenha frases curtas. :\n\n {text_input}"
)

fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)
facts = fact_extraction_chain.run(article)

wrapped_text = textwrap.fill(facts, 
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)
print(wrapped_text)
print(len(wrapped_text))

checker_chain = LLMSummarizationCheckerChain(llm=OpenAI(temperature=0), 
                                             verbose=True, 
                                             max_checks=2
                                             )

final_summary = checker_chain.run(article)
print(final_summary)
print(len(final_summary))

