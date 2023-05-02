##Código Teste sobre o LangChain

##A OPENAI_API_KEY deve ser gerado na sua conta no OpenAI: https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

import yaml, json

from langchain.agents import (
    create_json_agent,
    AgentExecutor
)
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.llms import HuggingFaceHub
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec

#with open("dualstudium_informatik.json") as f:
    #data = yaml.load(f, Loader=yaml.FullLoader)
    #print(data)

with open("dualstudium_informatik.json") as data_file:
    data = json.load(data_file)
    print(data)

json_spec = JsonSpec(dict_=data)
json_toolkit = JsonToolkit(spec=json_spec)
print(json_toolkit)


#llm = OpenAI(model_name='text-davinci-003', temperature=0)

#repo_id = "google/flan-t5-xl"
repo_id = "stabilityai/stablelm-tuned-alpha-3b"
llm_hf = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0, "max_length":500})

json_agent_executor = create_json_agent(
    llm=llm_hf,
    toolkit=json_toolkit,
    verbose=True
)

answer = json_agent_executor.run("How many names are in the dataset?")
print(answer)
