from urllib import response
import openai
import langchain
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain   
from langchain.chains import SimpleSequentialChain
from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.memory import ConversationBufferMemory


# print(os.getenv("API_KEY"))

from langchain.llms import OpenAI
llm=OpenAI(openai_api_key='API_KEY')


cities=["Paris", "London", "Berlin", "Madrid"]
def practice():
    
    promt=PromptTemplate.from_template("What is the capital of {place}?")

    llm=OpenAI(temperature=0.3)
    chain=LLMChain(llm=llm, prompt=promt)

    for city in cities:
        ouput=chain.run(place="France")
        print(ouput)

    ouput=chain.run(place="France")
    print(ouput)

promt=PromptTemplate.from_template("what is the name of e commerce that sells product {prduct}?")
llm=OpenAI(temperature=0.3)
chain1=LLMChain(llm=llm, prompt=promt)

#llm to names of products from an e commerce site
llm=OpenAI(temperature=0.3)
chain=LLMChain(llm=llm, prompt=promt)
chain2=LLMChain(llm=llm, prompt=promt)

overall_chain=SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

print(overall_chain({"era": "laptop"}))


# print(llm.predict("What is the capital of France?")) 

# import time
# time.sleep(5)


#create an overall chain from simple simple sequential chain

chain=SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
ouput=chain.run("laptop")
print(ouput)

llm=OpenAI(temperature=0.3)
tools=load_tools(["wikipedia", "llm=math"],llm=llm)
agent=initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
ouput=agent.run("how old is M.S.Dhoni in 2020?") 
print(ouput)

#memory in llms

llm=OpenAI(temperature=0.3)
promt=PromptTemplate.from_template("what is the name of e commerce that sells product {prduct}?")
chain=LLMChain(llm=llm, prompt=promt,memory=ConversationBufferMemory())
ouput=chain.run("laptop")

import time
time.sleep(5)

print(chain.memory)
print(chain.memory.buffer)
print(ouput)