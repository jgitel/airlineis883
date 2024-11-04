import streamlit as st
from openai import OpenAI
import os


st.title("Trip Feedback")

prompt = st.text_input("Share with us your experience of the latest trip.", "")

### Load your API Key
my_secret_key = st. secrets ["MyOpenAIKey"]
os.environ ["OPENAI_API_KEY"] = my_secret_key

from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


### Create the LLM API object
llm = OpenAI(openai_api_key=my_secret_key)
# llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")


### Create a template to handle the case where the price is not mentioned.
airline_template = """You are an expert at customer service for an airline.
From the following text, determine whether customer had a negative experience on their flight.

Do not respond with more than one word.

Text:
{text}

"""


### Create the decision-making chain

experience_type_chain = (
    PromptTemplate.from_template(airline_template)
    | llm
    | StrOutputParser()
)


fault_chain = PromptTemplate.from_template(
    """You are a customer service agent that is experienced with an airline's responsibilities. 
Determine if the cause of the traveler's dissatisfaction is the airline's fault (e.g., lost luggage) or beyond the airline's control (e.g., a weather-related delay) from the following text.
Do not respond with any reasoning. Just respond professionally as a customer service agent. Respond in first-person mode.

Your response should follow these guidelines:
    1. Do not provide any reasoning behind the determination of fault. Just respond professionally as a customer service chat agent.
    2. Address the customer directly



Text:
{text}

"""
) | llm


compensation_chain = PromptTemplate.from_template(
    """You are a customer service agent for an airline.
    Given the text below, determine if the traveler should receive compensation from the airline.

    Your response should follow these guidelines:
    1. You will offer the traveler sympathies 
    2. If the airline is at fault for a negative experience, you will inform the user that customer service will contact them soon to resolve the issue or provide compensation
    3. If the airline is not at fault for a negative experience, you will explain that the airline is not liable in such situations
    4. If the user's experience is positive, thank them for their feedback and for choosing to fly with the airline. 
    5. Do not respond with any reasoning. Just respond professionally as a travel chat agent.
    6. Address the customer directly

Text:
{text}

"""
) | llm


from langchain_core.runnables import RunnableBranch

### Routing/Branching chain
branch = RunnableBranch(
    (lambda x: "fault" in x["fault_type"].lower(), fault_chain),
    compensation_chain,
)

### Put all the chains together
full_chain = {"fault_type": fault_chain, "text": lambda x: x["text"]} | branch



import langchain
langchain.debug = False

full_chain.invoke({"text": prompt})


### Invoke it and print
answer = full_chain.run("text")
print(answer)
