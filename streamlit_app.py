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


### Create a template to handle the case where the customer gives an unstructured review.
airline_template = """You are an expert at customer service for an airline.
From the following text, determine whether customer had a negative experience on their flight.
If they did, determine if the cause was the airline's fault (e.g., lost luggage) or beyond the airline's control (e.g., a weather-related delay).
Respond with two words. 
1. Positive or negative. Pick one.
2. Fault, not, or leave blank if 1. is positive. Pick one. 

Do not respond with more than two words.

Text:
{text}

"""


### Create the decision-making chain

flight_experience_chain = (
    PromptTemplate.from_template(airline_template)
    | llm
    | StrOutputParser()
)


negative_chain = PromptTemplate.from_template(
    """You are an expert at customer service for an airline. 
Given the text below, determine if the traveler who had a negative experience should receive compensation from the airline.
Do not respond with any reasoning. Just respond professionally as a customer service agent. Respond in first-person mode. Leave text blank if traveler had positive experience.

Your response should follow these guidelines:
    1. You will offer the traveler sympathies
    2. If the airline is at fault for a negative experience, you will inform the user that customer service will contact them soon to resolve the issue or provide compensation
    3. If the airline is not at fault for a negative experience, you will explain that the airline is not liable in such situations
    4. Do not respond with any reasoning. Just respond professionally as a travel chat agent.
    5. Address the customer directly  


Text:
{text}

"""
) | llm


positive_chain = PromptTemplate.from_template(
    """You are an expert at customer service for an airline.
    Given the text below, if the traveler had a positive experience, respond with the following.
    Do not respond with any reasoning. Just respond professionally as a customer service agent. Respond in first-person mode. Leave text blank if traveler had negative experience.

    Your response should follow these guidelines:
    1. You will thank the traveler for their feedback
    2. You will thank the traveler for flying with our airline
    3. You will wish the traveler well and offer them to reach out if they need anything
    4. Do not respond with any reasoning. Just respond professionally as a travel chat agent.
    5. Address the customer directly

Text:
{text}

"""
) | llm


from langchain_core.runnables import RunnableBranch

### Routing/Branching chain
branch = RunnableBranch(
    (lambda x: "negative" in x["airline_template"].lower(), flight_experience_chain),
    (lambda x: "positive" in x["airline_template"].lower(), flight_experience_chain),
    negative_chain, positive_chain
)

### Put all the chains together
full_chain = {"airline_template": flight_experience_chain, "text": lambda x: x["negative"], "text": lambda x: x["positive"], "text": negative_chain, "text": positive_chain} | branch



import langchain
langchain.debug = False

response = full_chain.invoke({"text": prompt})

### Display
st.write(response)
