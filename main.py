from dotenv import load_dotenv
import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain


load_dotenv()


llm = GoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt_template = PromptTemplate(
    template="Give me an  example of a meal could be made using the following ingredients: {ingredients}",
    input_variables={"ingredients"},
)
gangster_template = """Re-write the meals given below in the style of a New York mafia gangster:

Meals:
{meals}
"""

gangster_template_prompt = PromptTemplate(
    template=gangster_template,
    input_variables={"meals"},
)

meal_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="meals",
    verbose=True,  # This will print the intermediate steps in the chain. Useful for debugging.
)

gangster_chain = LLMChain(
    llm=llm,
    prompt=gangster_template_prompt,
    output_key="gangster_meals",
    verbose=True,  # This will print the intermediate steps in the chain. Useful for debugging.
)

overall_chain = SequentialChain(
    chains=[
        meal_chain,
        gangster_chain,
    ],
    input_variables=["ingredients"],
    output_variables=["meals", "gangster_meals"],
    verbose=True,  # This will print the intermediate steps in the chain. Useful for debugging.  # noqa: E501
)

st.title("Meal Planner")
user_prompt = st.text_input("Enter a comma-seperated list  of ingredients")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = overall_chain.run({"ingredients": user_prompt})

        col1, col2 = st.columns(2)
        col1.write(output["meals"])
        col2.write(output["gangster_meals"])
