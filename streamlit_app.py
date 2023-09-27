import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
import pandas as pd

st.set_page_config(page_title="🦜🔗 Clinical Note Generator App")
st.title('🦜🔗 Clinical Note Generator App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Define the PromptTemplate from the provided script
template = """Write a clinical note reflecting this doctor-patient dialogue. 
Use the example notes below to decide the structure of the clinical note. 
Do not make up information.
{examples}

DIALOGUE: {dialogue}
CLINICAL NOTE:
"""
prompt = PromptTemplate(input_variables=['examples', 'dialogue'], template=template)

# Load the in-context examples from the fixed Excel file
df = pd.read_excel('examples.xlsx')
# Assuming the examples are in a column named 'note'
examples_list = df['note'].tolist()
examples = "\n".join([f"EXAMPLE NOTE:\n{example.strip()}" for example in examples_list])

def generate_response(dialogue, examples=examples):
    llm = OpenAI(model_name='text-davinci-003', openai_api_key=openai_api_key)
    prompt_query = prompt.format(dialogue=dialogue, examples=examples)
    response = llm(prompt_query)
    return response

with st.form('myform'):
    dialogue_text = st.text_area('Enter the doctor-patient dialogue:', '')
    submitted = st.form_submit_button('Submit')
    
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        response = generate_response(dialogue_text)
        st.info(response)

