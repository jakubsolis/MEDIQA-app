import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
import pandas as pd

# Set the page title
st.set_page_config(page_title="🦜🔗 Clinical Note Generator App")
st.title('🦜🔗 Clinical Note Generator App')

# Get the OpenAI API Key from the sidebar
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Define a simplified template
template = """
Generate a clinical note using these examples:
{examples}

For dialogue: {dialogue}
CLINICAL NOTE:
"""
prompt = PromptTemplate(input_variables=['examples', 'dialogue'], template=template)

# Load the in-context examples from the fixed Excel file
df = pd.read_csv('examples.csv')
# Assuming the examples are in a column named 'note', take only top 4 to reduce token count
examples_list = df['note'].tolist()[:4]  
examples = "\n".join([f"NOTE:\n{example.strip()}" for example in examples_list])

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
