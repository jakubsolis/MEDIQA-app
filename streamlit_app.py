import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
import pandas as pd
import openai

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
examples_list = df['note'].tolist()

def construct_prompt(dialogue, examples_list):
    constructed_examples = ""
    for example in examples_list:
        temp_prompt = template.format(examples=constructed_examples + "\nNOTE:\n" + example, dialogue=dialogue)
        # Checking token count without invoking the API. This is a rough estimate, and can vary based on model specifics.
        if len(openai.Completion.create(model="text-davinci-003", prompt=temp_prompt, max_tokens=0).choices[0]['usage']['total_tokens']) <= 4097:
            constructed_examples += "\nNOTE:\n" + example
        else:
            break
    return template.format(examples=constructed_examples, dialogue=dialogue)

def generate_response(dialogue):
    llm = OpenAI(model_name='text-davinci-003', openai_api_key=openai_api_key)
    prompt_query = construct_prompt(dialogue, examples_list)
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
