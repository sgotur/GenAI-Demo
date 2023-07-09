import json
import boto3
import streamlit as st
import datetime
from io import BytesIO
from io import StringIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
import uuid
import ai21
import string
import anthropic
import textract
from pypdf import PdfReader
import os


key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

if 'formula_summary' not in st.session_state:
    st.session_state['formula_summary'] = None
if 'formula' not in st.session_state:
    st.session_state['formula'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
ant_name = os.environ['ant_name']
p_summary = ''
st.set_page_config(page_title="GenAI Formulator", page_icon="test_tube")

st.markdown(
    """
    ### :red[Note] 
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Learn formulas at the speed of Formula 1")
st.sidebar.header("GenAI Formulator")
st.sidebar.write('''
        ### Example prompts you can try \n\n
        standard deviation \n
        air pressure for an airplane take off \n
        escape velocity \n
    ''')

model = 'Anthropic Claude'
  
def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=1024,
    )
    return resp['completion']

def GetAnswers(original_text, query):
    generated_text = ''
    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic(original_text+'. Answer from the formula summary: '+ query.strip("query:"))
        if generated_text != '':
            answer = str(generated_text)
        else:
            answer = 'Claude did not find an answer to your question, please try again'   
    return answer          


st.write("**Instructions:** \n 1. Type a prompt for which you need the formula \n 2. Demo generates formula and summary \n 3. Type additional questions for more insights \n ")
formula_idea = st.text_input('**What is the formula for?**', key='textf1')
generated_formula = ''
result = ''
if formula_idea != '':
    if model.lower() == 'anthropic claude':        
        generated_formula = call_anthropic("What is the formula for "+formula_idea+"? Provide only the formula with no prefix or explanations")
        st.latex(generated_formula)
        formula_summary = call_anthropic("Explain the formula in detail with an example: " + generated_formula)
        st.session_state.formula_summary = formula_summary
        st.write(st.session_state.formula_summary)
        p_text = call_anthropic('Generate 3 prompts to get additional insights from the formula summary: '+ st.session_state.formula_summary)
        p_text1 = []
        p_text2 = ''
        if p_text != '':
            p_text.replace("$","USD")
            p_text1 = p_text.split('\n')
            for i,t in enumerate(p_text1):
                if i > 1:
                    p_text2 += t.split('\n')[0]+'\n\n'
            p_summary = p_text2
        st.sidebar.write('### Suggested prompts for further insights \n\n' + 
                p_summary)
        

    input_text = st.text_input('**What insights would you like?**', key='textf1insights')
    if input_text != '':
        result = GetAnswers(st.session_state.formula_summary,input_text)
        result = result.replace("$","\$")
        st.write(result)




