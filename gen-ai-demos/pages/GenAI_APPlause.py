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

if 'code_summary' not in st.session_state:
    st.session_state['code_summary'] = None
if 'new_code' not in st.session_state:
    st.session_state['new_code'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
rekognition = boto3.client('rekognition',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
ant_name = os.environ['ant_name']
p_summary = ''
st.set_page_config(page_title="GenAI APPlause", page_icon="sparkles")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Build Apps at the speed of thought")
st.sidebar.header("GenAI APPlause")
st.sidebar.write('''
        ### Example build ideas you can try \n\n
        Python Streamlit code for a banking app using DynamoDB \n
        HTML and Javascript code for a reward redemption app with DynamoDB for storing reward points \n
        C++ class to dynamically calculate hotel room rate using rate types, seasonality, location and length of stay \n
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
        generated_text = call_anthropic(original_text+'. Modify this code and return markdowns for each module using the suggestions in: '+ query.strip("query:"))
        if generated_text != '':
            answer = str(generated_text)
        else:
            answer = 'Claude did not find an answer to your question, please try again'   
    return answer          


st.write("**Instructions:** \n 1. Type a simple app idea \n 2. Demo generates code for your idea \n 3. Try to fine tune your idea using the search bar \n 4. Upcoming feature - auto-containerize and deploy your app using AWS Fargate \n")
generated_code = ''
prog_language = ''
new_language = ''
code_idea = st.text_input('**What would you like to build?**', key='text1')
if code_idea != '':
    if model.lower() == 'anthropic claude':
        tab1, tab2, tab3 = st.tabs(["Code", "Summary", "AWS Well-Architected Recommendations"])
        with tab1:        
            generated_code = call_anthropic("Generate and return the code for each module using the programming language and programming framework requested in "+code_idea)
            prog_language = call_anthropic("Answer only in a single word. What is the programming language for this code: " + generated_code)
            st.code(generated_code, language=prog_language)
            #st.write(generated_code)
            st.session_state.new_code = generated_code
            p_text = call_anthropic('Generate 3 architecture ideas to improve quality and performance without sample code: '+ generated_code)
            p_text1 = []
            p_text2 = ''
            if p_text != '':
                p_text.replace("$","USD")
                p_text1 = p_text.split('\n')
                for i,t in enumerate(p_text1):
                    if i > 1:
                        p_text2 += t.split('\n')[0]+'\n\n'
                p_summary = p_text2
            st.sidebar.write('### Code tuning suggestions \n\n' + 
                    p_summary)
        with tab2:
            code_summary = call_anthropic("Summarize the code in 150 words or less: " + generated_code)
            st.session_state.code_summary = code_summary
            st.write(code_summary)
        with tab3:
            aws_war = call_anthropic("Perform AWS Well Architected review of this code: " + generated_code)
            st.write(aws_war)
        

result = ''
input_text = st.text_input('**How would you like to modify this code?**', key='ctext')
if input_text != '':
    mtab1, mtab2, mtab3 = st.tabs(["Modified Code", "Modified Summary", "Updated AWS WAR"])
    with mtab1:
        result = GetAnswers(generated_code,input_text)
        result = result.replace("$","\$")
        new_language = call_anthropic("Answer only in a single word. What is the programming language for this code: " + result)
        st.code(result, language=new_language)
    with mtab2:
        code_summary = call_anthropic("Summarize the code in 150 words or less: " + result)
        st.session_state.code_summary = code_summary
        st.write(code_summary)
    with mtab3:
        aws_mwar = call_anthropic("Perform AWS Well Architected review of this code: " + result)
        st.write(aws_mwar)            
    





