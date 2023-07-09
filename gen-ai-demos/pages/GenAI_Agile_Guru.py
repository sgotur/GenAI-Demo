import json
import boto3
import streamlit as st
import datetime
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
import uuid
import ai21
import string
import anthropic
import os

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

if 'user_stories' not in st.session_state:
    st.session_state['user_stories'] = None
if 'data_model' not in st.session_state:
    st.session_state['data_model'] = None
if 'api_specs' not in st.session_state:
    st.session_state['api_specs'] = None


# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']

st.set_page_config(page_title="GenAI Agile Guru", page_icon="high_brightness")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# From epic to Epic in seconds")
st.sidebar.header("GenAI Agile Guru")
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

def GetAnswers(query):
    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic('Create 5 agile scrum user stories and acceptance criteria for each user story in '+language+' for '+ query.strip("query:"))
        if generated_text != '':
            generated_text = generated_text.replace("$","\$")
            us_answer = str(generated_text)
        else:
            us_answer = 'Claude did not find an answer to your question, please try again'   
    return us_answer                       

st.write("**Instructions:** \n - Type an epic story \n - You will see user stories, data model, api specs, and BDD scenarios automatically generated for your epic \n")

p_summary = '''
- funds transfer for banking \n
- login to member portal and check balance \n
- track and redeem rewards points \n
- create customized landing page for website \n
'''
  
st.sidebar.write('### Suggested epics to get started \n\n' + 
            p_summary)

input_text = st.text_input('**Type an epic**', key='text_ag')
default_lang_ix = languages.index('English')
language = st.selectbox(
    '**Select an output language.**',
    options=languages, index=default_lang_ix)
generated_text = ''
dm_generated_text = ''
as_generated_text = ''
bd_generated_text = ''
us_answer = ''
as_answer = ''
dm_answer = ''
bd_answer = ''
if input_text != '':
    us_answer = GetAnswers(input_text)
    tab1, tab2, tab3, tab4 = st.tabs(["User Stories", "Data Model", "API Specs", "BDD Secenarios"])
    #c1, c2 = st.columns(2)
    with tab1:
        if us_answer:
            st.write("**User stories for your epic**")
            st.write(us_answer)
    with tab2:
        dm_generated_text = call_anthropic('Create a data model in '+language+' for each of the user stories in '+str(us_answer))
        if dm_generated_text != '':
            dm_generated_text = dm_generated_text.replace("$","\$")
            dm_answer = str(dm_generated_text)
            st.session_state.data_model = dm_answer
        else:
            dm_answer = 'Claude did not find an answer to your question, please try again' 
        if dm_answer:
            st.write("**Data model for your user stories**")    
            st.write(dm_answer)
    with tab3:
        as_generated_text = call_anthropic('Create microservices API specifications in '+language+' for each of the data models in '+ str(dm_answer))
        if as_generated_text != '':
            as_generated_text = as_generated_text.replace("$","\$")
            as_answer = str(as_generated_text)
            st.session_state.api_specs = as_answer
        else:
            as_answer = 'Claude did not find an answer to your question, please try again'
        if as_answer:        
            st.write("**API Specs for your user stories**")  
            st.write(as_answer)
    with tab4:
        bd_generated_text = call_anthropic('Create behavior driven development scenarios using cucumber in '+language+' for each of the user stories in '+ str(us_answer))
        if bd_generated_text != '':
            bd_generated_text = bd_generated_text.replace("$","\$")
            bd_answer = str(bd_generated_text)
        else:
            bd_answer = 'Claude did not find an answer to your question, please try again'
        if bd_answer:
            st.write("**BDD Scenarios for your user stories**")
            st.write(bd_answer)
    