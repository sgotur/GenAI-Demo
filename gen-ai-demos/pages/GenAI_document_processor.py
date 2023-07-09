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
import pdf2image
import os

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
txt_summary = ''

if 'txt_summary' not in st.session_state:
    st.session_state['txt_summary'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
rekognition = boto3.client('rekognition',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
textract = boto3.client('textract',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']

st.set_page_config(page_title="GenAI Document Processor", page_icon="ledger")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Derive insights from documents")
st.sidebar.header("GenAI Document Processor")

st.sidebar.markdown("### Make your pick")
industry = ''
industry = st.sidebar.selectbox(
    'Select an industry',
    ('Automotive', 'Financial Services', 'Healthcare', 'Legal', 'Energy'))

if industry == 'Automotive':
    st.sidebar.markdown('''
        ### Available doc types for analysis \n\n
        - BMW 3-Series owner manual \n
    ''')
elif industry == 'Financial Services':
    st.sidebar.markdown('''
        ### Available doc types for analysis \n\n
        - Selection of 10K docs \n
        - Annual reports \n
    ''')
elif industry == 'Healthcare':
    st.sidebar.markdown('''
        ### Available doc types for analysis \n\n
        - Georgia advanced directive for healthcare \n
        - Sample inpatient health record  \n
    ''')
elif industry == 'Legal':
    st.sidebar.markdown('''
        ### Available doc types for analysis \n\n
        - Sample business purchase contract \n
        - Amtrak railroad accident report \n
    ''')
elif industry == 'Energy':
    st.sidebar.markdown('''
        ### Available doc types for analysis \n\n
        - Demo Well Logs data \n
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

def GetAnswers(new_contents, query):
    pii_list = []
    sentiment = comprehend.detect_sentiment(Text=query, LanguageCode='en')['Sentiment']
    resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode='en')
    for pii in resp_pii['Entities']:
        if pii['Type'] not in ['NAME','AGE','ADDRESS','DATE_TIME']:
            pii_list.append(pii['Type'])
    if len(pii_list) > 0:
        answer = "I am sorry but I found PII entities " + str(pii_list) + " in your query. Please remove PII entities and try again."
        return answer
    query_type = ''
    if "you" in query:
        query_type = "BEING"

    if query == "cancel":
        answer = 'It was swell chatting with you. Goodbye for now'
    
    #elif sentiment == 'NEGATIVE':
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'

    elif query_type == "BEING":
        answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'
            
    else:
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic(new_contents+'. Answer from this text: '+ query.strip("query:"))
            if generated_text != '':
                answer = str(generated_text)+' '
                answer = answer.replace("$","\$")
            else:
                answer = 'Claude did not find an answer to your question, please try again'   
    return answer          


st.write("**Instructions:** \n - Select a document for processing \n - You will see a summary of the entire document \n - Type your query in the search bar to get document insights")

option = ''
if industry == 'Financial Services':
    option = st.selectbox(
        'Select a document to proceed:',
        ('Select...','BOFA-10K', 'Factset-Annual-Report', 'Moodys-Annual-Report', 'S-and-P-10K', 'Sunlife-Annual-Report-2022'))
elif industry == 'Automotive':
    option = st.selectbox(
        'Select a document to proceed:',
        ('Select...','BMW-2016-3series'))
elif industry == 'Healthcare':
    option = st.selectbox(
        'Select a document to proceed:',
        ('Select...','GA-Advanced-Directive-Healthcare', 'Sample-Inpatient-Health-Records', 'Healthcare-Claim-Form'))
elif industry == 'Legal':
    option = st.selectbox(
        'Select a document to proceed:',
        ('Select...','Sample-Business-Purchase-Contract', 'Amtrak-Accident-Report'))
elif industry == 'Energy':
    option = st.selectbox(
        'Select a document to proceed:',
        ('Select...','Demo-Well-Log-Files'))
if option != '':
    st.session_state.txt_summary = ''

#select docs for analysis
def GetSummary(option):
    summary = ''
    contents = ''
    # Get an image of the document
    textract_bucket = 'genai-tests'
    in_prefix = 'textract/'
    in_file = option+'.pdf'
    #s3.download_file(textract_bucket, in_prefix+industry.split(' ')[0]+'/'+in_file, in_file)

    # download the Textract output file for the selection
    textract_prefix = 'textract/output/'
    textract_file = option+'.txt'
    s3.download_file(textract_bucket, textract_prefix+industry.split(' ')[0]+'/'+textract_file, textract_file)
    with open(textract_file) as f:
        contents = f.read()
    new_contents = contents[:20000]

    if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic('Create a 200 words summary of this document: '+ new_contents)
            if generated_text != '':
                summary = str(generated_text)+' '
                summary = summary.replace("$","\$")
            else:
                summary = 'Claude did not find an answer to your question, please try again'    
    return new_contents, summary    

new_contents = ''
if option not in ('Select...',''): 
    if st.session_state.txt_summary == '':
        new_contents, txt_summary = GetSummary(option)
        st.session_state['txt_summary'] = txt_summary
        st.markdown('**Document summary:** \n') 
        st.write(str(st.session_state['txt_summary']))
        if model.lower() == 'anthropic claude':  
            p_text = call_anthropic('Generate three prompts to query the summary: '+ st.session_state.txt_summary)
            p_text1 = []
            p_text2 = ''
            if p_text != '':
                p_text.replace("$","USD")
                p_text1 = p_text.split('\n')
                for i,t in enumerate(p_text1):
                    if i > 1:
                        p_text2 += t.split('?')[0]+'\n\n'
                p_summary = p_text2
        st.sidebar.markdown('### Suggested prompts for further insights \n\n' + 
                p_summary)
    
if st.session_state.txt_summary != '':
    input_text = st.text_input('**What insights would you like?**', key='text')
    if input_text != '':
        result = GetAnswers(new_contents,input_text)
        result = result.replace("$","\$")
        st.write(result)





