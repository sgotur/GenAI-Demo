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
import subprocess
import sys
import os



st.set_page_config(page_title="GenAI Demo Solutions Architecture", page_icon="telescope")

st.markdown("## The architecture behind the demos you love!!")

st.sidebar.header("GenAI Demo Solutions Architecture")

st.sidebar.markdown('''
    ### AWS AI Services used: \n\n
    - [Amazon Comprehend](https://aws.amazon.com/comprehend/) \n
    - [Amazon Kendra](https://aws.amazon.com/kendra/) \n
    - [Amazon Lex](https://aws.amazon.com/lex/) \n
    - [Amazon Textract](https://aws.amazon.com/textract/) \n
    - [Amazon Transcribe](https://aws.amazon.com/transcribe/) \n
    - [Amazon Rekognition](https://aws.amazon.com/rekognition/) \n
''')

st.sidebar.markdown('''
    ### Foundation Models used: \n\n
    - [Anthropic Claude - coming soon on Amazon Bedrock](https://www.anthropic.com/index/introducing-claude) \n
    - [Stability - SageMaker Jumpstart](https://aws.amazon.com/blogs/machine-learning/generate-images-from-text-with-the-stable-diffusion-model-on-amazon-sagemaker-jumpstart/) \n
''')

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
s3_bucket = 'genai-tests'
s3_arch_prefix = 'architecture-diagrams'

if 'arch_summary' not in st.session_state:
    st.session_state['arch_summary'] = None
if 'abstract' not in st.session_state:
    st.session_state['abstract'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
rekognition = boto3.client('rekognition',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
ant_name = os.environ['ant_name']
#languages = ['English', 'Spanish', 'German', 'Italian', 'Irish', 'French', 'Portugese', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
demos = ['GenAI_call_analyzer','GenAI_ChatAway','GenAI_content_analyzer','GenAI_digital_persona','GenAI_document_processor','GenAI_energy_upstream_analyzer','GenAI_enterprise_search_interpreter','GenAI_geospatial_analyzer','GenAI_image_analyzer','GenAI_knowledge_valet','GenAI_product_ideator','GenAI_product_recommender','GenAI_video_analyzer']
p_summary = ''

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

def GetAnswers(summary, query):
    pii_list = []
    sentiment = comprehend.detect_sentiment(Text=query, LanguageCode='en')['Sentiment']
    resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode='en')
    for pii in resp_pii['Entities']:
        if pii['Type'] not in ['ADDRESS','DATE_TIME']:
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
            generated_text = call_anthropic(summary+'. Answer from this summary in '+language+ ': '+ query.strip("query:"))
            if generated_text != '':
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again'   
    return answer          


#upload audio file to S3 bucket
def detect_image_labels(arch_image_file):
    summary = ''
    label_text = ''
    response = rekognition.detect_labels(
        Image={'S3Object':{'Bucket':s3_bucket,'Name':s3_arch_prefix+'/'+arch_image_file}},
        Features=['GENERAL_LABELS']
    )
    text_res = rekognition.detect_text(
        Image={'S3Object':{'Bucket':s3_bucket,'Name':s3_arch_prefix+'/'+arch_image_file}}
    )

    for text in text_res['TextDetections']:
        label_text += text['DetectedText'] + ' '

    for label in response['Labels']:
        label_text += label['Name'] + ' '

    if model.lower() == 'anthropic claude':  
        abstract = call_anthropic('Create a 100 words business abstract highlighting the enterprise use case and industries applicable in '+language+' using only the terms mentioned in these labels: '+ label_text)
        generated_text = call_anthropic('Explain the architecture step by step in '+language+' in 300 words using only the terms mentioned in these labels: '+ label_text)
        if generated_text != '':
            generated_text.replace("$","USD")
            summary = str(generated_text)+' '
        else:
            summary = 'Claude did not find an answer to your question, please try again'    
        return abstract, summary    

st.write("**Instructions:** \n - Select a GenAI demo and output language \n - You will see its solution architecture, an abstract and an explanation \n - You can type questions to get more insights on the architecture \n")
default_demo_ix = demos.index('GenAI_call_analyzer')
arch_img = ''
demo = st.selectbox(
    'Select a GenAI demo',
    options=demos, index=default_demo_ix)
if demo is not None:
    arch_img = demo.lower()+'_architecture.png'
    subprocess.run([f"{sys.executable}", demo+'_architecture.py'])
    s3.upload_file(arch_img, s3_bucket, s3_arch_prefix+'/'+arch_img)

default_lang_ix = languages.index('English')
language = st.selectbox(
    'Select an output language',
    options=languages, index=default_lang_ix)
arch_summary = ''
p_summary = ''
abstract = ''
if language is not None:
    if st.button('Draw and explain'):
        with st.spinner('Drawing architecture, detecting Amazon Rekognition labels and generating explanation...'):
            st.markdown("**Solution Architecture**: \n")
            st.image(arch_img)
            abstract, arch_summary = detect_image_labels(demo.lower()+'_architecture.png')
            arch_summary = arch_summary.replace("$","\$")
    
if len(arch_summary) >= 5:
    st.session_state['arch_summary'] = arch_summary
if len(abstract) >= 5:
    st.session_state['abstract'] = abstract

if st.session_state.arch_summary:
    st.markdown('**Abstract**: \n')
    st.write(str(st.session_state['abstract']))
    st.markdown('**Explanation**: \n')
    st.write(str(st.session_state['arch_summary']))
    if model.lower() == 'anthropic claude':  
        p_text = call_anthropic('Generate three prompts in '+language+' to query the summary: '+ st.session_state.arch_summary)
        p_text1 = []
        p_text2 = ''
        if p_text != '':
            p_text.replace("$","USD")
            p_text1 = p_text.split('\n')
            for i,t in enumerate(p_text1):
                if i > 1:
                    p_text2 += t.split('\n')[0]+'\n\n'
            p_summary = p_text2
    st.sidebar.markdown('### Suggested prompts for further insights \n\n' + 
                p_summary)
    
    input_text = st.text_input('**What insights would you like?**', key='text')
    if input_text != '':
        result = GetAnswers(st.session_state.arch_summary,input_text)
        result = result.replace("$","\$")
        st.image(arch_img)
        st.write(result)
