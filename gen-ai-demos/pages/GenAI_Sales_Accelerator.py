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
import re


key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
s3_bucket = 'genai-tests'
s3_prefix = 'general/content'
falcon_endpoint = os.environ['falcon_endpoint_name']

if 'doc_summary' not in st.session_state:
    st.session_state['doc_summary'] = None
if 'out_template' not in st.session_state:
    st.session_state['out_template'] = None
if 'dashboard_contents' not in st.session_state:
    st.session_state['dashboard_contents'] = None
if 'generated_report' not in st.session_state:
    st.session_state['generated_report'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
bedrock = boto3.client('bedrock',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
# Get environment variables
bucket = os.environ['bucket']
p_summary = ''
st.set_page_config(page_title="GenAI Sales Accelerator", page_icon="roller_skate")

st.markdown("# Turbo charge your sales reporting!!")
st.sidebar.header("GenAI Sales Accelerator")
values = [1, 2, 3, 4, 5]
default_ix = values.index(3)
ftypes = ['csv', 'pptx', 'rtf','xls','xlsx','txt', 'pdf', 'doc', 'docx']
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Italian', 'Korean', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']

models = ['Bedrock Titan', 'Bedrock Jurassic-2', 'Bedrock Claude', 'SageMaker Falcon']
default_model = models.index('Bedrock Claude')
default_ix = values.index(3)
model = st.sidebar.selectbox('Select a FM', models, index=default_model)

# Prompts
prompt_2x2 = 'Generate an output document that describes the Business Update, Action Items, Major Risks, Challenges, Blockers, Cross-segment trends, Customer Observations, Sales Signals, Successful Initiatives, Wins, Top opportunities, Losses from all the pages of this document in '
prompt_summary = 'Generate a summary in 300 words of this document in '


def call_bedrock_titan(prompt_text, max_token_count=1024, temperature=1, top_p=1, stop_sequences=[]):
    model_id = "amazon.titan-tg1-large"
    body_string = "{\"inputText\":\"" + f"{prompt_text}" +\
                    "\",\"textGenerationConfig\":{" +\
                    "\"maxTokenCount\":" + f"{max_token_count}" +\
                    ",\"temperature\":" + f"{temperature}" +\
                    ",\"topP\":" + f"{top_p}" +\
                    ",\"stopSequences\":" + f"{stop_sequences}" +\
                    "}}"
    body = bytes(body_string, 'utf-8')
    response = bedrock.invoke_model(
        modelId = model_id,
        contentType = "application/json",
        accept = "application/json",
        body = body)
    response_lines = response['body'].readlines()
    json_str = response_lines[0].decode('utf-8')

    json_obj = json.loads(json_str)
    result_text = json_obj['results'][0]['outputText']
    return result_text

def call_bedrock_claude(prompt_text, max_tokens_to_sample=1024, temperature=1, top_k=250, top_p=1):
    model_id = "anthropic.claude-v1"
    body = {
        "prompt": anthropic.HUMAN_PROMPT+prompt_text+anthropic.AI_PROMPT,
        "max_tokens_to_sample": max_tokens_to_sample
    }
    body_string = json.dumps(body)
    body = bytes(body_string, 'utf-8')
    response = bedrock.invoke_model(
        modelId = model_id,
        contentType = "application/json",
        accept = "application/json",
        body = body)
    response_lines = response['body'].readlines()
    json_str = response_lines[0].decode('utf-8')
    json_obj = json.loads(json_str)
    result_text = json_obj['completion']
    return result_text

def call_bedrock_jurassic(prompt_text, max_token_count=1024, temperature=1, top_p=1, stop_sequences=[]):
    model_id = "ai21.j2-jumbo-instruct"
    body_string = "{\"prompt\":\"" + f"{prompt_text}" + "\"" +\
                    ",\"maxTokens\":" + f"{max_token_count}" +\
                    ",\"temperature\":"  + f"{temperature}" +\
                    ",\"topP\":" + f"{top_p}" +\
                    ",\"stopSequences\":" + f"{stop_sequences}" +\
                    ",\"countPenalty\":{\"scale\":0}" +\
                    ",\"presencePenalty\":{\"scale\":0}" +\
                    ",\"frequencyPenalty\":{\"scale\":0}" +\
                    "}"      
    body = bytes(body_string, 'utf-8')
    response = bedrock.invoke_model(
        modelId = model_id,
        contentType = "application/json",
        accept = "application/json",
        body = body)
    response_lines = response['body'].readlines()
    json_str = response_lines[0].decode('utf-8')
    json_obj = json.loads(json_str)
    result_text = json_obj['completions'][0]['data']['text']
    return result_text

def call_falcon(query):
    payload = {
    "inputs": query,
        "parameters": {
            "max_new_tokens": 1024,
            "return_full_text": True,
            "do_sample": True,
            "temperature": 0.5,
            "repetition_penalty": 1.03,
            "top_p": 0.9,
            "top_k":1,
            "stop": ["<|endoftext|>", "</s>"]
        }
    }
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=falcon_endpoint, ContentType='application/json', Body=json.dumps(payload).encode('utf-8'))
    model_predictions = json.loads(response['Body'].read())
    falcon_resp = str(model_predictions[0]['generated_text'][len(query):])
    return falcon_resp

models = {
    "bedrock titan" : call_bedrock_titan,
    "bedrock jurassic-2" : call_bedrock_jurassic,
    "bedrock claude" : call_bedrock_claude,
    "sagemaker falcon" : call_falcon
}



def readpdf(filename):
    # creating a pdf reader object
    reader = PdfReader(filename)
    # getting a specific page from the pdf file
    raw_text = []
    for page in reader.pages:
        raw_text.append(page.extract_text())
    return '\n'.join(raw_text)

def GetAnswers(original_text, query):
    
    if query == "cancel":
        resp = 'It was swell chatting with you. Goodbye for now'
    
    #elif sentiment == 'NEGATIVE':
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'        
    else:
        query = 'Tune the generated output narrative using this input text ' + original_text + ' as a guidance to '+ prompt_2x2 +language+ ' from: '+ st.session_state.dashboard_contents
        func = models[model.lower()]
        resp = func(query) 
   
    return resp          


def upload_doc_get_summary(file_type, s3_file_name):
    summary = ''
    # download the csv file for the selection
    s3.download_file(s3_bucket, s3_prefix+'/'+s3_file_name, s3_file_name)
    
    if file_type not in ['pdf','txt']:
        contents = textract.process(s3_file_name).decode('utf-8')
        new_contents = contents[:50000].replace('$','\$')
    elif file_type == 'pdf':
        contents = readpdf(s3_file_name)
        new_contents = contents[:50000].replace("$","\$")
    else:
        with open(s3_file_name, 'rb') as f:
            contents = f.read()
        new_contents = contents[:10000].decode('utf-8')

    new_contents = re.sub('[^A-Za-z0-9]+', ' ', new_contents)    
    query = prompt_summary +language+ ': '+ new_contents
    func = models[model.lower()]
    summary = func(query) 
        
    return new_contents, summary    


doc_summary = ''
label_text = ''
file_type = ''
new_contents = ''
template_contents = ''
answer = ''


st.write("**Instructions:** \n 1. Upload your sales dashboard data in a supported format \n 2. Select an output language, default is English \n 3. You will get a generated summary and 2X2 \n 4. Type your query in the search bar to tune the narrative")

st.subheader("Upload your sales dashboard data")
uploaded_sales_data = st.file_uploader("**Select a file**", type=ftypes, key='c1SA')

st.subheader("Select an output language")
default_lang_ix = languages.index('English')
language = st.selectbox(
    'Only Alpha and Beta quadrant languages supported. For new requests, please contact C-3PO',
    options=languages, index=default_lang_ix)

if uploaded_sales_data is not None:
    file_type = str(uploaded_sales_data.name).split('.')[1]
    if str(uploaded_sales_data.name).split('.')[1] in ftypes:            
        st.success(uploaded_sales_data.name + ' is ready for upload')
        if st.button('Upload', key='c1btSA'):
            with st.spinner('Uploading sales data and generating your document...'):
                #stringio = StringIO(uploaded_img.getvalue().decode("utf-8"))
                s3.upload_fileobj(uploaded_sales_data, s3_bucket, s3_prefix+'/'+uploaded_sales_data.name)
                new_contents, doc_summary = upload_doc_get_summary(file_type, uploaded_sales_data.name)
                doc_summary = doc_summary.replace("$","\$")
                if len(doc_summary) > 5:
                    st.session_state['doc_summary'] = doc_summary
                new_contents = new_contents.replace("$","\$")
                st.session_state.dashboard_contents = new_contents
                
                tab1, tab2 = st.tabs(['Summary','2X2'])
                with tab1:
                    st.write(st.session_state.doc_summary)
                with tab2:
                    query = prompt_2x2 + language + ': '+ st.session_state.dashboard_contents
                    func = models[model.lower()]
                    answer = func(query)
                    answer = answer.replace('$','\$')
                    st.session_state.generated_report = answer
                    st.write("**Generated document:**")
                    st.write(st.session_state.generated_report)

input_text = st.text_input('**Tune the narrative**', key='textSA')
if input_text != '':
    if st.session_state.dashboard_contents:
        result = GetAnswers(st.session_state.dashboard_contents,input_text)
        result = result.replace("$","\$")
        st.write("**Tuned output based on your inputs:**")
        st.write(result)
    else:
        st.write("I am sorry it appears you have not uploaded any files for analysis. Can you please upload a file and then try again?")                





