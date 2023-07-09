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
s3_bucket = 'genai-tests'
s3_prefix = 'code-migrator/content/'+str(uuid.uuid4())


if 'code_summary' not in st.session_state:
    st.session_state['code_summary'] = None
if 'new_contents' not in st.session_state:
    st.session_state['new_contents'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)


# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['br_endpoint_name']
ant_name = os.environ['ant_name']
p_summary = ''
st.set_page_config(page_title="GenAI Code Migrator", page_icon="coffee")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Automate code mining and conversion")
st.sidebar.header("GenAI Code Migrator")
values = [1, 2, 3, 4, 5]
default_ix = values.index(3)
ftypes = ['cs','json','ipynb','py','java','yaml','cbl','rpgle','cpp','ts','sql','js','go','m','feature']
targets = ['C#','JSON','Python','Java','YAML','Cobol','RPGLE','C++','TypeScript','JavaScript','Golang','NodeJS','Stored Procedure','MATLAB','Gherkin']
target_extns = {'C#':'cs','Java':'java','JSON':'json','YAML':'yaml','Cobol':'cbl','RPGLE':'rpgle','C++':'cpp','TypeScript':'ts','Python':'py','JavaScript':'js','Golang':'go','NodeJS':'js','Stored Procedure':'sql','MATLAB':'m','Gherkin':'feature'}
p_count = st.sidebar.selectbox('Select the count of auto-prompts to consider', values, index=default_ix)
model = 'Anthropic Claude'
  
def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=2048,
    )
    return resp['completion']


def GetAnswers(original_text, query):
    generated_text = ''
    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic(original_text+'. Answer from this text with no hallucinations, false claims or illogical statements: '+ query.strip("query:"))
        if generated_text != '':
            answer = str(generated_text)
        else:
            answer = 'Claude did not find an answer to your question, please try again'   
    return answer          

 
def upload_code_get_summary(file_type, s3_file_name):
    summary = ''
    # download the csv file for the selection
    s3.download_file(s3_bucket, s3_prefix+'/'+s3_file_name, s3_file_name)
    
    with open(s3_file_name, 'rb') as f:
        contents = f.read()
    new_contents = contents.decode('utf-8')

    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic('Create a 100 words summary of this code: '+ new_contents)
        if generated_text != '':
            summary = str(generated_text)+' '
            summary = summary.replace("$","\$")
        else:
            summary = 'Claude did not find an answer to your question, please try again'    
    return new_contents, summary    


st.write("**Instructions:** \n 1. Drag & drop or select your source code in one of the supported languages \n 2. Select multiple target languages, default is Python \n 3. Click Upload button \n 4. You will see a short summary and converted code \n 5. Type your query in the search bar to get more insights or download new code")

uploaded_img = st.file_uploader("**Upload your source code**", type=ftypes)
default_lang_ix = targets.index('Python')
languages = st.multiselect(
    'Select one or more target languages',
    options=targets, default='Python')
code_summary = ''
new_contents = ''
if uploaded_img is not None:
    file_type = str(uploaded_img.name).split('.')[1]            
    if st.button('Upload'):
        with st.spinner('Uploading source code and migrating...'):
            #stringio = StringIO(uploaded_img.getvalue().decode("utf-8"))
            s3.upload_fileobj(uploaded_img, s3_bucket, s3_prefix+'/'+uploaded_img.name)
            new_contents, code_summary = upload_code_get_summary(file_type, uploaded_img.name)
            code_summary = code_summary.replace("$","\$")
            if len(code_summary) > 5:
                st.session_state['code_summary'] = code_summary
            new_contents = new_contents.replace("$","\$")
            st.session_state.new_contents = new_contents
            st.success('File uploaded and summary generated')


code = ''
if uploaded_img is not None:
    if st.session_state.code_summary:
        if len(st.session_state.code_summary) > 5:
            st.markdown('**Code Summary**: \n')
            st.write(str(st.session_state.code_summary).replace("$","\$"))
            if model.lower() == 'anthropic claude':
                p_text = call_anthropic('Generate'+str(p_count)+'prompts to query the text: '+ st.session_state.code_summary)
                p_text1 = []
                p_text2 = ''
                if p_text != '':
                    p_text.replace("$","\$")
                    p_text1 = p_text.split('\n')
                    for i,t in enumerate(p_text1):
                        if i > 1:
                            p_text2 += t.split('\n')[0]+'\n\n'
                    p_summary = p_text2
            st.sidebar.markdown('### Generated auto-prompts \n\n' + 
                        p_summary)
            tabs = st.tabs(languages)
            for i in range(0, len(languages)):
                with tabs[i]:
                    code = call_anthropic("Generate full and complete target code in "+str(languages[i])+" from this source code: "+str(st.session_state.new_contents))
                    st.code(code,language=str(languages[i]))
                    source_file = str(uploaded_img.name).strip('.')[0]
                    code_file = source_file+'.'+str(target_extns[languages[i]])
                    st.download_button('Download Code', code, file_name=code_file) 


input_text = st.text_input('**What insights would you like?**', key='text')
if input_text != '':
    if st.session_state.code_summary:
        result = GetAnswers(st.session_state.new_contents,input_text)
        result = result.replace("$","\$")
        st.write(result)
    else:
        st.write("I am sorry it appears you have not uploaded any files for analysis. Can you please upload a file and then try again?")                





