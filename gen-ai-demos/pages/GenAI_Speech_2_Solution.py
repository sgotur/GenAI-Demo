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
import time
import os
import sys
import subprocess
import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events


key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']


s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
bedrock = boto3.client('bedrock',region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)
s3_bucket = 'genai-tests'
s3_prefix = 'speech2solution'

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
ant_name = os.environ['ant_name']
p_summary = ''
st.set_page_config(page_title="GenAI Speech 2 Solution", page_icon="microphone")

languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']

if 's2s_transcript' not in st.session_state:
    st.session_state['s2s_transcript'] = ' '
if 'disc_summary' not in st.session_state:
    st.session_state['disc_summary'] = None
if 'design_spec' not in st.session_state:
    st.session_state['design_spec'] = None
if 'code' not in st.session_state:
    st.session_state['code'] = None
if 'docker' not in st.session_state:
    st.session_state['docker'] = None
if 'deps' not in st.session_state:
    st.session_state['deps'] = None

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)


st.markdown("# Dont build, just talk")
st.sidebar.header("GenAI Speech 2 Solution")
st.sidebar.write('''
        ### Say something or have a meeting to discuss \n\n
        Serverless order management solution using AWS Lambda python and DynamoDB \n
        utility predictions using Amazon Forecast and AWS Lambda \n
        Dynamically calculate hotel room pricing using rate types, seasonality, location and length of stay \n
    ''')

fms = ['Anthropic Claude', 'Bedrock Claude', 'Bedrock Titan']
default_model = fms.index('Anthropic Claude')

default_lang_ix = languages.index('English')
language = st.selectbox(
    '**Select an output language.**',
    options=languages, index=default_lang_ix)

model = ''
model = st.sidebar.selectbox(
    'Select a FM',
    options=fms, index=default_model)
  
def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=2048,
    )
    return resp['completion']


def call_bedrock_titan(prompt_text, max_token_count=2048, temperature=0.7, top_k=250, top_p=1, stop_sequences=[]):
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

def call_bedrock_claude(prompt_text, max_tokens_to_sample=2048, temperature=0.7, top_k=250, top_p=1):
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

def call_falcon(query):
    payload = {
    "inputs": query,
        "parameters": {
            "max_new_tokens": 2048,
            "return_full_text": True,
            "do_sample": True,
            "temperature": 0.5,
            "repetition_penalty": 1.03,
            "top_p": 0.9,
            "top_k":1,
            "stop": ["<|endoftext|>", "</s>"]
        }
    }


models = {
    "bedrock titan" : call_bedrock_titan,
    "bedrock claude" : call_bedrock_claude,
    "sagemaker falcon" : call_falcon,
    "anthropic claude": call_anthropic
}

st.write("**Instructions:** \n 1. Click Speak away and start talking about what you want to build \n 2. When you are ready to see results, click Generate \n 3. Demo live generates discussion summary, architecture, design specs and code based on what it hears \n 4. Click Speak away to continue talking, you can negate or augment what you already said, and click Generate again for updated content \n")

# Get the live transcript and stream print it
# Open microphone and start streaming

s2s_transcript = ''


stt_button_start = Button(label="Speak Away", width=100)

stt_button_start.js_on_event("button_click", CustomJS(code="""
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
 
    recognition.onresult = function (e) {
        var value = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
            }
        }
        if ( value != "") {
            document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
        }
    }
    recognition.start();
    """))

result = streamlit_bokeh_events(
    stt_button_start,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)

if result:
    if "GET_TEXT" in result:
        st.session_state.s2s_transcript += str(result.get("GET_TEXT"))
        #st.write(str(st.session_state.full_transcript))

# Now we go back to some Streamlit and GenAI magic
#full_transcript = 'serverless solution using AWS Lambda to read csv from S3 and write to amazon dynamoDB'
#st.write(st.session_state.full_transcript)
#if "generate" in st.session_state.s2s_transcript:
if st.session_state.s2s_transcript != ' ':
    if st.button('Generate'):
        #st.session_state.s2s_transcript = str(st.session_state.s2s_transcript).replace("generate","")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Discussion Summary", "Architecture Diagram", "Design Specifications", "Code", "Dockerfile", "Dependencies"])
        with tab1:
            query = "Summarize the transcript, identify action items, and expected outcomes in " + language + " for: " + str(st.session_state.s2s_transcript)
            func = models[model.lower()]
            answer = func(query)    
            st.write(answer)
            st.session_state.disc_summary = answer
        with tab2:        
            query = "Generate a full diagrams code using the python-diagrams package with proper import statements, correct indentations with no prefix or explanations. Use the correct syntax with correct APIs and class names the from python-diagrams package. The global context should be set as: with Diagram('architecture'): after the import statements "+ str(st.session_state.s2s_transcript)+ ". Do not include any python tags or descriptions."
            func = models[model.lower()]
            answer = func(query)
            st.write("**ETA soon to see arch digram live")
            answer = answer.replace("->",">>").replace("<-","<<").lstrip()
            answer = answer.replace("DynamoDB","Dynamodb")
            st.code(answer)
            f = open("architecture.py", "w")
            f.write(answer)
            f.close()
            with open("architecture.py", "rb") as s3_f:
                s3.upload_fileobj(s3_f, s3_bucket, s3_prefix+'/architecture.py')
            #os.system('black architecture.py')
            #subprocess.run(["black", 'architecture.py'])
            #arch_img = 'architecture.png'
            #subprocess.run([f"{sys.executable}", 'architecture.py'])
            #time.sleep(2)
            #st.image(arch_img)
        with tab3:        
            query = "Generate a design specification with team names as owners for building the solution discussed in " + language + " for: " + str(st.session_state.s2s_transcript)
            func = models[model.lower()]
            answer = func(query)
            st.write(answer)
            st.session_state.design_spec = answer
        with tab4:
            if st.session_state.disc_summary is not None:        
                query = "Generate and return the full and complete code for each module without prefix statements or explanations using the programming language and programming framework requested considering all the action items in "+ str(st.session_state.disc_summary)
                func = models[model.lower()]
                answer = func(query)
                st.code(answer)
                st.session_state.code = answer
        with tab5:   
            if st.session_state.code is not None:     
                query = "Generate and return the full Dockerfile for this solution code without prefix or explanations: "+ str(st.session_state.code)
                func = models[model.lower()]
                answer = func(query)
                st.code(answer)
                st.session_state.docker = answer
            else:
                st.write("The dockerfile will be displayed after code is fully generated")
        with tab6:
            if st.session_state.code is not None:        
                query = "Generate and return the requirements.txt and packages.txt necessary to create the docker image for this solution code without prefix or explanations "+ str(st.session_state.code)
                func = models[model.lower()]
                answer = func(query)
                st.code(answer)
                st.session_state.deps = answer
            else:
                st.write("The dependent libraries will be displayed after code is fully generated")    






