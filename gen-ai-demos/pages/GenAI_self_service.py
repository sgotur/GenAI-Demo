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
from streamlit_chat import message
from boto3.dynamodb.conditions import Key
import os

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
lex = boto3.client('lexv2-runtime',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
dynamodb = boto3.client('dynamodb',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
ddb = boto3.resource('dynamodb',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
table = ddb.Table('genai-selfservice-history')


# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']

#languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Italian', 'Korean', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']

if 'SSsessionID' not in st.session_state:
    st.session_state['SSsessionID'] = str(uuid.uuid4())
if 'SScount' not in st.session_state:
    st.session_state['SScount'] = 0

def get_old_chats():
    res = table.query(
    KeyConditionExpression=Key('session_id').eq(str(st.session_state.SSsessionID))
    )
    return res['Items']

def store_chat(session, turn, question, answer):
    dynamodb.put_item(
        TableName="genai-selfservice-history",
        Item={
            'session_id': {'S': str(session)},
            'turn': {'S': str(turn)},
            'question': {'S': str(question)},
            'answer': {'S': str(answer)}
            }
        )

def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=1024,
    )
    return resp['completion']

st.set_page_config(page_title="GenAI Self Service Kiosk", page_icon="question")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# GenAI Self Service Kiosk")

st.sidebar.header("GenAI Self Service Kiosk")
st.sidebar.markdown("### Make your pick")
industry = ''
industry = st.sidebar.selectbox(
    'Select an industry',
    ('Insurance', 'Banking'))
model = 'Anthropic Claude'

if industry == 'Insurance':
    st.sidebar.markdown('''
        ### Available requests: \n\n
        - pay auto insurance premium \n
        - get a policy quote \n
        - make a claim \n
        - **Use this sample data:** [Insurance Sample Data](https://lex-usecases-templates.s3.amazonaws.com/Auto_Insurance_Sample_Data.html)
''')
elif industry == 'Banking':
    st.sidebar.markdown('''
        ### Available requests: \n\n
        - Activate my card \n
        - I want a new card \n
        - Pay my card bill \n
        - **Use this sample data:** [Banking Sample Data](https://lex-usecases-templates.s3.amazonaws.com/Card_Services_Sample_Data.html)
''')

st.write("**Instructions:** \n - Type a request from available requests on the left to get started \n - You will be guided through a series of Q&A steps \n - Click a tab to view the chat window or chat summary :smiley \n")

session_id = "genai-ins-demo"
locale_Id = "en_US"

if industry == 'Insurance':
    bot_id = '8IMD4GLISQ'
    bot_alias_id = 'TSTALIASID'
    session_id = "genai-ins-demo"
elif industry == 'Banking':
    bot_id = 'L498TSB4LD'
    bot_alias_id = 'TSTALIASID'
    session_id = "genai-bank-demo"    

answer = ''
chat_summary = ''
old_chats = ''
input_text = st.text_input('**Lets start chatting**', key='text')
if input_text != '':
    tab1, tab2, tab3 = st.tabs(["Chat Window", "Chat Summary", "Fraud Analysis"])
    with tab1:
        response = lex.recognize_text(
            botId=bot_id,
            botAliasId=bot_alias_id,
            localeId=locale_Id,
            sessionId=session_id,
            text=input_text,
            )
        result = response['messages'][0]['content']
        st.session_state.SScount = int(st.session_state.SScount) + 1
        old_chats = get_old_chats()
        for chat in old_chats:
                message(chat['question'], is_user=True, key=str(uuid.uuid4()))
                message(chat['answer'], key=str(uuid.uuid4()))
        message(input_text, is_user=True)
        message(result)
        store_chat(st.session_state.SSsessionID, st.session_state.SScount, input_text, result)
    with tab2:
        default_lang_ix = languages.index('English')
        language = st.selectbox(
            'Select an output language',
            options=languages, index=default_lang_ix)
        old_chats = get_old_chats()
        if old_chats:
            chat_summary = call_anthropic("Summarize the following chat messages in "+language+" in 200 words: " + str(old_chats))
            chat_summary = chat_summary.replace("$","\$")
            st.write(chat_summary)
    with tab3:
        fraud_results = call_anthropic("Identify fraudulent data in the following chat messages and explain why: "+str(old_chats))
        fraud_results = fraud_results.replace("$","\$")
        st.write(fraud_results)

