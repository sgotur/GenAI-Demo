import json
import boto3
import streamlit as st
import datetime
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
import pandas as pd
import uuid
import ai21
import string
import anthropic
import os

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
prefix = 'personalize/hotel-recommendations'

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
personalize = boto3.client('personalize',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
personalize_runtime = boto3.client('personalize-runtime',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
campaign_arn = os.environ['campaign_arn']
ant_name = os.environ['ant_name']

st.set_page_config(page_title="AI Product Recommender", page_icon="genie")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Get Personalized Recommendations")
st.sidebar.header("AI Product Recommender")

st.sidebar.markdown("### Make your pick")
industry = ''
industry = st.sidebar.selectbox(
    'Select an industry',
    ('Hospitality', 'Retail'))

if industry == 'Hospitality':
    st.sidebar.markdown('''
        ### Find similar hotels to London city hotels such as: \n\n
        - The Dorchester \n
        - The Savoy \n
        - The Rembrandt \n
''')
elif industry == 'Retail':
    st.sidebar.markdown('''
        ### Coming soon...
''')

# The industry variable is currently for illustration purposes only
# The current campaign ARN is for SIMS recipe for Hotel recommendations
# As we add more industries we will add more campaigns correspondingly and modify the code below

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


def GetRecommendations(query):
    recom = personalize_runtime.get_recommendations(
        campaignArn=campaign_arn,
        itemId=str(meta_df.query('BRAND == @query')['ITEM_ID'].item()),
        numResults=5
    )
    return recom

def GetAnswers(query):
    
    sentiment = 'POSITIVE'
    pii_list = []
    if query and len(query) > 5:
        sentiment = comprehend.detect_sentiment(Text=query, LanguageCode='en')['Sentiment']
        resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode='en')
        for pii in resp_pii['Entities']:
            if pii['Type'] not in ['NAME', 'AGE', 'ADDRESS','DATE_TIME']:
                pii_list.append(pii['Type'])
        if len(pii_list) > 0:
            answer = "I am sorry but I found PII entities " + str(pii_list) + " in your query. Please remove PII entities and try again."
            return answer
    else:
        answer = "Insufficient search query. Please expand the query and/or add more context."
    
    query_type = ''
    if "you" in query:
        query_type = "BEING"

    if query in ["cancel","clear"]:
        answer = 'Thanks come back soon...'
        return answer

    elif sentiment == 'NEGATIVE':
        answer = 'I apologize but I do not answer questions that are negatively worded or that concern me. Kindly rephrase your question and try again.'
        return answer
    
    elif query_type == "BEING":
        answer = 'I apologize but I do not answer questions that are negatively worded or that concern me. Kindly rephrase your question and try again.'
        return answer

    else:
        generated_text = ''
        if model.lower() == 'anthropic claude':   
            generated_text = call_anthropic(query)
            if generated_text != '':
                generated_text = generated_text.replace("$","\$")
                answer = str(generated_text)
            else:
                answer = 'Claude did not find an answer to your question, please try again'            

        return answer

s3.download_file(bucket, prefix+'/hotel-metadata.csv', 'metadata.csv')
meta_df = pd.read_csv('metadata.csv', header=0)

if industry == 'Hospitality':
    st.write("**Instructions:** \n - Select a London city hotel name \n - You will get five recommendations for hotels nearby \n - Using the next search bar you can find more details about these hotels \n")
    brands = []
    for i, brand in enumerate(meta_df['BRAND']):
        if i < 5:
            brands.append(brand)
    input_text = st.selectbox(
    'Select a London city hotel',
    options=brands)
    if input_text != '':
        results = GetRecommendations(input_text)
        st.write('**Here are some similar hotels nearby:**')
        for item in results['itemList']:
            a = item['itemId']
            c1, c2, c3 = st.columns(3)
            c1.write(meta_df.query('ITEM_ID == @a')['BRAND'].item())
            c2.write('$'+str(meta_df.query('ITEM_ID == @a')['PRICE'].item())+ ' per night')
            c3.write(meta_df.query('ITEM_ID == @a')['DESCRIPTION'].item())
        
        search_text = st.text_input('**Ask me for more details**', key='search_text')
        if search_text != '':
            details = GetAnswers(search_text)
            details = details.replace("$","\$")
            st.write(details)
else:
    st.write("### :blue[Demo coming soon, stay tuned...] \n")