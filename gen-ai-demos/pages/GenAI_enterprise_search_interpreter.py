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
kendra = boto3.client("kendra",region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

# Get environment variables
fsi_index_id = os.environ['fsi_index_id']
energy_index_id = os.environ['energy_index_id']
travel_index_id = os.environ['travel_index_id']
legal_index_id = os.environ['legal_index_id']
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']

st.set_page_config(page_title="GenAI Enterprise Search and Interpreter", page_icon="mag_right")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# GenAI Enterprise Search")
st.sidebar.header("GenAI search and interpret enterprise content")
st.sidebar.markdown("### Make your pick")
industry = ''
industry = st.sidebar.selectbox(
    'Select an industry',
    ('Financial Services', 'Energy', 'Legal', 'Travel and Transport'))

if industry == 'Financial Services':
    st.sidebar.markdown('''
        ### Example prompts you can try \n\n
        What is a company's EPS and what does it mean? \n
        Why are EU members not aligned on fiscal policy? \n
    ''')
elif industry == 'Energy':
    st.sidebar.markdown('''
        ### Example prompts you can try \n\n
        List the steps how oil is refined? \n
        What are the risks in hydrocarbon transport? \n
    ''')
elif industry == 'Travel and Transport':
    st.sidebar.markdown('''
        ### Example prompts you can try \n\n
        What is the maximum range for Airbus A330? \n
        What is the final approach speed for A350? \n
    ''')
elif industry == 'Legal':
    st.sidebar.markdown('''
        ### Example prompts you can try \n\n
        Who are the buyers and sellers mentioned in the contract? \n
        What is the closing date for the contract? \n
        What documents do the buyers need to approve? \n
    ''')
model = 'Anthropic Claude'
#model = st.sidebar.selectbox(
#    'Select a LLM',
#    ('J2 Jumbo Instruct', 'Anthropic Claude'))


# Now using Jurassic Jumbo instruct
def call_jumbo(query,tokens,temp):
    print("query to J2 is: " + str(query))
    presence_penalty = {}
    count_penalty = {}
    frequency_penalty = {}
    presence_penalty['scale'] = '5.0'
    count_penalty['scale'] = '2.0'
    frequency_penalty['scale'] = '42.7'
    response = ai21.Completion.execute(sm_endpoint=tx_endpoint_name,
                    prompt=query,
                    minTokens=tokens,
                    maxTokens=500,
                    temperature=temp,
                    presencePenalty=presence_penalty,
                    countPenalty=count_penalty,
                    frequencyPenalty=frequency_penalty,
                    numResults=1,
                    topP=1,
                    topKReturn=0,
                    stopSequences=["##"])

    generated_text = response['completions'][0]['data']['text']    
    print("response from J2 is:" + str(generated_text))
    return generated_text

def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=1024,
    )
    return resp['completion']
def call_Kendra(query_string, index_id):
    d = 0
    a = 0
    answer_text = ''
    document_text = ''
    result_dict = {}
    sources_list = []
    result = ' '    
    if index_id != '':
        response = kendra.query(
                QueryText = query_string,
                IndexId = index_id)
    #print(response['ResultItems'])       
        
        for query_result in response["ResultItems"]:
            if query_result["Type"]=="ANSWER" or query_result["Type"]=="QUESTION_ANSWER":
                a += 1
                if a <= 1:
                    answer_text = query_result["DocumentExcerpt"]["Text"]
                    src = "Document:{} and Page:{}".format(query_result['DocumentAttributes'][0]['Value']['StringValue'], query_result['DocumentAttributes'][1]['Value']['LongValue'])
                    if src not in sources_list:
                        sources_list.append(src)
                    
                #print(answer_text)

            if query_result["Type"]=="DOCUMENT":
                d += 1
                if "DocumentTitle" in query_result:
                    document_title = query_result["DocumentTitle"]["Text"]
                    #print("Title: " + document_title)
                if d <= 1:
                    document_text += query_result["DocumentExcerpt"]["Text"]
                    sources_list.append("Document:{} and Page:{}".format(query_result['DocumentAttributes'][0]['Value']['StringValue'], query_result['DocumentAttributes'][1]['Value']['LongValue']))
                #print(document_text)
               
        result = answer_text +' '+document_text
        result = result.translate(str.maketrans('','',string.punctuation))
        result = result.replace('\n',' ')
        result_dict['snippet'] = result
        result_dict['sources'] = sources_list
    else:
        result_dict['snippet'] = ' '
        result_dict['sources'] = ' '    
    
    return result_dict


def GetAnswers(query):
    tokens=25
    temp=0.7
    pii_list = []
    sentiment = 'POSITIVE'
    if industry.lower() == 'financial services':
        index_id = fsi_index_id
    elif industry.lower() == 'energy':
        index_id = energy_index_id
    elif industry.lower() == 'travel and transport':
        index_id = travel_index_id
    elif industry.lower() == 'legal':
        index_id = legal_index_id
    else:
        index_id = ''

    if query and len(query) > 5:
        sentiment = comprehend.detect_sentiment(Text=query, LanguageCode='en')['Sentiment']
        resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode='en')
        for pii in resp_pii['Entities']:
            if pii['Type'] not in ['NAME','AGE','ADDRESS','DATE_TIME']:
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

    #elif sentiment == 'NEGATIVE':
    #    answer = 'I apologize but I do not answer questions that are negatively worded or that concern me. Kindly rephrase your question and try again.'
    #    return answer
    
    elif query_type == "BEING":
        answer = 'I apologize but I do not answer questions that are negatively worded or that concern me. Kindly rephrase your question and try again.'
        return answer

    else:
        generated_text = ''
        # Kendra calls
        results = call_Kendra(query.strip("query:"), index_id)
        # Based on model selected
        if model.lower() == 'anthropic claude':
            if results['snippet'] == ' ':    
                generated_text = call_anthropic(query.strip("query:"))
                if generated_text != '':
                    generated_text = generated_text.replace("$","\$")
                    answer = generated_text
                else:
                    answer = 'Claude did not find an answer to your question, please try again'
            else:  
                generated_text = call_anthropic(results['snippet']+'. Answer from this text:'+query.strip("query:"))
                sources = ''
                if generated_text != '':
                    generated_text = generated_text.replace("$","\$")
                    for txt in results['sources']:
                        if txt not in sources:
                            sources += '- '+txt+'\n'    
                    answer = str(generated_text)+'\n\n\n **My sources are**: \n' + sources
                    results['sources'] = '' 
                else:
                    answer = 'Claude did not find an answer to your question, please try again'      

        return answer

st.write("**Instructions:** \n - Type your query \n - You will get the top answer explained. If a match was not found in Amazon Kendra, you will get a general answer for your query \n")
input_text = st.text_input('**What are you searching for?**', key='text')
result = ''
if input_text != '':
    result = GetAnswers(input_text)
    result = result.replace("$","\$")
    st.write(result)

if result != '':
    if model.lower() == 'anthropic claude':  
        p_text = call_anthropic('Generate three prompts to query the text: '+ result)
        p_text1 = []
        p_text2 = ''
        if p_text != '':
            p_text = p_text.replace("$","\$")
            p_text1 = p_text.split('\n')
            for i,t in enumerate(p_text1):
                if i > 1:
                    p_text2 += t.split('?')[0]+'\n\n'
            p_summary = p_text2
    st.sidebar.markdown('### Suggested prompts for further insights \n\n' + 
            p_summary)