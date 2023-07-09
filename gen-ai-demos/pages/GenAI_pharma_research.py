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
ls_index_id = os.environ['ls_index_id']
#tax_index_id = os.environ['tax_index_id']
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']

st.set_page_config(page_title="GenAI Pharma Research", page_icon="seedling")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Level up your pharma knowledge")
st.sidebar.header("GenAI Pharma Research")
#st.sidebar.markdown("### Make your pick")
industry = 'Life Sciences'
#industry = st.sidebar.selectbox(
#    'Select an industry',
#    ('Life Sciences', ''))

if industry == 'Life Sciences':
    st.sidebar.markdown('''
        ### Example prompts you can try \n\n
        What is the prescription medication for myeloma? \n
        What are the treatment options for plaque psoriasis? \n
    ''')

if industry.lower() == 'life sciences':
    index_id = ls_index_id
else:
    index_id = ''

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
                    if len(query_result['DocumentAttributes']) > 1:
                        sources_list.append("Document:{} and Page:{}".format(query_result['DocumentAttributes'][0]['Value']['StringValue'], query_result['DocumentAttributes'][1]['Value']['LongValue']))
                    else:
                        sources_list.append("Source:{}".format(query_result['DocumentAttributes'][0]['Value']['StringValue']))

            if query_result["Type"]=="DOCUMENT":
                d += 1
                if "DocumentTitle" in query_result:
                    document_title = query_result["DocumentTitle"]["Text"]
                    #print("Title: " + document_title)
                if d <= 1:
                    document_text += query_result["DocumentExcerpt"]["Text"]
                    if len(query_result['DocumentAttributes']) > 1:
                        sources_list.append("Document:{} and Page:{}".format(query_result['DocumentAttributes'][0]['Value']['StringValue'], query_result['DocumentAttributes'][1]['Value']['LongValue']))
                    else:
                        sources_list.append("Source:{}".format(query_result['DocumentAttributes'][0]['Value']['StringValue']))

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
    
    pii_list = []
    sentiment = 'POSITIVE'
    resp_pii = ''

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
    


    if query in ["cancel","clear"]:
        answer = 'Thanks come back soon...'
        return answer

    #elif sentiment == 'NEGATIVE':
    #    answer = 'I apologize but I do not answer questions that are negatively worded or that concern me. Kindly rephrase your question and try again.'
    #    return answer
    
    else:
        generated_text = ''
        # Kendra calls
        results = {}
        if index_id != '':
            results = call_Kendra(query, index_id)
        else:
            results['snippet'] = ' '
        
        if model.lower() == 'anthropic claude':
            if results['snippet'] == ' ':    
                generated_text = call_anthropic("Explain in maximum 200 words without bullets: " + query)
                if generated_text != '':
                    generated_text = generated_text.replace("$","\$")
                    answer = "I did not find an answer in my enterprise knowledge base, but here is a general answer: \n"+str(generated_text)
                else:
                    answer = 'Claude did not find an answer to your question, please try again'
            else:
                generated_text = call_anthropic(results['snippet']+'. Answer from this text:'+query)
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

st.write("**Instructions:** \n - Type a query using suggested prompts \n - Review the answers \n")
input_text = st.text_input('**What are you searching for?**', key='text')
result = ''
if input_text != '':
    result = GetAnswers(input_text)
    result = result.replace("$","\$")
    st.write(result)

if result != '':
    if model.lower() == 'anthropic claude':  
        p_text = call_anthropic('Generate three prompts for the text. Do not give answers: '+ result)
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