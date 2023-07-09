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
import os

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
s3_bucket = 'genai-tests'
s3_prefix = 'energy/upstream'

if 'img_summary' not in st.session_state:
    st.session_state['img_summary'] = None
if 'csv_summary' not in st.session_state:
    st.session_state['csv_summary'] = None
if 'new_summary' not in st.session_state:
    st.session_state['new_summary'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
rekognition = boto3.client('rekognition',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['br_endpoint_name']
ant_name = os.environ['ant_name']
p_summary = ''
#languages = ['English', 'Spanish', 'German', 'Italian', 'French', 'Portugese', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
st.set_page_config(page_title="GenAI Energy Upstream Analyzer", page_icon="oil_drum")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Simplify Upstream Analysis tasks")
st.sidebar.header("GenAI Energy Upstream Analyzer")
values = [1, 2, 3, 4, 5]
default_ix = values.index(3)
p_count = st.sidebar.selectbox('Select the count of auto-prompts to consider', values, index=default_ix)


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

def GetAnswers(original_text, query):
    pii_list = []
    sentiment = comprehend.detect_sentiment(Text=query, LanguageCode='en')['Sentiment']
    resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode='en')
    for pii in resp_pii['Entities']:
        if pii['Type'] not in ['NAME', 'AGE', 'ADDRESS','DATE_TIME']:
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
    #    answer = 'I do not answer questions that lean towards negativity or that concern me at this time. Kindly rephrase your question and try again.'

    elif query_type == "BEING":
        answer = 'I do not answer questions that concern me at this time. Kindly rephrase your question and try again.'
            
    else:
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic(original_text+'. Answer from this text: '+ query.strip("query:"))
            if generated_text != '':
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again'   
    return answer          


#upload image file to S3 bucket
def upload_image_detect_labels(bytes_data):
    summary = ''
    label_text = ''
    response = rekognition.detect_labels(
        Image={'Bytes': bytes_data},
        Features=['GENERAL_LABELS']
    )
    text_res = rekognition.detect_text(
        Image={'Bytes': bytes_data}
    )

    celeb_res = rekognition.recognize_celebrities(
        Image={'Bytes': bytes_data}
    )

    for celeb in celeb_res['CelebrityFaces']:
        label_text += celeb['Name'] + ' ' 

    for text in text_res['TextDetections']:
        label_text += text['DetectedText'] + ' '

    for label in response['Labels']:
        label_text += label['Name'] + ' '

    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic('Identify and explain the equipment used in this image in ' + language+' in 500 words from these labels: '+ label_text)
        if generated_text != '':
            generated_text.replace("$","USD")
            summary = str(generated_text)+' '
        else:
            summary = 'Claude did not find an answer to your question, please try again'    
        return summary    

def upload_csv_get_summary(s3_file_name):
    summary = ''
    # download the csv file for the selection
    s3.download_file(s3_bucket, s3_prefix+'/'+s3_file_name, s3_file_name)
    with open(s3_file_name, 'rb') as f:
        contents = f.read()
    new_contents = contents[:10000].decode('utf-8')

    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic('Perform a well log interpretation in 500 words in '+language+ ' for this text: '+ new_contents)
        if generated_text != '':
            summary = str(generated_text)+' '
            summary = summary.replace("$","\$")
        else:
            summary = 'Claude did not find an answer to your question, please try again'    
    return new_contents, summary    

def auto_summarize(p1, s3_file_name, file_type):
    auto_in = ''
    new_summary = ''
    if file_type == 'csv':
        # download the csv file for the selection
        s3.download_file(s3_bucket, s3_prefix+'/'+s3_file_name, s3_file_name)
        with open(s3_file_name) as f:
            contents = f.read()
        auto_in = contents[:10000]
    elif file_type == 'image':
        auto_in = st.session_state.img_summary    
    if auto_in:
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic('Answer the questions ' + p_summary + ' from the text in '+language+': ' + auto_in)
            generated_text = call_anthropic(p1 + auto_in + ' ' + generated_text)
            if generated_text != '':
                new_summary = str(generated_text)+' '
                new_summary = new_summary.replace("$","\$")
            else:
                new_summary = 'Claude did not find an answer to your question, please try again'    
    
    return new_summary    

st.write("**Instructions:** \n - Browse and select your input file. You can [download these selection of samples](https://amazon.awsapps.com/workdocs/index.html#/folder/6580f34096d382e1da0400524d215acf84c65231b9041b6825d780651da4d34b) or use your own \n - Select an output language and click Upload \n - You will see a text summary \n - Select the number of prompts for auto-summarization from the left pane \n - You will see a new summary \n - Type your query in the search bar to get more insights")

c1, c2 = st.columns(2)
c1.subheader("Upload your file")
uploaded_img = c1.file_uploader("**Select a file**", type=['png','jpg','jpeg','csv'])
default_lang_ix = languages.index('English')
c2.subheader("Select an output language")
language = c2.selectbox(
    'Only Alpha and Beta quadrant languages supported. For new requests, please contact C-3PO',
    options=languages, index=default_lang_ix)
new_summary = ''
img_summary = ''
csv_summary = ''
file_type = ''
new_contents = ''
if uploaded_img is not None:
    if 'jpg' in uploaded_img.name or 'png' in uploaded_img.name or 'jpeg' in uploaded_img.name:
        file_type = 'image'        
        c1.success(uploaded_img.name + ' is ready for upload')
        if c1.button('Upload'):
            with st.spinner('Uploading image file and starting summarization with Amazon Rekognition label detection...'):
                img_summary = upload_image_detect_labels(uploaded_img.getvalue())
                img_summary = img_summary.replace("$","\$")
                if len(img_summary) > 5:
                    st.session_state['img_summary'] = img_summary
                st.success('File uploaded and summary generated')
    elif 'csv' in uploaded_img.name:
        file_type = 'csv'            
        c1.success(uploaded_img.name + ' is ready for upload')
        if c1.button('Upload'):
            with st.spinner('Uploading file and starting summarization...'):
                #stringio = StringIO(uploaded_img.getvalue().decode("utf-8"))
                s3.upload_fileobj(uploaded_img, s3_bucket, s3_prefix+'/'+uploaded_img.name)
                new_contents, csv_summary = upload_csv_get_summary(uploaded_img.name)
                csv_summary = csv_summary.replace("$","\$")
                if len(csv_summary) > 5:
                    st.session_state['csv_summary'] = csv_summary
                st.success('File uploaded and summary generated')
    else:
        st.failure('Incorrect file type provided. Please select either a JPG or PNG file to proceed')

    # Check job status
    


p1 = ''
if uploaded_img is not None:
    if st.session_state.img_summary:
        if len(st.session_state.img_summary) > 5:
            if uploaded_img is not None:
                st.image(uploaded_img)
            st.markdown('**Image summary**: \n')

            st.write(str(st.session_state['img_summary']))
            if model.lower() == 'anthropic claude':  
                p_text = call_anthropic('Generate'+str(p_count)+'prompts to query the summary: '+ st.session_state.img_summary)
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
            #p1 = 'Generate an updated summary in 500 words: '
            #new_summary = auto_summarize(p1, uploaded_img.name, file_type)
            #if len(new_summary) > 5:
            #   st.session_state['new_summary'] = new_summary
            #st.write("**Updated summary based on auto-prompts**")
            #st.write(st.session_state.new_summary)
    if st.session_state.csv_summary:
        if len(st.session_state.csv_summary) > 5:
            st.markdown('**Summary**: \n')
            st.write(str(st.session_state['csv_summary']))
            if model.lower() == 'anthropic claude':

                p_text = call_anthropic('Generate'+str(p_count)+'prompts to query the text: '+ st.session_state.csv_summary)
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
            p1 = 'Perform a well log interpretation in 500 words in ' +language+': '
            new_summary = auto_summarize(p1, uploaded_img.name, file_type)
            if len(new_summary) > 5:
                st.session_state['new_summary'] = new_summary
            st.write("**Updated well log interpretation based on auto-prompts**")
            st.write(st.session_state.new_summary)



input_text = st.text_input('**What insights would you like?**', key='text')
if input_text != '':
    if st.session_state.img_summary:
        result = GetAnswers(st.session_state.img_summary,input_text)
        result = result.replace("$","\$")
        st.write(result)
    elif st.session_state.csv_summary:
        s3.download_file(s3_bucket, s3_prefix+'/'+uploaded_img.name, uploaded_img.name)
        with open(uploaded_img.name, 'rb') as f:
            contents = f.read()
        new_contents = contents[:10000].decode('utf-8')
        result = GetAnswers(new_contents,input_text)
        result = result.replace("$","\$")
        st.write(result)
    else:
        st.write("I am sorry it appears you have not uploaded any files for analysis. Can you please upload a file and then try again?")        





