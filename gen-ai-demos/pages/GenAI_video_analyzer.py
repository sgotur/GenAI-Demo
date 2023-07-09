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
import time
import string
import anthropic
import os

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

if 'vid_summary' not in st.session_state:
    st.session_state['vid_summary'] = None
if 'vid_labels' not in st.session_state:
    st.session_state['vid_labels'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
rekognition = boto3.client('rekognition',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Korean', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
video_prefix = 'streamlit_video'
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']
vtypes = ['mp4']

st.set_page_config(page_title="GenAI Video Analyzer", page_icon="flashlight")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Derive insights from videos")
st.sidebar.header("GenAI Video Analyzer")


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

def GetAnswers(labels, query):
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
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'

    elif query_type == "BEING":
        answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'
            
    else:
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic(labels+'. Explain in '+language+' from these labels: '+ query.strip("query:"))
            if generated_text != '':
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again'   
    return answer          


#upload audio file to S3 bucket
def upload_video_detect_labels(uploaded_vid):
    summary = ''
    label_text = ''
    response = ''
    start_res = rekognition.start_label_detection(
        Video={
            'S3Object': {
                'Bucket':bucket,
                'Name':video_prefix+'/'+uploaded_vid.name
            }
        })

    with st.spinner('Starting Amazon Rekognition video label detection...'): 
        finish = False
        st.markdown("This should take a couple of minutes. Please check out these [great features of Amazon Rekognition](https://aws.amazon.com/rekognition) in the meanwhile...")
        while finish == False:
            time.sleep(30)
            response = rekognition.get_label_detection(
                JobId = start_res['JobId']
                    )
            job_status = response['JobStatus']
            st.write("Status of your video analysis is: " + job_status)
            if job_status.upper() == 'SUCCEEDED':
                st.success("Video analysis status: " + job_status)
                finish = True
                break
            elif job_status.upper() == 'FAILURE':
                st.failure("Video analysis failed, please check the job status for "+str(start_res['JobId']) + " in the Amazon Rekognition console")
                finish = True
                break
    if job_status.upper() != 'FAILURE':
        for label in response['Labels']:
            label_text += label['Label']['Name'] + ' ' 

    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic('Summarize the video in '+language+' in 200 words from these labels: '+ label_text)
        if generated_text != '':
            generated_text.replace("$","USD")
            summary = str(generated_text)+' '
        else:
            summary = 'Claude did not find an answer to your question, please try again'    
        return label_text, summary    


st.write("**Instructions:** \n - Browse and select your video. You can [download these samples](https://amazon.awsapps.com/workdocs/index.html#/folder/98bd14e37279244b29dc119ba91612ccddeb129e0cb4d4e6b672c38cb625c90f) or use your own \n - You will see a summary based on labels identifed by Amazon Rekognition \n - Type your query in the search bar to get video insights")

c1, c2 = st.columns(2)
c1.subheader("Upload your video")
uploaded_vid = c1.file_uploader("**Select a video**", type=vtypes)
default_lang_ix = languages.index('English')
c2.subheader("Select an output language")
language = c2.selectbox(
    'Only Alpha and Beta quadrant languages supported. For new requests, please contact C-3PO',
    options=languages, index=default_lang_ix)
vid_summary = ''
vid_labels = ''
p_summary = ''
if uploaded_vid is not None:       
    c1.success(uploaded_vid.name + ' is ready for upload')
    if c1.button('Upload'):
        s3.upload_fileobj(uploaded_vid, bucket, video_prefix+'/'+uploaded_vid.name)
        st.success('Video uploaded')
        vid_labels, vid_summary = upload_video_detect_labels(uploaded_vid)
        vid_summary = vid_summary.replace("$","\$")
    # Check job status
    

if len(vid_summary) >= 5:
    st.session_state['vid_summary'] = vid_summary
if len(vid_labels) >= 5:
    st.session_state['vid_labels'] = vid_labels

if st.session_state.vid_summary:
    if uploaded_vid is not None:
        st.video(uploaded_vid)
    st.markdown('**Video summary**: \n')

    st.write(str(st.session_state['vid_summary']))
    if model.lower() == 'anthropic claude':  
        p_text = call_anthropic('Generate three prompts to query the summary: '+ st.session_state.vid_summary)
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
        result = GetAnswers(vid_labels,input_text)
        result = result.replace("$","\$")
        st.write(result)





