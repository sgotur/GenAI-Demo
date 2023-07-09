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

if 'img_summary' not in st.session_state:
    st.session_state['img_summary'] = None
if 'csv_summary' not in st.session_state:
    st.session_state['csv_summary'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
rekognition = boto3.client('rekognition',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
bedrock = boto3.client('bedrock',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
falcon_endpoint = os.environ['falcon_endpoint_name']
#br_endpoint_name = os.environ['br_endpoint_name']
ant_name = os.environ['ant_name']
p_summary = ''
st.set_page_config(page_title="GenAI Content Explorer", page_icon="sparkles")

st.markdown("## Yosemite is in its formative stages, ETA soon")
st.sidebar.header("GenAI Content Explorer")
values = [1, 2, 3, 4, 5]
default_ix = values.index(3)
atypes = ['csv', 'pptx', 'rtf','xls','xlsx','txt', 'pdf', 'png','jpg','jpeg','doc', 'docx', 'json','ipynb','py','java']
ftypes = ['csv', 'pptx', 'rtf','xls','xlsx','txt', 'pdf', 'doc', 'docx', 'json','ipynb','py','java']
itypes = ['png','jpg','jpeg',]
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Korean', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
p_count = st.sidebar.selectbox('Select the count of auto-prompts to consider', values, index=default_ix)

models = ['Bedrock Jurassic-2','Bedrock Claude', 'SageMaker Falcon']
default_model = models.index('Bedrock Claude')
default_ix = values.index(3)
model = st.sidebar.selectbox('Select a FM', models, index=default_model)

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
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=falcon_endpoint, ContentType='application/json', Body=json.dumps(payload).encode('utf-8'))
    model_predictions = json.loads(response['Body'].read())
    falcon_resp = str(model_predictions[0]['generated_text'][len(query):])
    return falcon_resp

models = {
    "Bedrock Jurassic-2" : call_bedrock_jurassic,
    "Bedrock Claude" : call_bedrock_claude,
    "SageMaker Falcon" : call_falcon
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
        answer = 'It was swell chatting with you. Goodbye for now'
    
    #elif sentiment == 'NEGATIVE':
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'        
    else:
        generated_text = ''
        query = original_text+'. Answer from the full text and all pages of this text concisely in 100 words or less without hallucination or making illogical statement sticking only to the terminologies and context in this text: '+ query.strip("query:")
        func = models[model]
        answer = func(query) 
   
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
        if text['Confidence'] > 90:
            label_text += text['DetectedText'] + ' '

    for label in response['Labels']:
        if label['Confidence'] > 90:
            label_text += label['Name'] + ' '

    query = 'Explain the contents of this image in 300 words from these labels in ' +language+ ': '+ label_text
    func = models[model]
    summary = func(query) 

    return label_text, summary    


def upload_csv_get_summary(file_type, s3_file_name):
    summary = ''
    # download the csv file for the selection
    s3.download_file(s3_bucket, s3_prefix+'/'+s3_file_name, s3_file_name)
    
    if file_type not in ['py','java','ipynb','pdf']:
        contents = textract.process(s3_file_name).decode('utf-8')
        new_contents = contents[:50000].replace('$','\$')
    elif file_type == 'pdf':
        contents = readpdf(s3_file_name)
        new_contents = contents[:50000].replace("$","\$")
    else:
        with open(s3_file_name, 'rb') as f:
            contents = f.read()
        new_contents = contents[:10000].decode('utf-8')

    #lang = comprehend.detect_dominant_language(Text=new_contents)
    #lang_code = str(lang['Languages'][0]['LanguageCode']).split('-')[0]
    #if lang_code in ['en']:
    #    resp_pii = comprehend.detect_pii_entities(Text=new_contents, LanguageCode=lang_code)
    #    immut_summary = new_contents
    #    for pii in resp_pii['Entities']:
    #        if pii['Type'] not in ['NAME', 'AGE','ADDRESS','DATE_TIME']:
    #            pii_value = immut_summary[pii['BeginOffset']:pii['EndOffset']]
    #            new_contents = new_contents.replace(pii_value, str('PII - '+pii['Type']))

    new_contents = re.sub('[^A-Za-z0-9]+', ' ', new_contents)
    query = 'Create a 300 words summary from all the pages or slides of this document sticking to the terminologies and keywords mentioned in this document without hallucination or expanding the context in ' +language+ ': '+ new_contents
    func = models[model]
    summary = func(query) 
        
    return new_contents, summary    


st.write("**Instructions:** \n 1. Browse and select an input file in one of the supported formats. You can [download this selection of samples](https://amazon.awsapps.com/workdocs/index.html#/folder/f0e3628de0758aaef1bf4a51ebda6f53d5762e788adaaccac1d2aaaef09d45e3) or use your own \n 2. Select an output language, default is English \n 3. Click Upload button \n 4. You will see a text summary \n 5. Type your query in the search bar to get more insights")

c1, c2 = st.columns(2)
c1.subheader("Upload your file")
uploaded_img = c1.file_uploader("**Select a file**", type=atypes)
default_lang_ix = languages.index('English')
c2.subheader("Select an output language")
language = c2.selectbox(
    'Only Alpha and Beta quadrant languages supported. For new requests, please contact C-3PO',
    options=languages, index=default_lang_ix)
img_summary = ''
csv_summary = ''
label_text = ''
file_type = ''
new_contents = ''
if uploaded_img is not None:
    file_type = str(uploaded_img.name).split('.')[1]
    if str(uploaded_img.name).split('.')[1] in itypes:
        c1.success(uploaded_img.name + ' is ready for upload')
        if c1.button('Upload'):
            with st.spinner('Uploading image file and starting summarization with Amazon Rekognition label detection...'):
                label_text, img_summary = upload_image_detect_labels(uploaded_img.getvalue())
                img_summary = img_summary.replace("$","\$")
                if len(img_summary) > 5:
                    st.session_state['img_summary'] = img_summary
                st.success('File uploaded and summary generated')
    elif str(uploaded_img.name).split('.')[1] in ftypes:
    #elif 'csv' in uploaded_img.name or 'txt' in uploaded_img.name:
        #st.session_state.csv_summary = None            
        c1.success(uploaded_img.name + ' is ready for upload')
        if c1.button('Upload'):
            with st.spinner('Uploading file and starting summarization...'):
                #stringio = StringIO(uploaded_img.getvalue().decode("utf-8"))
                s3.upload_fileobj(uploaded_img, s3_bucket, s3_prefix+'/'+uploaded_img.name)
                new_contents, csv_summary = upload_csv_get_summary(file_type, uploaded_img.name)
                csv_summary = csv_summary.replace("$","\$")
                if len(csv_summary) > 5:
                    st.session_state['csv_summary'] = csv_summary
                new_contents = new_contents.replace("$","\$")
                st.success('File uploaded and summary generated')
    else:
        st.failure('Incorrect file type provided. Please check and try again')


p1 = ''
if uploaded_img is not None:
    if st.session_state.img_summary:
        if len(st.session_state.img_summary) > 5:
            st.image(uploaded_img)
            st.markdown('**Image summary**: \n')
            st.write(str(st.session_state['img_summary']))
    elif st.session_state.csv_summary:
        if len(st.session_state.csv_summary) > 5:
            st.markdown('**Text Summary**: \n')
            st.write(str(st.session_state.csv_summary).replace("$","\$"))
            #if model.lower() == 'anthropic claude':
            #    p_text = call_bedrock('Generate'+str(p_count)+'prompts to query the text: '+ st.session_state.csv_summary, model_type)
            #    p_text1 = []
            #    p_text2 = ''
            #    if p_text != '':
            #        p_text.replace("$","\$")
            #        p_text1 = p_text.split('\n')
            #        for i,t in enumerate(p_text1):
            #            if i > 1:
            #                p_text2 += t.split('\n')[0]+'\n\n'
            #        p_summary = p_text2
            #st.sidebar.markdown('### Generated auto-prompts \n\n' + 
            #            p_summary)
    #p1 = 'Perform a well log interpretation in 500 words: '
    #new_summary = auto_summarize(p1, uploaded_img.name, file_type)
    #st.write("**Updated well log interpretation based on auto-prompts**")
    #st.write(new_summary)



input_text = st.text_input('**What insights would you like?**', key='text')
if input_text != '':
    if st.session_state.csv_summary:
        s3.download_file(s3_bucket, s3_prefix+'/'+uploaded_img.name, uploaded_img.name)
        if file_type not in ['py','java','ipynb','pdf']:
            contents = textract.process(uploaded_img.name).decode('utf-8')
            new_contents = contents[:10000].replace('$','\$')
        elif file_type == 'pdf':
            contents = readpdf(uploaded_img.name)
            new_contents = contents[:50000].replace("$","\$")
        elif file_type in itypes:
            new_contents = label_text
        else:
            with open(uploaded_img.name, 'rb') as f:
                contents = f.read()
            new_contents = contents[:50000].decode('utf-8')
        new_contents = re.sub('[^A-Za-z0-9]+', ' ', new_contents)    
    
    if st.session_state.csv_summary or st.session_state.img_summary:                
        result = GetAnswers(new_contents,input_text)
        result = result.replace("$","\$")
        st.write(result)
    else:
        st.write("I am sorry it appears you have not uploaded any files for analysis. Can you please upload a file and then try again?")                





