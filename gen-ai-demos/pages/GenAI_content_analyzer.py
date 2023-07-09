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
s3_prefix = 'general/content'

if 'img_summary' not in st.session_state:
    st.session_state['img_summary'] = None
if 'csv_summary' not in st.session_state:
    st.session_state['csv_summary'] = None
if 'new_contents' not in st.session_state:
    st.session_state['new_contents'] = None
if 'label_text' not in st.session_state:
    st.session_state['label_text'] = None

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
st.set_page_config(page_title="GenAI Content Analyzer", page_icon="sparkles")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Analyze any content and derive insights")
st.sidebar.header("GenAI Content Analyzer")
values = [1, 2, 3, 4, 5]
default_ix = values.index(3)
ftypes = ['csv', 'pptx', 'rtf','xls','xlsx','txt', 'pdf', 'doc', 'docx', 'json','ipynb','py','java']
atypes = ['csv', 'pptx', 'rtf','xls','xlsx','txt', 'pdf', 'doc', 'docx', 'json','ipynb','py','java', 'png', 'jpg']
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Korean', 'Irish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
p_count = st.sidebar.selectbox('Select the count of auto-prompts to consider', values, index=default_ix)

hallucinegator = "With reference to science, physics, math, and programming languages as we know it, what is the hallucination or false or illogical claim in this generated content: "

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

def readpdf(filename):
    # creating a pdf reader object
    reader = PdfReader(filename)
    # getting a specific page from the pdf file
    raw_text = []
    for page in reader.pages:
        raw_text.append(page.extract_text())
    return '\n'.join(raw_text)

def GetAnswers(original_text, query):
    #pii_list = []
    #sentiment = comprehend.detect_sentiment(Text=query, LanguageCode='en')['Sentiment']
    #resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode='en')
    #for pii in resp_pii['Entities']:
    #    if pii['Type'] not in ['NAME', 'AGE','ADDRESS','DATE_TIME']:
    #        pii_list.append(pii['Type'])
    #if len(pii_list) > 0:
    #    answer = "I am sorry but I found PII entities " + str(pii_list) + " in your query. Please remove PII entities and try again."
    #    return answer
    #query_type = ''
    #if "you" in query:
    #    query_type = "BEING"

    if query == "cancel":
        answer = 'It was swell chatting with you. Goodbye for now'
    
    #elif sentiment == 'NEGATIVE':
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'

    #elif query_type == "BEING":
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'
            
    else:
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic(original_text+'. Answer from this text with no hallucinations, false claims or illogical statements: '+ query.strip("query:"))
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
        if text['Confidence'] > 90:
            label_text += text['DetectedText'] + ' '

    for label in response['Labels']:
        if label['Confidence'] > 90:
            label_text += label['Name'] + ' '

    st.session_state.label_text = label_text

    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic('Explain the contents of this image in 300 words from these labels in ' +language+ ': '+ label_text)
        if generated_text != '':
            generated_text.replace("$","USD")
            summary = str(generated_text)+' '
        else:
            summary = 'Claude did not find an answer to your question, please try again'    
        return summary    

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
        new_contents = contents[:50000].decode('utf-8')

    #lang = comprehend.detect_dominant_language(Text=new_contents[:1000])
    #lang_code = str(lang['Languages'][0]['LanguageCode']).split('-')[0]
    #if lang_code in ['en']:
    #    resp_pii = comprehend.detect_pii_entities(Text=new_contents, LanguageCode=lang_code)
    #    immut_summary = new_contents
    #    for pii in resp_pii['Entities']:
    #        if pii['Type'] not in ['NAME', 'AGE','ADDRESS','DATE_TIME']:
    #            pii_value = immut_summary[pii['BeginOffset']:pii['EndOffset']]
    #            new_contents = new_contents.replace(pii_value, str('PII - '+pii['Type']))


    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic('Create a 300 words summary of this document in ' +language+ ': '+ new_contents)
        if generated_text != '':
            summary = str(generated_text)+' '
            summary = summary.replace("$","\$")
        else:
            summary = 'Claude did not find an answer to your question, please try again'    
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
file_type = ''
new_contents = ''
if uploaded_img is not None:
    if 'jpg' in uploaded_img.name or 'png' in uploaded_img.name or 'jpeg' in uploaded_img.name:
        #st.session_state.img_summary = None
        file_type = 'image'        
        c1.success(uploaded_img.name + ' is ready for upload')
        if c1.button('Upload'):
            with st.spinner('Uploading image file and starting summarization with Amazon Rekognition label detection...'):
                img_summary = upload_image_detect_labels(uploaded_img.getvalue())
                img_summary = img_summary.replace("$","\$")
                if len(img_summary) > 5:
                    st.session_state['img_summary'] = img_summary
                st.success('File uploaded and summary generated')
    elif str(uploaded_img.name).split('.')[1] in ftypes:
    #elif 'csv' in uploaded_img.name or 'txt' in uploaded_img.name:
        #st.session_state.csv_summary = None
        file_type = str(uploaded_img.name).split('.')[1]            
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
                st.session_state.new_contents = new_contents
                st.success('File uploaded and summary generated')
                st.balloons()
    else:
        st.write('Incorrect file type provided. Please check and try again')

h_results = ''
p1 = ''
m_summary = ''

if uploaded_img is not None:
    if st.session_state.img_summary:
        if len(st.session_state.img_summary) > 5:
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
            st.markdown('### Hallucination Analysis')
            if st.button("Halluci-Negator"):
                tab1, tab2 = st.tabs(["Hallucination Analysis", "Rewritten Summary"])
                with tab1:
                    h_results = call_anthropic(hallucinegator+" "+st.session_state.img_summary+" based on the original data provided in "+st.session_state.label_text)
                    h_results = h_results.replace("$", "\$")
                    st.write(h_results)
                with tab2:
                    m_summary = call_anthropic("Rewrite a 300 words summary in "+language+" from "+st.session_state.img_summary+" without hallucinations, false claims or illogical statements sticking only to available factual data")
                    st.write(m_summary)
    elif st.session_state.csv_summary:
        if len(st.session_state.csv_summary) > 5:
            st.markdown('**Summary**: \n')
            st.write(str(st.session_state.csv_summary).replace("$","\$"))
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
            st.markdown('### Hallucination Analysis')
            if st.button("Halluci-Negator"):
                tab1, tab2 = st.tabs(["Hallucination Analysis", "Rewritten Summary"])
                with tab1:
                    h_results = call_anthropic(hallucinegator+" "+st.session_state.csv_summary+" based on original data provided in "+st.session_state.new_contents)
                    h_results = h_results.replace("$", "\$")
                    st.write(h_results)
                with tab2:
                    m_summary = call_anthropic("Rewrite a 300 words summary in "+language+" from "+st.session_state.new_contents+" without hallucinations, false claims or illogical statements sticking only to available factual data")
                    st.write(m_summary)

    #p1 = 'Perform a well log interpretation in 500 words: '
    #new_summary = auto_summarize(p1, uploaded_img.name, file_type)
    #st.write("**Updated well log interpretation based on auto-prompts**")
    #st.write(new_summary)



input_text = st.text_input('**What insights would you like?**', key='text')
if input_text != '':
    if st.session_state.img_summary:
        result = GetAnswers(st.session_state.img_summary,input_text)
        result = result.replace("$","\$")
        st.write(result)
    elif st.session_state.csv_summary:

        s3.download_file(s3_bucket, s3_prefix+'/'+uploaded_img.name, uploaded_img.name)
        if file_type not in ['py','java','ipynb','pdf']:
            contents = textract.process(uploaded_img.name).decode('utf-8')
            new_contents = contents[:50000].replace('$','\$')
        elif file_type == 'pdf':
            contents = readpdf(uploaded_img.name)
            new_contents = contents[:50000].replace("$","\$")
        else:
            with open(uploaded_img.name, 'rb') as f:
                contents = f.read()
            new_contents = contents[:50000].decode('utf-8')


        result = GetAnswers(new_contents,input_text)
        result = result.replace("$","\$")
        st.write(result)
    else:
        st.write("I am sorry it appears you have not uploaded any files for analysis. Can you please upload a file and then try again?")                





