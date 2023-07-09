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
import re
import sagemaker
from sagemaker import ModelPackage, get_execution_role
from stability_sdk_sagemaker.predictor import StabilityPredictor
from stability_sdk_sagemaker.models import get_model_package_arn
from stability_sdk.api import GenerationRequest, GenerationResponse, TextPrompt
from PIL import Image
from typing import Union
import io
import base64

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

if 'img_summary' not in st.session_state:
    st.session_state['img_summary'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
rekognition = boto3.client('rekognition',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']

#bucket = os.environ['bucket'] # should we be saving images to the bucket under a specific prefix rather than on the local file system?

# Setup a SageMaker session and initialize the Stability AI Predictor with the SDXL endpoint
session = boto3.session.Session(aws_access_key_id=key,aws_secret_access_key=secret,region_name=region)
sagemaker_session = sagemaker.Session()

deployed_model = StabilityPredictor(endpoint_name=im_endpoint_name, sagemaker_session=sagemaker_session)

st.set_page_config(page_title="GenAI Photo Booth", page_icon="camera_with_flash")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Generate picture perfect moments!!")
st.write("## Check back soon, work in progress...")
st.sidebar.header("GenAI Photo Booth")

def call_sdxl(query):
    output = deployed_model.predict(GenerationRequest(text_prompts=[TextPrompt(text=query)],
                                             style_preset="cinematic",
                                             seed = 3
                                             ))
    return output

def sdxl_decode_and_show(model_response: GenerationResponse) -> None:
    """
    Decodes and displays an image from SDXL output

    Args:
        model_response (GenerationResponse): The response object from the deployed SDXL model.

    Returns:
        None
    """
    image = model_response.artifacts[0].base64
    image_data = base64.b64decode(image.encode())
    image = Image.open(io.BytesIO(image_data))
    return image

def generate_image_var(image_prompt: str, init_image: str = None, cfg_scale: int = None, image_strength: float = None, style_preset: str = None, seed: int = None):
    generation_request = GenerationRequest(
        text_prompts=[TextPrompt(text=image_prompt)],
        init_image=init_image,
        cfg_scale=cfg_scale,
        image_strength=image_strength,
        style_preset=style_preset,
        seed=seed,
    )
    return deployed_model.predict(generation_request)

def campaign_images(prompt: str, init_image: str = None, style_preset: str = None, seed: int = None, img_strength=None):
    if not img_strength: 
        img_strength = [0.5, 0.3]
    if not init_image:
        first_image = generate_image_var(prompt, style_preset=style_preset, seed=seed)
        _ = decode_and_show(first_image)
        return [first_image.artifacts[0].base64, _]
    else:
        for img_strength in img_strength:
            output = generate_image_var(prompt, init_image=init_image, cfg_scale=10, image_strength=img_strength, style_preset=style_preset, seed=seed)
            decode_and_show(output)

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

def GetAnswers(summary, query):

    if query == "cancel":
        answer = 'It was swell chatting with you. Goodbye for now'
    
    #elif sentiment == 'NEGATIVE':
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'

            
    else:
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic(summary+'. Answer from this summary in '+language+': '+ query.strip("query:"))
            if generated_text != '':
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again'   
    return answer          


#upload audio file to S3 bucket
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

    if model.lower() == 'anthropic claude':  
        generated_text = call_anthropic('Summarize the image in '+language+' without any halluciations in 200 words strictly adhering to what is mentioned in the labels only: '+ label_text)
        if generated_text != '':
            generated_text.replace("$","USD")
            summary = str(generated_text)+' '
        else:
            summary = 'Claude did not find an answer to your question, please try again'    
        return summary    


st.write("**Instructions:** \n - Browse and select your image file. You can [download these samples](https://amazon.awsapps.com/workdocs/index.html#/folder/8a926bdf7852d8c940d4238017b48678d6fcfadaaa918d03a7965db6822041f6) or use your own \n - You will see a summary based on labels identifed by Amazon Rekognition \n - Type your query in the search bar to get image insights")

c1, c2 = st.columns(2)
c1.subheader("Upload your image file")
uploaded_img = c1.file_uploader("**Select an image file**", type=['jpg','jpeg','png'])
default_lang_ix = languages.index('English')
c2.subheader("Select an output language")
language = c2.selectbox(
    'Only Alpha and Beta quadrant languages supported at this time. For new requests, please contact C-3PO',
    options=languages, index=default_lang_ix)
img_summary = ''
p_summary = ''
if uploaded_img is not None:
    if 'jpg' in uploaded_img.name or 'png' in uploaded_img.name or 'jpeg' in uploaded_img.name:        
        c1.success(uploaded_img.name + ' is ready for upload')
        if c1.button('Upload'):
            with st.spinner('Uploading image file and starting Amazon Rekognition label detection...'):
                inapp_res = rekognition.detect_moderation_labels(Image={'Bytes': uploaded_img.getvalue()})
                if len(inapp_res['ModerationLabels']) == 0:
                    img_summary = upload_image_detect_labels(uploaded_img.getvalue())
                    #img_summary = re.escape(img_summary).replace("\","")
                    img_summary = img_summary.replace("$","\$")
                else:
                    st.write("Inappropriate content detected in image. Please change your image and try again")
    else:
        st.write('Incorrect file type provided. Please select either a JPG or PNG file to proceed')
    # Check job status
    

if len(img_summary) >= 5:
    st.session_state['img_summary'] = img_summary

if st.session_state.img_summary:
    if uploaded_img is not None:
        st.image(uploaded_img)
    st.markdown('**Image summary**: \n')

    st.write(str(st.session_state['img_summary']))
    if model.lower() == 'anthropic claude':  
        p_text = call_anthropic('Generate three prompts to query the summary: '+ st.session_state.img_summary)
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
        result = GetAnswers(st.session_state.img_summary,input_text)
        result = result.replace("$","\$")
        st.write(result)





