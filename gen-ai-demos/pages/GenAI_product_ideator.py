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
# SDXL imports
import sagemaker
from sagemaker import ModelPackage, get_execution_role
from stability_sdk_sagemaker.predictor import StabilityPredictor
from stability_sdk_sagemaker.models import get_model_package_arn
from stability_sdk.api import GenerationRequest, GenerationResponse, TextPrompt
from PIL import Image
from typing import Union
import io
import base64
# End SDXL imports

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
iam_role = os.environ['IAM_ROLE']

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
sagemaker_session = sagemaker.Session()

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Korean', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
deployed_model = StabilityPredictor(endpoint_name=im_endpoint_name, sagemaker_session=sagemaker_session)


st.set_page_config(page_title="GenAI Product Ideator", page_icon="high_brightness")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Take your product idea to the next level")
st.sidebar.header("GenAI product ideator")
st.sidebar.markdown("### Make your pick")
industry = ''
industry = st.sidebar.selectbox(
    'Select an industry',
    ('Retail', 'Fashion', 'Manufacturing', 'Technology', 'Transport'))
model = 'Anthropic Claude'


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


#def query_im_endpoint(text):
#    client = boto3.client('runtime.sagemaker')
#    payload = {
#    "prompt": text,
#    "width": 480,
#    "height": 480,
#    "num_inference_steps": 200,
#    "seed": 42,
#    "guidance_scale": 8.5
#    }
#    body = json.dumps(payload).encode('utf-8')
#    response = client.invoke_endpoint(EndpointName=im_endpoint_name, ContentType='application/json', Body=body, Accept='application/json;jpeg')

#    return response

def parse_im_response(query_im_response):
    response_dict = json.loads(query_im_response['Body'].read())
    return response_dict['generated_images'], response_dict['prompt']


# Now using Jurassic Jumbo instruct
def call_jumbo(query,tokens, temp):
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

def save_image(img, prmpt):
    plt.figure(figsize=(12,12))
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.title(prmpt)
    prefix = "test-"+str(uuid.uuid4())+".jpg"
    plt.savefig("/tmp/"+prefix)
    #print("image name before S3 upload is: " + "/tmp/"+prefix)
    s3.upload_file("/tmp/"+prefix, bucket, prefix)
    img_url = 'http://dzlvehx4kcg5h.cloudfront.net/'+prefix
    return img_url

def GetAnswers(query):
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

    if "you" in query:
        query_type = "BEING"

    if query == "cancel":
        answer = 'It was swell chatting with you. Goodbye for now'
        return answer
    elif sentiment == 'NEGATIVE':
        answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'
        return answer
    elif query_type == "BEING":
        answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'
        return answer
    else:
        # Call the Stability model to get the image for our query, save it in S3 and build a response card
        #response = query_im_endpoint("Detailed image of " + query+" in " + industry.lower())
        #timg, prmpt = parse_im_response(response)
        #generated_image_decoded = BytesIO(base64.b64decode(timg[0].encode()))
        #generated_image_rgb = Image.open(generated_image_decoded).convert("RGB")
        #img_url_new = save_image(generated_image_rgb, prmpt)
        st.write("**Example image for your product idea**: \n")
        sd_query = "Detailed image of " + query+" in " + industry.lower() 
        st.image(sdxl_decode_and_show(call_sdxl(sd_query)))
        #st.image(img_url_new)
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic('Create a product description in '+language+' in 200 words for '+ query.strip("query:"))
            if generated_text != '':
                generated_text = generated_text.replace("$","\$")
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again'   
        return answer                       

st.write("**Instructions:** \n - Type a product idea prompt \n - You will see an image, a product description, and press release generated for your product idea")


input_text = st.text_input('**What is your product idea?**', key='text')
default_lang_ix = languages.index('English')
language = st.selectbox(
    '**Select an output language.** Only Alpha and Beta quadrant languages supported. For new requests, please contact C-3PO',
    options=languages, index=default_lang_ix)
key_phrases = ''
if input_text != '':
    result = GetAnswers(input_text)
    result = result.replace("$","\$")
    tab1, tab2, tab3, tab4 = st.tabs(["Product description", "Internal memo", "Press release", "Social Media Ad"])
    #c1, c2 = st.columns(2)
    with tab1:
        st.write("**Description for your product idea**")
        st.write(result)
    with tab2:
        st.write("**Internal memo for your product idea**")
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic('Generate an internal memo announcing the launch decision in '+language+' for '+ input_text.strip("query:"))
            if generated_text != '':
                generated_text = generated_text.replace("$","\$")
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again' 
            st.write(answer)
    with tab3:
        st.write("**Press release for your product idea**")
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic('Generate a press release in '+language+' for '+ input_text.strip("query:"))
            if generated_text != '':
                generated_text = generated_text.replace("$","\$")
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again' 
            st.write(answer)
    with tab4:
        st.write("**Social Media Ad for your product idea**")
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic('Generate a social media ad in '+language+' for '+ input_text.strip("query:"))
            if generated_text != '':
                generated_text = generated_text.replace("$","\$")
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again' 
            st.write(answer)
            st.balloons()
    