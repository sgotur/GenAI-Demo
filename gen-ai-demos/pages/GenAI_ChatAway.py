import json
import boto3
import streamlit as st
import uuid
import anthropic
import logging
from streamlit_chat import message
from boto3.dynamodb.conditions import Key
import os

# create a unique widget
if 'key' not in st.session_state:
    st.session_state.key = str(uuid.uuid4())

# Get environment variables
ant_api_key = os.environ['ant_api_key']
ant_name = os.environ['ant_name']

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
falcon_endpoint = os.environ['falcon_endpoint_name']
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
ddb = boto3.resource('dynamodb',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
dynamodb = boto3.client('dynamodb',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
table = ddb.Table('genai-chat-history')
bedrock = boto3.client('bedrock',region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)

if 'sessionID' not in st.session_state:
    st.session_state['sessionID'] = str(uuid.uuid4())
if 'prevText' not in st.session_state:
    st.session_state['prevText'] = None
if 'count' not in st.session_state:
    st.session_state['count'] = 0

st.set_page_config(page_title="GenAI ChatAway", page_icon="flashlight")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.sidebar.header("GenAI ChatAway")

fms = ['Anthropic Claude', 'SageMaker Falcon', 'Bedrock Titan']
default_model = fms.index('Anthropic Claude')

model = ''
model = st.sidebar.selectbox(
    'Select a FM',
    options=fms, index=default_model)
st.markdown("# Chat with "+model)


def get_old_chats():
    res = table.query(
    KeyConditionExpression=Key('session_id').eq(str(st.session_state.sessionID))
    )
    return res['Items']


def store_chat(turn, question, answer):
    dynamodb.put_item(
        TableName="genai-chat-history",
        Item={
            'session_id': {'S': str(st.session_state.sessionID)},
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
            "max_new_tokens": 512,
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
    "bedrock titan" : call_bedrock_titan,
    "bedrock claude" : call_bedrock_claude,
    "bedrock jurassic-2" : call_bedrock_jurassic,
    "sagemaker falcon" : call_falcon,
    "anthropic claude": call_anthropic
}


def GetAnswers(query):
    if query == "cancel":
        answer = 'It was swell chatting with you. Goodbye for now'
    
    #elif sentiment == 'NEGATIVE':
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'

    #elif query_type == "BEING":
    #    answer = 'Kindly rephrase your question keeping it impersonal and try again.'
            
    else:
        func = models[model.lower()]
        answer = func(query)   
    return answer          


st.write("**Instructions:** \n - Type your query in the search bar \n - Only last five chats displayed for brevity \n")
input_text = st.text_input('**Chat with me**', key='text')

p_summary = ''
if input_text != '':
    message(input_text, is_user=True, key=str(uuid.uuid4()))
    if st.session_state.prevText is not None:
        result = GetAnswers(f'Answer from this text if the question is related to this text. Otherwise answer the question directly without referring to this text: {str(st.session_state.prevText)} {input_text}')
    else:
        result = GetAnswers(input_text)
    result = result.replace("$","\$")
    st.session_state.prevText = result
    message(result, key=str(uuid.uuid4()))
    if int(st.session_state.count) <= 5:
        old_chats = get_old_chats()
        if old_chats:
            for chat in old_chats:
                message(chat['question'], is_user=True, key=str(uuid.uuid4()))
                message(chat['answer'], key=str(uuid.uuid4()))
    st.session_state.count = int(st.session_state.count) + 1
    store_chat(st.session_state.count, input_text, result)
    p_text = call_anthropic('Generate three prompts to query the text: '+ result)
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
