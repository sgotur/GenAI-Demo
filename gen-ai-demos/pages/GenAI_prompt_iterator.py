import boto3
import json
import streamlit as st
import anthropic
import os

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

if 'prompt' not in st.session_state:
    st.session_state['prompt'] = None
if 'Bedrock Titan Text' not in st.session_state:
    st.session_state['Bedrock Titan Text'] = None
if 'Bedrock Claude' not in st.session_state:
    st.session_state['Bedrock Claude'] = None
if 'SageMaker Falcon' not in st.session_state:
    st.session_state['SageMaker Falcon'] = None
if 'SageMaker Flan-T5 XXL' not in st.session_state:
    st.session_state['SageMaker Flan-T5 XXL'] = None
if 'Bedrock Jurassic-2' not in st.session_state:
    st.session_state['Bedrock Jurassic-2'] = None

ant_api_key = os.environ['ant_api_key']
ant_name = os.environ['ant_name']
bucket = os.environ['bucket']

falcon_endpoint_name = os.environ['falcon_endpoint_name']
flan_endpoint_name = os.environ['flan_endpoint_name']

#languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
st.set_page_config(page_title="GenAI Prompt Iterator", page_icon="chart_with_upwards_trend")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)


st.markdown("# Be prompt with your Prompts!!")
st.sidebar.header("GenAI Prompt Iterator")

bedrock = boto3.client('bedrock',region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)
sagemaker = boto3.client('runtime.sagemaker',region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)

newline, bold, unbold = '\n', '\033[1m', '\033[0m'

def call_bedrock_titan(prompt_text, max_token_count=512, temperature=1, top_p=1, stop_sequences=[]):
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

def call_falcon(prompt_text, max_new_tokens=512, top_k=1):
    payload = {"inputs": prompt_text, "parameters": {"do_sample": True, "top_p": 0.9, "temperature": 0.8, "max_new_tokens": max_new_tokens, "repetition_penalty": 1.03, "stop": ["\nUser:","<|endoftext|>","</s>"]}}    
    encoded_query = json.dumps(payload).encode('utf-8')
    query_response = sagemaker.invoke_endpoint(EndpointName=falcon_endpoint_name, ContentType='application/json', Body=encoded_query)
    model_predictions = json.loads(query_response['Body'].read())
    return model_predictions[0]["generated_text"][len(prompt_text):]

def call_flan(prompt_text, max_length=512, top_k=1):
    payload = {"text_inputs": prompt_text, "max_length": max_length, "top_k" : top_k}
    encoded_query = json.dumps(payload).encode('utf-8')
    query_response = sagemaker.invoke_endpoint(EndpointName=flan_endpoint_name, ContentType='application/json', Body=encoded_query)
    model_predictions = json.loads(query_response['Body'].read())
    return model_predictions['generated_texts'][0]

def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=1024,
    )
    return resp['completion']

models = {
    "Bedrock Titan Text" : call_bedrock_titan,
    "Bedrock Claude" : call_bedrock_claude,
    "Bedrock Jurassic-2" : call_bedrock_jurassic,
    "SageMaker Falcon" : call_falcon,
    "SageMaker Flan-T5 XXL": call_flan
}

fm_list = ['Bedrock Titan Text', 'Bedrock Claude', 'Bedrock Jurassic-2', 'SageMaker Falcon', 'SageMaker Flan-T5 XXL']

st.write("**Instructions:** \n - Enter your prompt and review the answers from the different FM candidates \n - Review evaluation of prompts and answer \n")

st.sidebar.markdown("""
    ### Example prompts to try out \n\n
    Write a poem about a clear blue sky in the summertime in the voice of Shakespeare \n
    Write the intro page of a hit new film about heroic Data Scientists saving the world from bureaucracy \n
    Write the introductory paragraph of a research paper on how Generative AI will change the world \n
    Write a letter to my boss asking for a raise with specific examples \n
    What are your thoughts on the state of the world? \n
    If a tree falls in the forest, does it make a sound? 
""")

input_text = st.text_input('**Prompt**', key='text_ag')
default_lang_ix = languages.index('English')
language = st.selectbox(
    '**Select an output language.**',
    options=languages, index=default_lang_ix)

if input_text != '':
    tabs = st.tabs(fm_list)
    for i in range(0, len(fm_list)):
        with tabs[i]:
            func = models[str(fm_list[i])]
            answer = func("Generate an answer in " + language + " for this prompt: " + input_text)
            st.session_state[fm_list[i]] = "Answer of "+fm_list[i]+" is: " + str(answer)
            st.write(answer)
    if st.session_state['SageMaker Flan-T5 XXL'] is not None:
        c1, c2 = st.columns(2)
        c1.write("**FM evaluation results**")
        c2.write("**Prompt corrections to consider**")
        eval_results = call_anthropic("For the prompts "+input_text+" compare, contrast and evaluate the completeness, correctness and relevance of these answers from the foundation models in " + language + ": " + st.session_state['Bedrock Titan Text'] + st.session_state['Bedrock Claude'] + st.session_state['Bedrock Jurassic-2'] + st.session_state['SageMaker Falcon'] + st.session_state['SageMaker Flan-T5 XXL'])
        c1.write(eval_results)
        new_prompts = call_anthropic("Suggest new prompts for each foundation model that most closely matches and can result in the most complete, correct and relevant answers documented in " + st.session_state['Bedrock Titan Text'] + st.session_state['Bedrock Claude'] + st.session_state['Bedrock Jurassic-2'] + st.session_state['SageMaker Falcon'] + st.session_state['SageMaker Flan-T5 XXL'])
        c2.write(new_prompts)

    