import json
import boto3
import re
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
from time import sleep
from boto3.session import Session
import datetime
from datetime import date
from datetime import datetime

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
iam_role = os.environ['IAM_ROLE']
falcon_endpoint = os.environ['falcon_endpoint_name']

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
bedrock = boto3.client('bedrock',region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)

# temp credentials needed to call Athena otherwise it errors out
sts = boto3.client("sts", region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
response = sts.get_session_token()
session_token = response['Credentials']['SessionToken']
access_key=response['Credentials']['AccessKeyId']
secret_access_key=response['Credentials']['SecretAccessKey']
athena_session = Session(region_name='us-east-1',aws_access_key_id=access_key, aws_secret_access_key=secret_access_key, aws_session_token=session_token)
athena = athena_session.client('athena')

# create variables for athena
database = "securitylakedatabase"
cloudtrail_management_table = "cloudtrailmgmteventtable"
S3_data_table = "s3dataeventtable"
lambda_data_table = "lambdadatatable"
route_53_table = "route53table"
vpc_flow_table = "vpcflowtable"
securityhub_table = "shfindingtable"
query_results_bucket = "s3://sl-gen-ai-query-results"

if 'securityhub_findings' not in st.session_state:
    st.session_state['securityhub_findings'] = None
if 'cloudtrail_findings' not in st.session_state:
    st.session_state['cloudtrail_findings'] = None
if 's3_findings' not in st.session_state:
    st.session_state['s3_findings'] = None
if 'lambda_findings' not in st.session_state:
    st.session_state['lambda_findings'] = None
if 'r53_findings' not in st.session_state:
    st.session_state['r53_findings'] = None
if 'securityhub_summary' not in st.session_state:
    st.session_state['securityhub_summary'] = None
if 'cloudtrail_summary' not in st.session_state:
    st.session_state['cloudtrail_summary'] = None
if 's3_summary' not in st.session_state:
    st.session_state['s3_summary'] = None
if 'lambda_summary' not in st.session_state:
    st.session_state['lambda_summary'] = None
if 'r53_summary' not in st.session_state:
    st.session_state['r53_summary'] = None


# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Star Trek - Klingon', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']

st.set_page_config(page_title="GenAI Security Advisor", page_icon="detective")

st.markdown(
    """
    ### :red[Note] 
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    """)

st.markdown("# Get prescriptive guidance from Amazon Security Lake")
st.sidebar.header("GenAI Security Advisor")

fms = ['Bedrock Claude', 'Bedrock Jurassic-2', 'SageMaker Falcon']
default_model = fms.index('Bedrock Claude')

model = ''
model = st.sidebar.selectbox(
    'Select a FM',
    options=fms, index=default_model)

def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=1024,
    )
    return resp['completion']

def call_bedrock_titan(prompt_text, max_token_count=1024, temperature=0.5, top_p=1, stop_sequences=[]):
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

def call_bedrock_claude(prompt_text, max_tokens_to_sample=1024, temperature=0.5, top_k=250, top_p=1):
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
    "bedrock jurassic-2" : call_bedrock_jurassic,
    "bedrock claude" : call_bedrock_claude,
    "sagemaker falcon" : call_falcon
}

finding_types = {
    "securityhub_findings" : "Suggest 3 prompts to further analyze findings in: " + str(st.session_state.securityhub_findings),
    "cloudtrail_findings" : "Suggest 3 prompts to further analyze findings in: " + str(st.session_state.cloudtrail_findings),
    "s3_findings" : "Suggest 3 prompts to further analyze findings in: " + str(st.session_state.s3_findings),
    "lambda_findings": "Suggest 3 prompts to further analyze findings in: " + str(st.session_state.lambda_findings),
    "r53_findings": "Suggest 3 prompts to further analyze findings in: " + str(st.session_state.r53_findings)
}

def sh_findings():
    # Query executes and then puts results in bucket
    execute_query = athena.start_query_execution(
        QueryString = f"SELECT DISTINCT eventday, finding, severity, state FROM {securityhub_table} WHERE eventDay BETWEEN '{start_date}' and '{end_date}' LIMIT 20;",
        QueryExecutionContext = {
            'Database': database
        },       
        ResultConfiguration={
            'OutputLocation': query_results_bucket
        }
    )
    
    sleep(4)
    # Grab results from bucket and print - maybe not needed
    sh_query_results = athena.get_query_results(
        QueryExecutionId = execute_query['QueryExecutionId']
    )
    return sh_query_results
    
# Athena query to grab the last 7 days of data related to CloudTrail management events in Security Lake
def cloudtrail_events():
    response = athena.start_query_execution(
        QueryString = f"SELECT DISTINCT eventday, api, type_name, severity, status FROM {cloudtrail_management_table} WHERE eventDay BETWEEN '{start_date}' and '{end_date}' LIMIT 20;",
        QueryExecutionContext = {
            'Database': database
        },       
        ResultConfiguration={
            'OutputLocation': query_results_bucket            
        }
    )
    
    sleep(4)
    # Grab results from bucket and print - maybe not needed
    ct_query_results = athena.get_query_results(
        QueryExecutionId = response['QueryExecutionId']
    )
    return ct_query_results
    
# Athena query to grab the last 7 days of data related to S3 data events in Security Lake
def s3_events():
    response = athena.start_query_execution(
        QueryString = f"SELECT DISTINCT eventday, api, resources, type_name, severity, status FROM {S3_data_table} WHERE eventDay BETWEEN '{start_date}' and '{end_date}' LIMIT 20;",
        QueryExecutionContext = {
            'Database': database
        },       
        ResultConfiguration={
            'OutputLocation': query_results_bucket            
        }
    )
    
    sleep(4)
    # Grab results from bucket and print - maybe not needed
    s3_query_results = athena.get_query_results(
        QueryExecutionId = response['QueryExecutionId']
    )
    return s3_query_results
    

# Athena query to grab the last 7 days of data related to Lambda data events in Security Lake
def lambda_events():
    response = athena.start_query_execution(
        QueryString = f"SELECT DISTINCT eventday, resources, type_name, severity, status FROM {lambda_data_table} WHERE eventDay BETWEEN '{start_date}' and '{end_date}' LIMIT 20;",
        QueryExecutionContext = {
            'Database': database
        },       
        ResultConfiguration={
            'OutputLocation': query_results_bucket            
        }
    )
    sleep(4)
    # Grab results from bucket and print - maybe not needed
    lm_query_results = athena.get_query_results(
        QueryExecutionId = response['QueryExecutionId']
    )
    return lm_query_results


# Athena query to grab the last 7 days of data related to Route 53 data events in Security Lake
def route_53_events():
    response = athena.start_query_execution(
        QueryString = f"SELECT DISTINCT eventday, query, answers, severity, type_name FROM {route_53_table} WHERE eventDay BETWEEN '{start_date}' and '{end_date}' LIMIT 20;",
        QueryExecutionContext = {
            'Database': database
        },       
        ResultConfiguration={
            'OutputLocation': query_results_bucket            
        }
    )
    sleep(4)
    # Grab results from bucket and print - maybe not needed
    rt_query_results = athena.get_query_results(
        QueryExecutionId = response['QueryExecutionId']
    )
    return rt_query_results          

def generate_autoprompts(finding_type):
    prompt_query = finding_types[finding_type]
    #prompt_query = "Generate 3 prompts to query the security findings in: " + findings
    func = models[model.lower()]
    p_text = func(prompt_query)
    p_text1 = []
    p_text2 = ''
    if p_text != '':
        p_text.replace("$","\$")
        p_text1 = p_text.split('\n')
        for i,t in enumerate(p_text1):
            if i > 1:
                p_text2 += t.split('\n')[0]+'\n\n'
        p_summary = p_text2
    st.markdown('### Suggested prompts for further analysis \n\n' + 
            p_summary)


st.write("**Instructions:** \n - Select a date range, output language and review overall security advisory from the demo security lake \n - For details, select a security event type tab \n - You will see a snapshot of security events for your date range and a summary \n - Type your query to get more insights for each tab")

today = date.today()
first_day = today.replace(day=1)
last_month = first_day.replace(today.year, today.month-1)



c1, c2 = st.columns(2)

s_date = c1.date_input(
    "Start date for security events",
    min_value = last_month,
    max_value = today)
start_date = str(s_date.strftime("%Y%m%d"))

e_date = c2.date_input(
    "End date for security events",
    min_value = last_month,
    max_value = today
    )
end_date = str(e_date.strftime("%Y%m%d"))

default_lang_ix = languages.index('English')
language = st.selectbox(
    '**Select an output language**',
    options=languages, index=default_lang_ix)

result = ''



if st.button('Start analysis'):
    with st.spinner('Querying the demo security lake...'):
        if st.session_state.securityhub_findings is None:
                result = str(sh_findings())
                st.session_state.securityhub_findings = "Security Hub findings: " + result
                func = models[model.lower()]
                answer = func("Summarize in 200 words in "+language+" the security events or findings mentioned in this JSON document " + st.session_state.securityhub_findings)
                st.session_state.securityhub_summary = answer
        if st.session_state.cloudtrail_findings is None:
                result = str(cloudtrail_events())
                st.session_state.cloudtrail_findings = "Cloudtrail findings: " + result
                func = models[model.lower()]
                answer = func("Summarize in 200 words in "+language+" the security events or findings mentioned in this JSON document " + st.session_state.cloudtrail_findings)
                st.session_state.cloudtrail_summary = answer
        if st.session_state.s3_findings is None:
                result = str(s3_events())
                st.session_state.s3_findings = "S3 findings: " + result
                func = models[model.lower()]
                answer = func("Summarize in 200 words in "+language+" the security events or findings mentioned in this JSON document " + st.session_state.s3_findings)
                st.session_state.s3_summary = answer
        if st.session_state.lambda_findings is None:
                result = str(lambda_events())
                st.session_state.lambda_findings = "AWS Lambda findings: " + result
                func = models[model.lower()]
                answer = func("Summarize in 200 words in "+language+" the security events or findings mentioned in this JSON document " + st.session_state.lambda_findings)
                st.session_state.lambda_summary = answer    
        if st.session_state.r53_findings is None:
                result = str(route_53_events())
                st.session_state.r53_findings = "Route53 findings: " + result
                func = models[model.lower()]
                answer = func("Summarize in 200 words in "+language+" the security events mentioned in this JSON document " + st.session_state.r53_findings)
                st.session_state.r53_summary = answer
        st.success("**Security lake findings obtained and summaries generated")


    #with st.spinner('Generating presciptive guidance...'):
    #    func = models[model.lower()]
        #st.write("**Amazon Security Lake Advisory**")
        #answer = func("Provide a summary of the security findings in 200 words along with mitigation options using AWS well architected security framework in "+language+" from these JSON documents " + st.session_state.securityhub_findings + st.session_state.cloudtrail_findings + st.session_state.s3_findings + st.session_state.lambda_findings + st.session_state.r53_findings)
        #st.write(answer)
        #st.write("**Amazon Security Lake Advisory - Guidance**")
        #answer = func("Provide step by step prescriptive guidance using AWS Well Architected security framework in "+language+" to mitigate the security events, alerts or findings mentioned in these JSON documents " + st.session_state.securityhub_findings + st.session_state.cloudtrail_findings + st.session_state.s3_findings + st.session_state.lambda_findings + st.session_state.r53_findings)
        #st.write(answer)


    
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Security Hub", "Cloud Trail", "Amazon S3", "AWS Lambda", "Route53"])
#c1, c2 = st.columns(2)
if st.session_state.securityhub_findings:
    with tab1:
        st.write("**Security Hub Events Summary**")
        st.write(st.session_state.securityhub_summary)
        generate_autoprompts("securityhub_findings")
        input_text = st.text_input('**What insights would you like?**', key='text1')
        if input_text != '':
            func = models[model.lower()]
            result = func("Provide a specific answer to the question " + input_text+ " in " + language + " from this JSON document " + st.session_state.securityhub_findings)
            result = result.replace("$","\$")
            st.write(result)
if st.session_state.cloudtrail_findings:
    with tab2:
        st.write("**Cloud Trail Events Summary**")
        st.write(st.session_state.cloudtrail_summary)
        generate_autoprompts("cloudtrail_findings")
        input_text = st.text_input('**What insights would you like?**', key='text2')
        if input_text != '':
            func = models[model.lower()]
            result = func("Provide a specific answer to the question " +input_text+ " in " + language + " from this JSON document " + st.session_state.cloudtrail_findings)
            result = result.replace("$","\$")
            st.write(result)
if st.session_state.s3_findings:
    with tab3:
        st.write("**S3 Events Summary**")
        st.write(st.session_state.s3_summary)
        generate_autoprompts("s3_findings")
        input_text = st.text_input('**What insights would you like?**', key='text3')
        if input_text != '':
            func = models[model.lower()]
            result = func("Provide a specific answer to the question " +input_text+ " in " + language + " from this JSON document " +  st.session_state.s3_findings)
            result = result.replace("$","\$")
            st.write(result)
if st.session_state.lambda_findings:
    with tab4:
        st.write("**AWS Lambda Events Summary**")
        st.write(st.session_state.lambda_summary)
        generate_autoprompts("lambda_findings")
        input_text = st.text_input('**What insights would you like?**', key='text4')
        if input_text != '':
            func = models[model.lower()]
            result = func("Provide a specific answer to the question " +input_text+ " in " + language + " from this JSON document " + st.session_state.lambda_findings)
            result = result.replace("$","\$")
            st.write(result)
if st.session_state.r53_findings:
    with tab5:
        st.write("**Route53 Events Summary**")
        st.write(st.session_state.r53_summary)
        generate_autoprompts("r53_findings")
        input_text = st.text_input('**What insights would you like?**', key='text5')
        if input_text != '':
            func = models[model.lower()]
            result = func("Provide a specific answer to the question " +input_text+ " in " + language + " from this JSON document " +  st.session_state.r53_findings)
            result = result.replace("$","\$")
            st.write(result)
