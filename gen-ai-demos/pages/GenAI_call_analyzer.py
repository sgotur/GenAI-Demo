import json
import boto3
import streamlit as st
import datetime
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
import uuid
import os
import time
import ai21
import string
import anthropic
import urllib3

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

if 'full_transcript' not in st.session_state:
    st.session_state['full_transcript'] = None
if 'turn_df' not in st.session_state:
    st.session_state['turn_df'] = pd.DataFrame()
if 'call_df' not in st.session_state:
    st.session_state['call_df'] = pd.DataFrame()
if 'model_summary' not in st.session_state:
    st.session_state['model_summary'] = None


s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
transcribe = boto3.client("transcribe",region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

# Get environment variables
fsi_index_id = os.environ['fsi_index_id']
iam_role = os.environ['IAM_ROLE']
energy_index_id = os.environ['energy_index_id']
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']
model_summary = ''
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
icols = ['job', 'turn','content', 'participant_role', 'loudness_score']
turn_df = pd.DataFrame(columns=icols)
ocols = ['job', 'non-talk-instances', 'non-talk-time', 'interruption_count', 'interruption_tot_duration', 'total_conv_duration']
call_df = pd.DataFrame(columns=ocols)



st.set_page_config(page_title="GenAI Call Analyzer", page_icon="headphones")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Call Analytics and Insights")
st.sidebar.header("GenAI Call Analyzer")
st.sidebar.markdown('''
        ### Example prompts for the transcript \n\n
        How could the overall experience be improved? \n
        Did the agent resolve the customer's request? \n
        What could the agent have done better? \n
        What can the agent upsell or cross-sell to the customer? \n
    ''')
model = 'Anthropic Claude'
#model = st.sidebar.selectbox(
#    'Select a LLM',
#    ('J2 Jumbo Instruct', 'Anthropic Claude'))


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

def call_models(full_transcript, command, query):
    generated_text = ''
    if model.lower() == 'j2 jumbo instruct':    
        #for i in range(3):    
        generated_text = call_jumbo(full_transcript+'. '+command+query.lower(),tokens=100, temp=0.7)
        if generated_text != '':
            answer = str(generated_text)+'\n\n\n My source is: ' +  tx_endpoint_name
        else:
            answer = 'J2 did not find an answer to your question, please try again'
        
    elif model.lower() == 'bedrock':
        '''
        if results['snippet'] == ' ':       
            generated_text = call_bedrock(full_transcript+'. '+command+query.lower(),tokens=100, temp=0.7)
            print("Kendra result is empty: "+str(generated_text))
            if generated_text != '':
                answer = str(generated_text)+'\n\n\n My source is: ' +  br_endpoint_name
            else:
                answer = 'Bedrock did not find an answer to your question, please try again'
        '''
        answer = 'Bedrock is not available yet, please try Jurassic for now'
    elif model.lower() == 'anthropic claude':    
        generated_text = call_anthropic(full_transcript+'. '+command+query.lower())
        if generated_text != '':
            answer = str(generated_text)
            answer = answer.replace("$","\$")
        else:
            answer = 'Claude did not find an answer to your question, please try again'
    return answer
    
def GetAnswers(full_transcript,query):
    tokens=25
    temp=0.7
    pii_list = []
    sentiment = 'POSITIVE'

    if query and len(query) > 5:
        sentiment = comprehend.detect_sentiment(Text=query, LanguageCode='en')['Sentiment']
        resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode='en')
        for pii in resp_pii['Entities']:
            if pii['Type'] not in ['ADDRESS','DATE_TIME']:
                pii_list.append(pii['Type'])
        if len(pii_list) > 0:
            answer = "I am sorry but I found PII entities " + str(pii_list) + " in your query. Please remove PII entities and try again."
            return answer
    else:
        answer = "Insufficient search query length. Please expand the query and/or add more context."
    
    query_type = ''
    if "you" in query:
        query_type = "BEING"

    if query in ["cancel","clear"]:
        answer = 'Thanks come back soon...'
        return answer

    #elif sentiment == 'NEGATIVE':
    #    answer = 'I am an AI assistant created to answer questions that you can learn from. I do not answer questions that are negatively worded or that concern my being at this time. Kindly rephrase your question and try again.'
    #    return answer
    
    elif query_type == "BEING":
        answer = 'I can answer questions that you can learn from. I do not answer questions that are negatively worded or that concern my being at this time. Kindly rephrase your question and try again.'
        return answer

    else:
        model_response = call_models(full_transcript, 'Answer neutrally without bias from this text:',query)                
        return model_response



# Method to run Transcribe call analytics
def runCallAnalytics(job_name, job_uri, output_location):
    try:
        transcribe.start_call_analytics_job(
             CallAnalyticsJobName = job_name,
             Media = {
                'MediaFileUri': job_uri
             },
             DataAccessRoleArn = iam_role,
             OutputLocation = output_location,
             ChannelDefinitions = [
                {
                    'ChannelId': 1, 
                    'ParticipantRole': 'AGENT'
                },
                {
                    'ChannelId': 0, 
                    'ParticipantRole': 'CUSTOMER'
                }
             ]
         )
        time.sleep(2)
        st.success("Transcribe Call analytics job submitted")
        st.snow()        
    except Exception as e:
        print(e)


#upload audio file to S3 bucket
def upload_audio_start_transcript(bytes_data, bucket, s3_file):
    output_prefix = 'streamlit_transcripts'
    s3.upload_fileobj(bytes_data, bucket, s3_file)
    st.success('Audio uploaded')
    #paginator = s3.get_paginator('list_objects_v2')
    #pages = paginator.paginate(Bucket=bucket, Prefix='streamlit_audio')
    job_name_list = []
    output_location = f"s3://{bucket}/{output_prefix}/"
    #for page in pages:
    #    for obj in page['Contents']:
    random = str(uuid.uuid4())
    audio_name = bytes_data.name
    job_name = audio_name + '-' + random
    job_name_list.append(job_name)
    job_uri = f"s3://{bucket}/{s3_file}"
    st.success('Submitting Amazon Transcribe call analytics for your audio: ' + job_name)
    # submit the transcription job now, we will provide our current bucket name as the output bucket
    runCallAnalytics(job_name, job_uri, output_location)
    return job_name_list
    
def upload_segments(job, i, transcript):
    # Get the turn by turn contents
    turn_idx = 0
    idx = len(turn_df)
    full_transcript = ""
    for turn in transcript['Transcript']:
        idx += 1
        turn_idx += 1
        # Build the base dataframe of call details, sentiment and loudness
        turn_df.at[idx,'job'] = job
        turn_df.at[idx, 'turn'] = turn_idx
        turn_df.at[idx,'content'] = str(turn['Content']).replace("'","").replace(",","")
        turn_df.at[idx, 'participant_role'] = turn['ParticipantRole']
        turn_df.at[idx, 'sentiment'] = turn['Sentiment']
        full_transcript += turn['Content']
        
        # Get an average loudness score for each turn
        tot_loud = 0
        for loud in turn['LoudnessScores']:
            if loud is not None:
                tot_loud += int(loud)
        avg_loudness = tot_loud/len(turn['LoudnessScores'])
        turn_df.at[idx, 'loudness_score'] = round(avg_loudness,0)
    
    # Finally get the overall call characteristics into a seperate dataframe
    call_df.at[i,'job'] = job
    call_df.at[i,'non-talk-instances'] = len(transcript['ConversationCharacteristics']['NonTalkTime']['Instances'])
    call_df.at[i,'non-talk-time'] = transcript['ConversationCharacteristics']['NonTalkTime']['TotalTimeMillis']
    call_df.at[i, 'interruption_count'] = transcript['ConversationCharacteristics']['Interruptions']['TotalCount']
    call_df.at[i, 'interruption_tot_duration'] = transcript['ConversationCharacteristics']['Interruptions']['TotalTimeMillis']
    call_df.at[i, 'total_conv_duration'] = transcript['ConversationCharacteristics']['TotalConversationDurationMillis']
    return full_transcript



st.write("**Instructions:** \n 1. Browse and select your contact center audio file. Download [this selection of sample files](https://amazon.awsapps.com/workdocs/index.html#/folder/5f9eb20255a6ed8512a0631392a8d40d3d4b433df64a8ae113b8d3cd9bb73e92) or use your own \n 2. Select an output language \n 3. Click Upload button \n 4. You will see call analytics and summary generated \n 5. Type your queries in the search bar to get conversation insights")

c1, c2 = st.columns(2)
c1.subheader("Upload your contact center audio file")
uploaded_audio = c1.file_uploader("**Select an audio file**")
default_lang_ix = languages.index('English')
c2.subheader("Select an output language")
language = c2.selectbox(
    'Only Alpha and Beta quadrant languages supported. For new requests, please contact C-3PO',
    options=languages, index=default_lang_ix)
full_transcript = ''
job_name_list = []
if uploaded_audio is not None:
    if 'wav' in uploaded_audio.name or 'mp4' in uploaded_audio.name or 'mp3' in uploaded_audio.name:        
        c1.success(uploaded_audio.name + ' ready for upload')
        if c1.button('Upload'):
            with st.spinner('Uploading audio file and starting Amazon Transcribe call analytics...'):
                job_name_list = upload_audio_start_transcript(uploaded_audio,bucket,'streamlit_audio/'+uploaded_audio.name)
    else:
        st.failure('Incorrect file type provided. Please select a speech wav file or a mp3 or a mp4 file to proceed')
    # Check job status
    
if len(job_name_list) > 0:
    st.session_state.full_transcript = None
    st.session_state.turn_df = pd.DataFrame()
    st.session_state.call_df = pd.DataFrame()
    st.session_state.model_summary = None
    with st.spinner('Awaiting call analytics job completion...'): 
        finish = False
        st.markdown("This should take a couple of minutes. Please check out these [great features of Amazon Transcribe](https://aws.amazon.com/transcribe/features/) in the meanwhile...")
        while finish == False:
            time.sleep(30)
            response = transcribe.get_call_analytics_job(CallAnalyticsJobName=job_name_list[0])
            job_status = response['CallAnalyticsJob']['CallAnalyticsJobStatus']
            st.write("Status of your call analytics job is: " + job_status)
            if job_status.upper() == 'COMPLETED':
                st.success("Call analytics job status: " + job_status)
                st.balloons()
                finish = True
                break

    # First we need an output directory
    dir = os.getcwd()+'/transcript_output'
    if not os.path.exists(dir):
        os.makedirs(dir)            

    # Get Call Analytics output
    i = -1
    for job in job_name_list:
        response = transcribe.get_call_analytics_job(CallAnalyticsJobName=job)
        json_file = response['CallAnalyticsJob']['Transcript']['TranscriptFileUri']
        a = json_file.split('/')
        tca_prefix = '/'.join(a[4:])
        s3.download_file(bucket,tca_prefix,dir+'/'+job)
        with open(dir+'/'+job) as f:
            data = json.load(f)
        i += 1
        full_transcript = upload_segments(str(job), i, data)   
    model_summary = call_models(full_transcript, 'Summarize this text to 100 words in '+language+':',query='')
    model_summary = model_summary.replace("$","\$")
    lang = comprehend.detect_dominant_language(Text=model_summary)
    lang_code = str(lang['Languages'][0]['LanguageCode']).split('-')[0]
    if lang_code in ['en']:
        resp_pii = comprehend.detect_pii_entities(Text=model_summary, LanguageCode=lang_code)
        immut_summary = model_summary
        for pii in resp_pii['Entities']:
            #if pii['Type'] not in ['ADDRESS','DATE_TIME']:
            pii_value = immut_summary[pii['BeginOffset']:pii['EndOffset']]
            model_summary = model_summary.replace(pii_value, str('PII - '+pii['Type']))



if st.session_state.full_transcript == None:
    st.session_state['full_transcript'] = full_transcript
if st.session_state.model_summary == None:
    st.session_state['model_summary'] = model_summary
if st.session_state.turn_df.empty:
    st.session_state.turn_df = pd.DataFrame(turn_df)
if st.session_state.call_df.empty:
    st.session_state.call_df = pd.DataFrame(call_df)

#display turn by turn
if not st.session_state.turn_df.empty:
    st.write("**Turn by turn sentiment** \n")
    st.bar_chart(st.session_state.turn_df[['sentiment']])
    #st.dataframe(st.session_state.turn_df, use_container_width=True)

#display call header
if not st.session_state.call_df.empty:
    st.write("**Call analytics** \n")
    st.table(st.session_state.call_df)

if st.session_state.model_summary:
    st.write('**Transcript Summary** \n') 
    st.write(st.session_state['model_summary'])
    if model.lower() == 'anthropic claude':  
        p_text = call_anthropic('Generate three prompts to query the summary: '+ st.session_state.model_summary)
        p_text1 = []
        p_text2 = ''
        if p_text != '':
            p_text = p_text.replace("$","\$")
            p_text1 = p_text.split('\n')
            for i,t in enumerate(p_text1):
                if i > 1:
                    p_text2 += t.split('?')[0]+'\n\n'
            p_summary = p_text2
        else:
            p_summary = '''
            1. How could the overall experience be improved? \n
            2. Did the agent resolve the customer's request? \n
            3. What could the agent have done better? \n
            '''  
    st.sidebar.markdown('### Suggested prompts for further insights \n\n' + 
            p_summary)

resp_pii = []
pii_list = []
pii_value = ''
if st.session_state.full_transcript:
    input_text = st.text_input('**What conversation insights would you like?**', key='text')
    if input_text != '':
        result = GetAnswers(st.session_state.full_transcript,input_text)
        result = result.replace("$","\$")
        resp_pii = comprehend.detect_pii_entities(Text=result, LanguageCode='en')
        immut_result = result
        for pii in resp_pii['Entities']:
            #if pii['Type'] not in ['ADDRESS','DATE_TIME']:
            pii_value = immut_result[pii['BeginOffset']:pii['EndOffset']]
            result = result.replace(pii_value, str('PII - '+pii['Type']))
        st.write(result)
