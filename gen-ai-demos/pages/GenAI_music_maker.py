import json
import boto3
import streamlit as st
import uuid
import os
import re
import time
import anthropic

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

if 'music_transcript' not in st.session_state:
    st.session_state['music_transcript'] = None
if 'additional_lyrics' not in st.session_state:
    st.session_state['additional_lyrics'] = None

s3 = boto3.client('s3',region_name=region,aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name=region,aws_access_key_id=key,aws_secret_access_key=secret)
transcribe = boto3.client("transcribe",region_name=region,aws_access_key_id=key,aws_secret_access_key=secret)
bedrock = boto3.client('bedrock',region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)

iam_role = os.environ['IAM_ROLE']
bucket = os.environ['bucket']
ant_name = os.environ['ant_name']
ant_api_key = os.environ['ant_api_key']
additional_lyrics = ''

songs = ['Lewis Carroll - Jabberwocky', 'Eaters - Surface Impact', 'Robert & The Misery Boys - Sleep it Off', 'Jack Skuller - S.U.R.E.', '']
#languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish','Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
emotions = ['Neutral', 'Happy', 'Hopeful', 'Fearful', 'Angry', 'Loving', 'Sad', 'Amusement', 'Tense', 'Dreamy', 'Serene', 'Defiant', 'Cheerful']

st.set_page_config(page_title="GenAI Music Maker", page_icon="musical_note")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Audio Inspiration ")

model = 'Amazon Titan Text'

def save_uploadedfile(uploadedfile):
    with open(os.path.join("/tmp/",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())

def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=1024,
    )
    return resp['completion']


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


def call_models(music_transcript, command, query):
    generated_text = call_bedrock_titan(music_transcript+'. '+command+query.lower(), max_token_count=100, temperature=0.7)
    answer = "Default Answer - will be replaced"
    if generated_text:
        answer = str(generated_text)+'\n\n\n My source is: Amazon Bedrock Titan Text'
    else:
        answer = 'Bedrock did not find an answer to your question, please try again'
    return answer


def transcription_job(job_name, job_uri, bucket, file_key):
    transcribe.start_transcription_job(
        TranscriptionJobName = job_name,
        IdentifyLanguage=True,
        Media = {
            'MediaFileUri': job_uri
        },
        JobExecutionSettings={
            'DataAccessRoleArn': iam_role
        },
        OutputBucketName=bucket,
        OutputKey=file_key,
    )
    print(f"Job Name: {job_name}, Job URI: {job_uri}, Bucket: {bucket}, file_key: {file_key}")
    time.sleep(2)


def upload_audio_start_transcript(bytes_data, bucket, s3_file):
    output_prefix = 'music_transcripts'
    s3.upload_fileobj(bytes_data, bucket, s3_file)
    job_name_list = []
    random = str(uuid.uuid4())
    audio_name = bytes_data.name
    audio_name = re.sub('[^A-Za-z0-9]+', '', audio_name)
    job_name = audio_name + '-' + random
    job_name_list.append(job_name)
    job_uri = f"s3://{bucket}/{s3_file}"
    transcription_job(job_name, job_uri, bucket, output_prefix)
    return job_name_list
    
def read_transcript(transcript):
    music_transcript = transcript['results']['transcripts'][0]['transcript']
    return music_transcript

st.write("**Instructions:** \n Load a song file and get the original lyrics as well as additional new lyrics in the language of your choice.")

c1, c2 = st.columns(2)
c1.subheader("Upload your audio file")
uploaded_audio = c1.file_uploader("**Select an audio file**")
if uploaded_audio is not None: 
    file_details = {"FileName":uploaded_audio.name,"FileType":uploaded_audio.type}
    save_uploadedfile(uploaded_audio)

default_lang_ix = languages.index('English')
default_mood_ix = emotions.index('Neutral')
c2.subheader("Select a language and mood")
language = c2.selectbox('Language:', options=languages, index=default_lang_ix)
mood = c2.selectbox('Mood:', options=emotions, index=default_mood_ix)
music_transcript = ''
job_name_list = []
if uploaded_audio is not None:
    if 'wav' in uploaded_audio.name or 'mp4' in uploaded_audio.name or 'mp3' in uploaded_audio.name or 'ogg' in uploaded_audio.name:        
        c1.success(uploaded_audio.name + ' ready for upload')
        if c1.button('Upload'):
            with st.spinner('Uploading audio file and starting Amazon Transcribe job.'):
                job_name_list = upload_audio_start_transcript(uploaded_audio,bucket,'music_maker/'+uploaded_audio.name)
                print(f"Job Name List: {job_name_list}")
    else:
        st.write('Incorrect file type provided. Please select a speech wav file or a mp3 or a mp4 file to proceed')
    
if len(job_name_list) > 0:
    st.session_state.music_transcript = None
    st.session_state.additional_lyrics = None
    with st.spinner('Transcribing audio file...'): 
        finish = False
        while finish == False:
            time.sleep(30)
            response = transcribe.get_transcription_job(TranscriptionJobName=job_name_list[0])
            job_status = response['TranscriptionJob']['TranscriptionJobStatus']
            st.write("Transcription in progress...")
            if job_status.upper() == 'COMPLETED':
                st.success("Transcription complete.")
                finish = True
                break

    dir = os.getcwd()+'/transcript_output'
    if not os.path.exists(dir):
        os.makedirs(dir)            

    i = -1
    for job in job_name_list:
        response = transcribe.get_transcription_job(TranscriptionJobName=job)
        json_file = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
        a = json_file.split('/')
        tca_prefix = '/'.join(a[4:])
        s3.download_file(bucket,tca_prefix,dir+'/'+job)
        with open(dir+'/'+job) as f:
            data = json.load(f)
        i += 1
        music_transcript = read_transcript(data)   
    additional_lyrics = call_models(music_transcript, 'Generate three additonal verses matching style of this song in '+language+' language and in '+mood+' mood :',query='')
    additional_lyrics = additional_lyrics.replace("$","\$")

if st.session_state.music_transcript == None:
    st.session_state['music_transcript'] = music_transcript
if st.session_state.additional_lyrics == None:
    st.session_state['additional_lyrics'] = additional_lyrics

if st.session_state.additional_lyrics:
    st.write('**Song Lyrics** \n') 
    st.write(st.session_state['music_transcript'])
    st.write('**Additional Lyrics** \n') 
    st.write(st.session_state['additional_lyrics'])

if uploaded_audio:
    audio_file = open(os.path.join("/tmp/",uploaded_audio.name), 'rb')
    audio_bytes = audio_file.read()
    audio_extension = os.path.splitext(uploaded_audio.name)
    audio_format= f"audio/{audio_extension[1]}"
    st.audio(audio_bytes, format=audio_format)

    p_text = call_anthropic('Describe these music lyrics three different ways and identify its genre '+ st.session_state.music_transcript)
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
        '''  
    st.sidebar.markdown('### Additional Audio Lyric Insights \n\n' + p_summary)
