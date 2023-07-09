import boto3
import streamlit as st
import uuid
import anthropic
from streamlit_chat import message
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
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)

if 'sessionID' not in st.session_state:
    st.session_state['sessionID'] = str(uuid.uuid4())
if 'prevText' not in st.session_state:
    st.session_state['prevText'] = None
if 'count' not in st.session_state:
    st.session_state['count'] = 0

st.set_page_config(page_title="GenAI Cool Cucumber", page_icon="runner")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Cool Cucumber")
st.sidebar.header("GenAI Cool Cucumber")
st.sidebar.markdown("""
    ### Example User Stories \n\n
    As a end user, I need to be able to log in, access my account, and check my rewards balance. \n
    As a site administrator, I must be able to access the admin portal, see all active sites, and respond to questions. \n
    New customers should get prompted for a site tour and within three minutes of use, have a support chat popup. \n
    Every six months, users should revieve a notification that they are edlligible for an account review with their personal advisor. \n
    As a developer lead, I want to be able to understand my engineers progress, so I can better report our sucesses and failures. 
""")

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

def GetAnswers(query):
    pii_list = []
    lang = comprehend.detect_dominant_language(Text=query)
    lang_code = str(lang['Languages'][0]['LanguageCode']).split('-')[0]
    if lang_code in ['en']:
        resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode=lang_code)
        for pii in resp_pii['Entities']:
            if pii['Type'] not in ['NAME','AGE','ADDRESS','DATE_TIME']:
                pii_list.append(pii['Type'])
        if len(pii_list) > 0:
            answer = "I am sorry but I found Personally Identifiable Information (PII) " + str(pii_list) + " in your query. Please remove PII entities and try again."
            return answer

    if query == "cancel":
        answer = 'It was swell chatting with you. Goodbye for now'
                
    else:
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic(query.strip("query:"))
            if generated_text != '':
                answer = str(generated_text)
            else:
                answer = 'Claude did not find an answer to your question, please try again'   
    return answer          

st.write("**Instructions:** \n - Type your user story in the text box below. \n - The story will be rewritten in [Cubumber](https://cucumber.io/docs/guides/overview/) format to support [Behavior Driven Development](https://cucumber.io/docs/bdd/).")
input_text = st.text_input('**User story**', key='text')
p_summary = ''
if input_text != '':
    message(input_text, is_user=True, key=str(uuid.uuid4()))
    if st.session_state.prevText is not None:
        result = GetAnswers('Answer from this text if the question is related to this text and. Otherwise answer the question directly without referring to this text and rewrite the text using Cucumber language to support Behavior Driven Development: '+str(st.session_state.prevText)+' ' + input_text)
    else:
        result = GetAnswers('Rewrite the following text using Cucumber language to support Behavior Driven Development.' + input_text)
    result = result.replace("$","\$")
    lang = comprehend.detect_dominant_language(Text=result)
    lang_code = str(lang['Languages'][0]['LanguageCode']).split('-')[0]
    if lang_code in ['en']:
        resp_pii = comprehend.detect_pii_entities(Text=result, LanguageCode=lang_code)
        immut_summary = result
        for pii in resp_pii['Entities']:
            if pii['Type'] not in ['NAME','AGE','ADDRESS','DATE_TIME']:
                pii_value = immut_summary[pii['BeginOffset']:pii['EndOffset']]
                result = result.replace(pii_value, str('PII - '+pii['Type']))
    st.session_state.prevText = result
    message(result, key=str(uuid.uuid4()))
    st.session_state.count = int(st.session_state.count) + 1
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
    st.sidebar.markdown('### Suggested prompts for further refinement \n\n' + 
            p_summary)
