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
from streamlit_agraph import agraph, Node, Edge, Config


# Get environment variables
ant_api_key = os.environ['ant_api_key']
ant_name = os.environ['ant_name']

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
lambda_client = boto3.client('lambda', region_name='us-east-1', aws_access_key_id=key, aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)


lambda_function = 'genai_call_neptune'
function_params = {}

def execute_genai_call_neptune(mode, start_graph_query, end_graph_query):
    if start_graph_query:
        function_params['start_graph_query'] = start_graph_query.strip()
    if end_graph_query:
        function_params['end_graph_query'] = end_graph_query.strip()
    function_params['mode'] = mode
    response = lambda_client.invoke(
                FunctionName=lambda_function,
                Payload=json.dumps(function_params))

    return response

if 'graph_summary' not in st.session_state:
    st.session_state['graph_summary'] = None


st.set_page_config(page_title="GenAI Graph Guru", page_icon="cyclone")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.markdown("# Change your Graph game")
st.sidebar.header("GenAI Graph Guru")

st.write("**Instructions:** \n - Airplane routes graph is pre-loaded for you in [Amazon Neptune](https://aws.amazon.com/neptune/) \n - Play with the graph trying different airport codes and review the summary \n - Traverses 15 nodes for airport search and 5 paths for route search \n - When ready, type your questions to get additional insights")

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
    sentiment = comprehend.detect_sentiment(Text=query, LanguageCode='en')['Sentiment']
    lang = comprehend.detect_dominant_language(Text=query)
    lang_code = str(lang['Languages'][0]['LanguageCode']).split('-')[0]
    if lang_code in ['en']:
        resp_pii = comprehend.detect_pii_entities(Text=query, LanguageCode=lang_code)
        for pii in resp_pii['Entities']:
            if pii['Type'] not in ['NAME','AGE','ADDRESS','DATE_TIME']:
                pii_list.append(pii['Type'])
        if len(pii_list) > 0:
            answer = "I am sorry but I found PII entities " + str(pii_list) + " in your query. Please remove PII entities and try again."
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


def start_visualize_graph(start, graph_results):
    nodes = []
    edges = []
    config = Config(width=600,
                height=600,
                directed=True, 
                physics=True, 
                hierarchical=False
                )

    nodes.append(Node(id=start, 
                   label=start, 
                   size=15,
                   color="#F48B94", 
                   shape="hexagon"))
    for i, path in enumerate(graph_results):
        if i <= 15: #showing just 10 results for brevity
            
            if len(path.split(',')) > 1:
                end = path.split(',')[1].strip()
                if end not in nodes:
                    nodes.append(Node(id=end, 
                           size=10,
                           label=end,
                           color="#F7A7A6",
                           shape="dot"))
                edges.append(Edge(source=start, 
                           label="", 
                           target=end
                           )) 
    return(agraph(nodes=nodes, 
                      edges=edges,
                      config=config))    

def end_visualize_graph(end, graph_results):
    nodes = []
    edges = []
    config = Config(width=600,
                height=600,
                directed=True, 
                physics=True, 
                hierarchical=False
                )

    nodes.append(Node(id=end, 
                   label=end, 
                   size=15,
                   color="#F48B94", 
                   shape="hexagon"))
    for i, path in enumerate(graph_results):
        if i <= 15: #showing just 10 results for brevity
            if len(path.split(',')) > 1:
                start = path.split(',')[1].strip()
                if start not in nodes:
                    nodes.append(Node(id=start, 
                           size=10,
                           label=start,
                           color="#F7A7A6",
                           shape="dot"))
                edges.append(Edge(source=start, 
                           label="", 
                           target=end
                           )) 
    return(agraph(nodes=nodes, 
                      edges=edges,
                      config=config))    


def start_end_visualize_graph(start, end, graph_results):
    nodes = []
    edges = []
    city_codes = []
    config = Config(width=600,
                height=600,
                directed=True, 
                physics=True, 
                hierarchical=False
                )

    for i, path in enumerate(graph_results):
        middle_paths = path.split(',')
        #st.text("middle paths are: " + str(middle_paths))
        limit = len(middle_paths)
        #st.text("limit is: " + str(limit))
        for x in range(0,limit,2):
            #st.text("x alone is: " + str(x))
            if middle_paths[x] not in city_codes:
                city_codes.append(middle_paths[x])
                nodes.append(Node(id=middle_paths[x], 
                   label=middle_paths[x], 
                   size=10,
                   color="#F48B94", 
                   shape="dot"))
            if x+2 <= limit-1:
                if middle_paths[x+2] not in city_codes:
                    city_codes.append(middle_paths[x+2])
                    nodes.append(Node(id=middle_paths[x+2], 
                       label=middle_paths[x+2], 
                       size=10,
                       color="#F48B94", 
                       shape="dot"))
            if x+1 <= limit-1:
                edges.append(Edge(source=middle_paths[x], 
                   label=str(middle_paths[x+1]), 
                   target=middle_paths[x+2]
                   ))     

    return(agraph(nodes=nodes, 
                      edges=edges,
                      config=config))    


c1, c2 = st.columns(2)
start_airport_code = c1.text_input('**Type a start airport code (for ex. LHR)**', key='text_ggsa')
end_airport_code = c2.text_input('**Type an end airport code (for ex. ATL)**', key='text_ggea')
mode = ''
if start_airport_code != '' and end_airport_code != '':
    mode = 'start-and-end'
    start_airport_code = call_anthropic("return only the three letter IATA code for "+start_airport_code)
    end_airport_code = call_anthropic("return only the three letter IATA code for "+end_airport_code)
    c1.text(start_airport_code)
    c2.text(end_airport_code)
    res = execute_genai_call_neptune(mode, start_airport_code.upper(), end_airport_code.upper())
    res1 = res['Payload']
    graph_results = json.load(res1)
    start_end_visualize_graph(start_airport_code.strip().upper(), end_airport_code.strip().upper(), graph_results)
    graph_results_summary = call_anthropic("Explain in 200 words what the graph results are depicting by interpreting the graph query results containing airport codes and miles: " + str(graph_results))
    st.write("**Graph results summary:**")
    st.write(graph_results_summary)
    p_text = call_anthropic('Generate three prompts to query the graph results containing airport codes and miles: '+ str(graph_results))
    p_text1 = []
    p_text2 = ''
    if p_text != '':
        p_text.replace("$","USD")
        p_text1 = p_text.split('\n')
        for i,t in enumerate(p_text1):
            if i > 1:
                p_text2 += t.split('\n')[0]+'\n\n'
        p_summary = p_text2
    st.sidebar.markdown('### Suggested graph query prompts \n\n' + 
            p_summary)

elif start_airport_code != '' and end_airport_code == '':
    start_airport_code = call_anthropic("return only the three letter IATA code for "+start_airport_code)
    c1.text(start_airport_code)
    mode = 'start'
    res = execute_genai_call_neptune(mode, start_airport_code.strip().upper(), None)
    res1 = res['Payload']
    graph_results = json.load(res1)
    start_visualize_graph(start_airport_code.upper(), graph_results)
    #st.write(graph_results)
    graph_results_summary = call_anthropic("Explain in 200 words what the graph results are depicting by interpreting the graph query results containing airport codes and miles: " + str(graph_results))
    st.write("**Graph results summary:**")
    st.write(graph_results_summary)
    p_text = call_anthropic('Generate three prompts to query the graph_results containing airport codes and miles: '+ str(graph_results))
    p_text1 = []
    p_text2 = ''
    if p_text != '':
        p_text.replace("$","USD")
        p_text1 = p_text.split('\n')
        for i,t in enumerate(p_text1):
            if i > 1:
                p_text2 += t.split('\n')[0]+'\n\n'
        p_summary = p_text2
    st.sidebar.markdown('### Suggested query prompts for source airports\n\n' + 
            p_summary)


elif start_airport_code == '' and end_airport_code != '':
    end_airport_code = call_anthropic("return only the three letter IATA code for "+end_airport_code)
    c2.text(end_airport_code)
    mode = 'end'
    res = execute_genai_call_neptune(mode, None, end_airport_code.strip().upper())
    res1 = res['Payload']
    graph_results = json.load(res1)
    end_visualize_graph(end_airport_code.upper(), graph_results)
    #st.write(graph_results)
    graph_results_summary = call_anthropic("Explain in 200 words what the graph results are depicting by interpreting the graph query results containing airport codes and miles: " + str(graph_results))
    st.write("**Graph results summary:**")
    st.write(graph_results_summary)
    p_text = call_anthropic('Generate three prompts to query the graph results containing airport codes and miles: '+ str(graph_results))
    p_text1 = []
    p_text2 = ''
    if p_text != '':
        p_text.replace("$","USD")
        p_text1 = p_text.split('\n')
        for i,t in enumerate(p_text1):
            if i > 1:
                p_text2 += t.split('\n')[0]+'\n\n'
        p_summary = p_text2
    st.sidebar.markdown('### Suggested query prompts for destination airports \n\n' + 
            p_summary)

st.write("What would you like the Graph to tell you?")
input_text = st.text_input('**Type your query**', key='text_gg')
p_summary = ''
if input_text != '' and mode != '':
    result = GetAnswers('Answer the query ' + input_text + ' from these graph query results containing airport codes and miles: '+str(graph_results))
    result = result.replace("$","\$")
    st.write(result)
    p_text = call_anthropic('Generate three prompts to query the graph results containing airport codes and miles: '+ str(graph_results))
    p_text1 = []
    p_text2 = ''
    if p_text != '':
        p_text.replace("$","USD")
        p_text1 = p_text.split('\n')
        for i,t in enumerate(p_text1):
            if i > 1:
                p_text2 += t.split('\n')[0]+'\n\n'
        p_summary = p_text2
    st.sidebar.markdown('### Suggested query prompts for routes \n\n' + 
            p_summary)
    

    





