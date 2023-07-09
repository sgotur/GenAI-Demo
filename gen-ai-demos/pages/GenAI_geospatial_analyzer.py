import json
import boto3
import streamlit as st
import datetime
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import uuid
import ai21
import string
import streamlit.components.v1 as components
import anthropic
import os

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']

if 'map_summary' not in st.session_state:
    st.session_state['map_summary'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
location = boto3.client('location')
location_place_index = 'geospatial-analyzer-index'

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']

st.set_page_config(page_title="GenAI Geospatial Analyzer", page_icon="flashlight", layout="wide")

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

st.sidebar.header("GenAI Geospatial Analyzer")
height = st.sidebar.slider("Height", 200, 1500, 300, 50)
width = st.sidebar.slider("Width", 200, 1500, 600, 50)

st.markdown("# Derive geospatial insights from maps")

map_html = """
    <html>
    <head>
    <link href="https://unpkg.com/maplibre-gl@1.14.0/dist/maplibre-gl.css" rel="stylesheet" />
    <style>
      body { margin: 0; }
      #map { height: 100vh; }
    </style>
  </head>
  <body>
    <div id="map" />
    <script src="https://unpkg.com/maplibre-gl@1.14.0/dist/maplibre-gl.js"></script>
    <script>
      const apiKey = "v1.public.eyJqdGkiOiI1NmViYmQ1MS1jZGM0LTRlZmQtOWQ3OS05MWIzOWIyODVjNzAifUy6CpGp_8LiWFF2-i_vaM3ZY9IRnJ90bu7d4y8EqmIm5S3eS8ddvSSF4otYnQVTL4BWte0PtXMW5WrkV2v3BjDsQrv5wIF4qLWHbNthKzLDJlcbqiRZVy916uINlbbeSu76uKY39LXffKAjRi6-WdLuWAPycNhusF8USJTV87ZHOrzMzWsle1ikU_PdYplv7KRihyOQae-CGtf-f4u_0Zx7UFZCyWmE4qka6JQ_a7vGctLpIiVGVamACa0iDj393XckJPioMrgtkV4FFKpgcNIGJaAiL0BsotwoXFNja1Rmp2i4j2qCoiKN-KnSxOYiaX8bHAdkm0oYwJSLEqkx8UE.ZWU0ZWIzMTktMWRhNi00Mzg0LTllMzYtNzlmMDU3MjRmYTkx";
      const region = "us-east-1";
      const mapName = "streamlit-geospatial-analyzer";
      const styleUrl = `https://maps.geo.${region}.amazonaws.com/maps/v0/maps/${mapName}/style-descriptor?key=${apiKey}`;
      const map = new maplibregl.Map({
        container: "map",
        style: styleUrl,
        center: [0,0],
        zoom: 10,
      });
      map.addControl(new maplibregl.NavigationControl(), "top-left");
      <blank>
        </script>
    </body>
    </html>
    """

map_dict = {}
map_df = pd.DataFrame(columns=['lat','lon'])
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

    if query == "cancel":
        answer = 'It was swell chatting with you. Goodbye for now'
    
    #elif sentiment == 'NEGATIVE':
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'

    #elif query_type == "BEING":
    #    answer = 'I do not answer questions that are negatively worded or that concern me at this time. Kindly rephrase your question and try again.'
            
    else:
        generated_text = ''
        if model.lower() == 'anthropic claude':  
            generated_text = call_anthropic(summary+'. Answer from this summary: '+ query.strip("query:"))
            if generated_text != '':
                answer = str(generated_text)+' '
            else:
                answer = 'Claude did not find an answer to your question, please try again'   
    return answer          

def plot_map(text_location):
    res = location.search_place_index_for_text(
    IndexName=location_place_index,
    MaxResults=1,
    Text=text_location
    )
    address = ''
    for result in res['Results']:
        map_df.loc[0,'lon'] = result['Place']['Geometry']['Point'][0]
        map_df.loc[0,'lat'] = result['Place']['Geometry']['Point'][1]
        address = result['Place']['Label']
    return address, map_df

def search_map(bias_position, search_text):
    res = location.search_place_index_for_text(
    BiasPosition=[
        bias_position.loc[0,'lon'],
        bias_position.loc[0,'lat']
    ],
    IndexName=location_place_index,
    MaxResults=15,
    Text=search_text
    )
    marker_sub = 'new maplibregl.Marker().setLngLat([long, lati]).setPopup(new maplibregl.Popup().setHTML("<p>Hello</p>")).addTo(map);'
    markers = ''
    places = ''
    for i, result in enumerate(res['Results']):
        i+=1
        markers += marker_sub.replace('long',str(result['Place']['Geometry']['Point'][0])).replace('lati',str(result['Place']['Geometry']['Point'][1])).replace("Hello",str(i)+'. '+result['Place']['Label']) + '\n'
        places += str(i)+'. '+result['Place']['Label'] + '\n' 
    return places, markers

st.write("**Instructions:** \n - Enter a point of interest \n - You will see it plotted on the map and a summary \n - Type your query in the search bar to get more insights")

c1, c2 = st.columns(2)
c1.subheader("Search for a location")
what_text = c1.text_input("**What are you searching for?**", key="what_location")
where_text = c1.text_input("**Where?**", key="where_location")
map_summary = ''
p_summary = ''
if what_text != '' and where_text != '':
    address, map_df = plot_map(where_text)
    places, markers = search_map(map_df,what_text)
    where_cent = '['+str(map_df.loc[0,'lon'])+','+str(map_df.loc[0,'lat'])+']'
    map_html = map_html.replace('[0,0]',where_cent)
    map_html = map_html.replace('<blank>',markers)
    components.html(
    map_html,
    height=height,
    width=width)
    st.write(address)
    map_summary = call_anthropic('Organize each location in bullets and provide a 100 word summary for each location and ensure all locations mentioned are explained: '+places)

if len(map_summary) >= 5:
    st.session_state['map_summary'] = map_summary

if st.session_state.map_summary:
    st.markdown('**Map summary**: \n')
    st.write(str(st.session_state['map_summary']))
    if model.lower() == 'anthropic claude':  
        p_text = call_anthropic('Generate three prompts to query the map summary: '+ st.session_state.map_summary)
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
        result = GetAnswers(st.session_state.map_summary,input_text)
        result = result.replace("$","\$")
        #lang = comprehend.detect_dominant_language(Text=result)
        #lang_code = str(lang['Languages'][0]['LanguageCode']).split('-')[0]
        #if lang_code in ['en']:
        #    resp_pii = comprehend.detect_pii_entities(Text=result, LanguageCode=lang_code)
        #    immut_summary = result
        #    for pii in resp_pii['Entities']:
                #if pii['Type'] not in ['ADDRESS','DATE_TIME']:
        #        pii_value = immut_summary[pii['BeginOffset']:pii['EndOffset']]
        #        result = result.replace(pii_value, str('PII - '+pii['Type']))
        st.write(result)





