import json
import boto3
import botocore
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
import sagemaker

key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
falcon_endpoint = os.environ['falcon_endpoint_name']

if 'map_summary' not in st.session_state:
    st.session_state['map_summary'] = None
if 'img_url' not in st.session_state:
    st.session_state['img_url'] = None
if 'img_prefix' not in st.session_state:
    st.session_state['img_prefix'] = None

s3 = boto3.client('s3',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
comprehend = boto3.client('comprehend',region_name='us-east-1',aws_access_key_id=key,aws_secret_access_key=secret)
bedrock = boto3.client('bedrock',region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)
location = boto3.client('location')
location_place_index = 'geospatial-analyzer-index'

# Get environment variables
ant_api_key = os.environ['ant_api_key']
bucket = os.environ['bucket']
im_endpoint_name = "jumpstart-nexity-sd-ft-model-txt2img-st-2023-06-28-21-15-40-915"
tx_endpoint_name = os.environ['tx_endpoint_name']
#br_endpoint_name = os.environ['fsi_index_id']
ant_name = os.environ['ant_name']

st.set_page_config(page_title="GenAI Property Visualizer", page_icon="house_buildings", layout="wide")

def call_anthropic(query):
    c = anthropic.Client(ant_api_key)
    resp = c.completion(
        prompt=anthropic.HUMAN_PROMPT+query+anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=1024,
    )
    return resp['completion']

def query_im_endpoint(text):
    #payload = {
    #"prompt": text,
    #"width": 480,
    #"height": 480,
    #"num_inference_steps": 200,
    #"seed": 42,
    #"guidance_scale": 8.5
    #}
    negative_prompts=[]
    payload = {"prompt": text, "negative_prompt": negative_prompts, "seed": 0}
    body = json.dumps(payload).encode('utf-8')
    predictor = sagemaker.Predictor(endpoint_name=im_endpoint_name)
    response = predictor.predict(body, {
            "ContentType": "application/x-text",
            "Accept": "application/json",
        },)
    #response = sagemaker_runtime.invoke_endpoint(EndpointName=im_endpoint_name, ContentType='application/x-text', Body=body, Accept='application/json;jpeg')
    return response

def parse_im_response(query_im_response):
    response_dict = json.loads(query_im_response)
    return response_dict['generated_image'], response_dict['prompt']

def save_image(img, prmpt):
    plt.figure(figsize=(6,6))
    plt.imshow(np.array(img))
    plt.axis('off')
    #plt.title(prmpt)
    prefix = "test-"+str(uuid.uuid4())+".jpg"
    plt.savefig("/tmp/"+prefix)
    #print("image name before S3 upload is: " + "/tmp/"+prefix)
    s3.upload_file("/tmp/"+prefix, bucket, prefix)
    img_url = 'https://dzlvehx4kcg5h.cloudfront.net/'+prefix
    return prefix, img_url


st.sidebar.header("GenAI Property Visualizer")
height = st.sidebar.slider("Height", 200, 1500, 300, 50)
width = st.sidebar.slider("Width", 200, 1500, 600, 50)

st.markdown("# Visualize buildings on a map")

map_html = """
<html>
<head>
<meta charset="utf-8">
<title>Dynamically visualize buildings</title>
<meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no">
<link href="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css" rel="stylesheet">
<script src="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js"></script>
<style>
body { margin: 0; padding: 0; }
#map { position: absolute; top: 0; bottom: 0; width: 100%; }
</style>
</head>
<body>
<div id="map"></div>

<script>
    mapboxgl.accessToken = 'pk.eyJ1IjoicHJlbXJhbmdhIiwiYSI6ImNsamZzMXI1NDAydjUzaW42MHUwaTNkcXAifQ.C-dGpcntgJfCVkwPSDt4Xw';
    const map = new mapboxgl.Map({
        container: 'map', // container ID
        // Choose from Mapbox's core styles, or make your own style with Mapbox Studio
        style: 'mapbox://styles/premranga/cljhumxdc002r01qph2pugsq3', // style URL
        zoom: 18, // starting zoom
        center: [0,0], // starting position
        pitch: 60, // pitch in degrees
        bearing: -80 // bearing in degrees
    });

    map.on('load', () => {
        // Load an image from an external URL.
        map.loadImage(
            'BUILDING_IMAGE_URL',
            (error, image) => {
                if (error) throw error;

                // Add the image to the map style.
                map.addImage('Nexity_Building', image);

                // Add a data source containing one point feature.
                map.addSource('point', {
                    'type': 'geojson',
                    'data': {
                        'type': 'FeatureCollection',
                        'features': [
                            {
                                'type': 'Feature',
                                'geometry': {
                                    'type': 'Point',
                                    'coordinates': [0,0]
                                }
                            }
                        ]
                    }
                });

                // Add a layer to use the image to represent the data.
                map.addLayer({
                    'id': 'points',
                    'type': 'symbol',
                    'source': 'point',
                    'layout': {
                        'icon-image': 'Nexity_Building',
                        'icon-size': 0.40,
                        'icon-allow-overlap': true
                    }
                });
            }
        );
    });
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

st.subheader("Dynamically model buildings and visualize")

map_summary = ''
p_summary = ''
what_text = st.text_input("**What building would you like to visualize?**", key="what_location")
if what_text != '':
    # Call the Stability model to get the image for our query, save it in S3 and build a response card
    if st.button('Generate image'):
        response = query_im_endpoint("Detailed image of " + "<<NEXITY_CONDO>> "+what_text)
        timg, prmpt = parse_im_response(response)
        prefix, img_url_new = save_image(timg, what_text)
        st.session_state.img_url = img_url_new
        st.session_state.img_prefix = prefix
    
    if st.session_state.img_url:
        st.image(st.session_state.img_url)

where_text = st.text_input("**Where would you like to place your image?**", key="where_location")
if where_text != '':
    address, map_df = plot_map(where_text)
    where_cent = '['+str(map_df.loc[0,'lon'])+','+str(map_df.loc[0,'lat'])+']'
    map_html = map_html.replace('[0,0]',where_cent)
    map_html = map_html.replace('BUILDING_IMAGE_URL',st.session_state.img_url)
    components.html(
    map_html,
    height=height,
    width=width)
    map_summary = call_anthropic('Explain the key features, the places of interest in the neighborhood for a real estate developer who wants to construct ' +what_text+ ' in this address: '+str(address).replace(',',''))

if len(map_summary) >= 5:
    st.session_state['map_summary'] = map_summary

if st.session_state.map_summary:
    st.markdown('**Location summary**: \n')
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
        st.write(result)





