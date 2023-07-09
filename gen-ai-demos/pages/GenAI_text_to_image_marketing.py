import streamlit as st
import sagemaker
from stability_sdk_sagemaker.predictor import StabilityPredictor
from stability_sdk.api import GenerationRequest, GenerationResponse, TextPrompt
from botocore.exceptions import ClientError

from PIL import Image
from typing import Union, List, Tuple
import io
import os
import base64
import boto3
import random
import time

# Getting environment variables
key = os.environ['AWS_ACCESS_KEY_ID']
secret = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['AWS_DEFAULT_REGION']
im_endpoint_name = os.environ['im_endpoint_name']

#bucket = os.environ['bucket'] # should we be saving images to the bucket under a specific prefix?

# Setup a SageMaker session and initialize the Stability AI Predictor with the SDXL endpoint
session = boto3.session.Session(aws_access_key_id=key,aws_secret_access_key=secret,region_name=region)
sm_session = sagemaker.Session(boto_session=session)

deployed_model = StabilityPredictor(endpoint_name=im_endpoint_name, sagemaker_session=sm_session)

st.set_page_config(
    page_title="SDXL Text-to-Image Marketing and Design Use Cases",
)

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and for internal AWS consumption.
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21). 
    - These can be shown to customers in a private setting under NDA. 
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    """)

# Add a title and description
st.markdown("# Re-imagine old, stock, and low quality images")

# Add a and description and instruction on the side bar
st.sidebar.header("GenAI for Media, Art and Creativity")
st.sidebar.markdown('''
        ### Example prompt for the sample image \n\n
        dress in stunning watercolour
    ''')

# These are the available style presets
available_style_presets = [
            "cinematic",
            "anime",
            "photographic",
            "digital-art",
            "comic-book",
            "fantasy-art",
            "line-art",
            "analog-film",
            "neon-punk",
            "isometric",
            "low-poly",
            "origami",
            "modeling-compound",
            "3d-model",
            "pixel-art",
            # "tile-texture",
        ]

def decode_and_show(model_response: Union[GenerationResponse, List[GenerationResponse]], image_styles: List = None) -> [int]:
    """
    Decodes and displays images from an AI model output or a list of AI model outputs
    If a `image_styles` list is provided, it will be used to label the images after printing the Seed

    Args:
        model_response (Union[GenerationResponse, List[GenerationResponse]]): A single SDXL samples or a list of samples.

    Returns:
        [int]
    """
    seeds = []
    if image_styles is not None:
        assert len(image_styles) == len(model_response), "image_styles must be the same length as model_response"

    # Helper function to display one image
    def show_image(artifact) -> None:
        image = artifact.base64
        image_data = base64.b64decode(image.encode())
        image = Image.open(io.BytesIO(image_data))
        if image_styles is not None:
            print("Style: ", image_styles.pop(0), " Seed: ", artifact.seed)
        else:
            print("Seed: ", artifact.seed)
        # Display image
        st.image(image, use_column_width=True)
        seeds.append(artifact.seed)

    # Case 1: Single GenerationResponse
    if isinstance(model_response, GenerationResponse):
        # Case 1.1: with one image
        if len(model_response.artifacts) == 1:
            show_image(model_response.artifacts[0])
        # Case 1.2: with multiple images
        else:
            for artifact in model_response.artifacts:
                show_image(artifact)
    # Case 2: List of GenerationResponse objects
    elif isinstance(model_response, list) and all(isinstance(item, GenerationResponse) for item in model_response):
        for response in model_response:
            show_image(response.artifacts[0])
    else:
        print("Input type not supported.")
    return seeds


def encode_image(image_data: None, resize: bool = True) -> Union[str, None]:
    """
    Encode an image as a base64 string, optionally resizing it to 512x512.

    Args:
        image_path (str): The path to the image file.
        resize (bool, optional): Whether to resize the image. Defaults to True.

    Returns:
        Union[str, None]: The encoded image as a string, or None if encoding failed.
    """

    if resize:
        image = Image.open(image_data)
        image = image.resize((512, 512))
        image_resized_path = f"image_base_resized.png"
        image.save(image_resized_path)
        image_path = image_resized_path
    image = Image.open(image_path)
    assert image.size == (512, 512)
    with open(image_path, "rb") as image_file:
        img_byte_array = image_file.read()
        # Encode the byte array as a Base64 string
        try:
            base64_str = base64.b64encode(img_byte_array).decode("utf-8")
            return base64_str
        except Exception as e:
            print(f"Failed to encode image {image_path} as base64 string.")
            print(e)
            return None

# Streamlit app code instruction
st.write("This example demonstrates how to use SDXL from Stability AI for some common media and creative use cases. ")

st.write("**Instructions:** \n - Browse and select your image. You can [download the sample](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Liberty_and_Company_tea_gown_c._1887.jpg/1200px-Liberty_and_Company_tea_gown_c._1887.jpg) or use your own. \n - Enter your prompt to re-imagine your original image. \n - Choose a style from the list of presets. \n - Adjust additional settings. \n - Generate the new image ")
    
def main():
     # Initialize the session state
    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False
        
    # Add a file uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Check if a file was uploaded
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image)
        uploaded_image = encode_image(uploaded_image)
        st.session_state.image_uploaded = True
        
    user_prompt= st.text_input("**You can re-imagine this image with a brand new style**") #text="dress in stunning watercolour"
    
    # Create a dropdown menu for the user to select the style
    selected_style = st.selectbox('Available Style Presets', available_style_presets)
    
    # Allow the user to choose an image strength to controle creativity
    selected_image_strength = st.slider("image strength", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
    
    # What about the seed?
    selected_seed = st.slider("Seed", min_value=42, max_value=4294967295, value=42, step=1)
    
    # What about the generation step? 
    selected_step = st.slider("Step", min_value=10, max_value=150, value=50, step=1)
    
    # Make a prediction when the button is clicked
    if st.button('Generate'):
        if not st.session_state.image_uploaded:
            st.write("You must upload an image first!")
        elif not user_prompt:
            st.warning('Please enter a prompt before continuing.')
            
        else:
            with st.spinner("Loading..."):
                output = deployed_model.predict(GenerationRequest(text_prompts=[TextPrompt(text=user_prompt)],
                                                  init_image= uploaded_image,
                                                  cfg_scale=9,
                                                  style_preset=selected_style,
                                                  image_strength=selected_image_strength,
                                                  seed=selected_seed,
                                                  step=selected_step
                                                  ))
                decode_and_show(output) 
                # Task completed, display the result
                st.success("Completed!")
    
    # -delete the saved images so it does not fill-up the system now that we are done ----------
    # Get the current directory
    current_path = os.getcwd()

    # Get a list of all files in the current directory
    files = os.listdir(current_path)

    # Iterate over the files and delete the image files
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            file_path = os.path.join(current_path, file)
            os.remove(file_path)
            
            
if __name__ == '__main__':
    main()
