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
bucket = os.environ['bucket']
im_endpoint_name = os.environ['im_endpoint_name']

# Setup a SageMaker session and initialize the Stability AI Predictor with the SDXL endpoint
session = boto3.session.Session(aws_access_key_id=key,aws_secret_access_key=secret,region_name=region)
sm_session = sagemaker.Session(boto_session=session)

deployed_model = StabilityPredictor(endpoint_name=im_endpoint_name, sagemaker_session=sm_session)

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
# Set the maximum number of images
max_num_images = 3

st.set_page_config(
    page_title="SDXL Text-to-Image Media Use Cases",
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
st.markdown("# Generate more expressive, high quality images")

# Add a and description and instruction on the side bar
st.sidebar.header("GenAI for Media, Art and Creativity")
st.sidebar.markdown('''
        ### Example prompts for the image \n\n
        A photograph of fresh pizza with basil and tomatoes, from a traditional oven \n
        Close up shot of stones on a beach, natural night, bokeh \n
        cartoon raincoat lizard lost in a forest in the rain. Canon DSLR. Tilt - shift. \n
        butterfly \n
    ''')

def generate_images(num_output_images: int, prompt_text: Union[str, List], style: Union[str, List] = "cinematic", cfg: int=10, height=512, width=512) -> List[GenerationResponse]:
    """
    Calls deployed_model.predict num_output_images times with style_preset="cinematic"
    """
    outputs = []
    if isinstance(prompt_text, str):
        prompt_text = [prompt_text] * num_output_images
    if isinstance(style, str):
        style = [style] * num_output_images
    for i in range(num_output_images):
        outputs.append(deployed_model.predict(GenerationRequest(
            text_prompts=[TextPrompt(text=prompt_text[i])],
            style_preset=style[i],
            cfg_scale=cfg,
            clip_guidance_preset="FAST_BLUE",
            height=height,
            width=width,
        )))
    return outputs


def decode_and_show(model_response: Union[GenerationResponse, List[GenerationResponse]], prompt_text: str, image_styles: List = None) -> None:
    """
    Decodes and displays images from an AI model output or a list of AI model outputs
    If a `image_styles` list is provided, it will be used to label the images after printing the Seed

    Args:
        model_response (Union[GenerationResponse, List[GenerationResponse]]): A single SDXL samples or a list of samples.

    Returns:
        None
    """
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
            caption = artifact.seed
        # Display image using st.image()
        st.image(image, caption=prompt_text, use_column_width=True)

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


def generate_images_random_style(num_output_images: int, prompt_text: Union[str, List], cfg: int=10) -> Tuple[List[str], List[GenerationResponse]]:
    """
    Calls deployed_model.predict num_output_images times with a random style_preset
    """
    outputs = []
    random_style_presets = []
    if isinstance(prompt_text, str):
        prompt_text = [prompt_text] * num_output_images
        
    for i in range(num_output_images):
        random_style_preset = random.choice(available_style_presets)
        random_style_presets.append(random_style_preset)
        outputs.append(deployed_model.predict(GenerationRequest(
            text_prompts=[TextPrompt(text=prompt_text[i])],
            style_preset=random_style_preset,
            clip_guidance_preset="FAST_BLUE",
            cfg_scale=cfg,
            height=512,
            width=512,
            
        )))
    return random_style_presets, outputs

# Streamlit app code instruction
st.write("This example demonstrates how to use SDXL from Stability AI for some common media and creative use cases. One common complaint with text-to-image models, especially in automated production environments, is that prompt engineering is challenging or undesirable. Using only a **single word prompt**, SDXL can render high quality images.")

st.write("**Instructions:** \n 1. Enter your prompt below. This can be either a long prompt or a single-word prompt. \n 2. Select the style you would like to use from the available preset list or choose random. \n 3. Click Generate \n 4. The model generates a list of images for you.")
    
def main():
        
     # Get user prompt
    prompt_text = st.text_input('Enter your prompt here')
    
    # Allow the user to manually select a style or let the system choose for them with random
    style_selector = st.radio(
        "Manually select the style from the preset-list or choose a random 👉",
        options=["Manual", "Random"],
    )
    
    # check if the user want to manually select the style
    if style_selector == "Manual":   
    # Create a dropdown menu on the sidebar
        selected_style = st.selectbox("Available Style Presets", available_style_presets)
        
    # Let the user specify the number of images with a maximum value of 5 or should we increase to 10?
    desired_images = st.number_input(f"Enter the desired number of images to generate, maximum of {max_num_images}", value=1, step=1, max_value= max_num_images)

    # Check if desired number of images exceeds the maximum value
    if  desired_images >  max_num_images:
        st.warning(f"desired number of images cannot exceed the maximum value of {max_num_images}. Resetting to {max_num_images}.")
        desired_images = max_num_images
    
    # Convert input to integer
    desired_images = int(desired_images)

    # Make a prediction when the button is clicked
    if st.button("Generate"):
        if not prompt_text:
            st.warning('Please enter a prompt before continuing.')
        else:
            with st.spinner("Loading..."):

        # Call the function to generate the images using the SageMaker endpoint
                if style_selector == 'Random':
                    random_style_presets, outputs = generate_images_random_style(desired_images, prompt_text)
                    decode_and_show(outputs,prompt_text,random_style_presets)
                else:
                    outputs = generate_images(desired_images, prompt_text, selected_style)
                    decode_and_show(outputs,prompt_text)
                # Task completed,
                st.success("Completed!")

if __name__ == "__main__":
    main()
