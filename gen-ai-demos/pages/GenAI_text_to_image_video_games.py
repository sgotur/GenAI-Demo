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

#bucket = os.environ['bucket'] # should we be saving images to the bucket under a specific prefix rather than on the local file system?

# Setup a SageMaker session and initialize the Stability AI Predictor with the SDXL endpoint
session = boto3.session.Session(aws_access_key_id=key,aws_secret_access_key=secret,region_name=region)
sm_session = sagemaker.Session(boto_session=session)

deployed_model = StabilityPredictor(endpoint_name=im_endpoint_name, sagemaker_session=sm_session)

# Hard-coded pre-fixes. Should we let the user set these two parameters or show it to them?
avatar_fixed_prompt = "headshot portait of a "
avatar_fixed_styles = ", isometric sheer prismatic material, sunrise, crystal circlet"

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
st.markdown("# Video Games: Generate Character Variants")

# Add a and description and instruction on the side bar
st.sidebar.header("GenAI for Media, Art and Creativity")
st.sidebar.markdown('''
        ### Example prompts for the image \n\n
        start with "**beautiful princess**" \n
        Add your own variation to the generated image eg **princess, evil, dark art, dark magic, piercing stare, ominous, doom, yikes, horror**. \n
    ''')

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


def encode_image(image_path: str, resize: bool = True) -> Union[str, None]:
    """
    Encode an image as a base64 string, optionally resizing it to 512x512.

    Args:
        image_path (str): The path to the image file.
        resize (bool, optional): Whether to resize the image. Defaults to True.

    Returns:
        Union[str, None]: The encoded image as a string, or None if encoding failed.
    """
    assert os.path.exists(image_path)

    if resize:
        image = Image.open(image_path)
        image = image.resize((512, 512))
        image_base = os.path.splitext(image_path)[0]
        image_resized_path = f"{image_base}_resized.png"
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

def generate_image_var(image_prompt: str, init_image: str = None, cfg_scale: int = None, image_strength: float = None, style_preset: str = None, seed: int = None):
    generation_request = GenerationRequest(
        text_prompts=[TextPrompt(text=image_prompt)],
        init_image=init_image,
        cfg_scale=cfg_scale,
        image_strength=image_strength,
        style_preset=style_preset,
        seed=seed,
    )
    return deployed_model.predict(generation_request)

def campaign_images(prompt: str, init_image: str = None, style_preset: str = None, seed: int = None, img_strength=None):
    if not img_strength: 
        img_strength = [0.5, 0.3]
    if not init_image:
        first_image = generate_image_var(prompt, style_preset=style_preset, seed=seed)
        _ = decode_and_show(first_image)
        return [first_image.artifacts[0].base64, _]
    else:
        for img_strength in img_strength:
            output = generate_image_var(prompt, init_image=init_image, cfg_scale=10, image_strength=img_strength, style_preset=style_preset, seed=seed)
            decode_and_show(output)

# Streamlit app code instruction
st.write("**Use case**: 2 images are generated and the user selects their favorite. Its easy to immediately generate more image suggestions based on the user's preference (click) or by preference + prompt modification..")
st.write("**Instructions:** \n 1. Enter your prompt below and click generate \n 2. We randomly select one of the 2 images and the model generates additonal variants based on that image. \n 3. Add your own variation with additional prompting and click Adjust\n 4.  The model generates a new variation of the image.")
    
def main():
        
     # Get user prompt
    user_avatar_prompt = st.text_input("**Enter your Avatar prompt here**")
    
    encoded_avatar_image =None  
    
    # Initialize the session state
    if "generate_clicked" not in st.session_state:
        st.session_state.generate_clicked = False

    # Make a prediction when the button is clicked
    if st.button("Generate"):
        if not user_avatar_prompt:
            st.warning('Please enter a prompt before continuing.')
        else:
            with st.spinner("Loading..."):
                # Intialize user preference variables with None
                user_pref_image =None
                user_choice = None
                fav_image_data = None
                
                merge_avatar_prompt = avatar_fixed_prompt + user_avatar_prompt + avatar_fixed_styles
                favorites_image, fav_seed = campaign_images(merge_avatar_prompt, style_preset="fantasy-art")

                # We can append different fixed styles (or user selected)
                avatar_fixed_styles_2 = "short hair, age twenty four"
        
                # Concatenate the new prompt
                merge_avatar_prompt = avatar_fixed_prompt + user_avatar_prompt + avatar_fixed_styles_2
                favorites_image2, fav_seed2 = campaign_images(merge_avatar_prompt, style_preset="fantasy-art")
                
                # Randomly select one  of the above images
                random_choice = random.choice(["1", "2"])

                # Display the selected item
                st.write("**In some applications, a user may select their favorite out of a selection of images. We'll randomly one of the image above as a an init image and create variations. The randomly selected image 👉 is :**",random_choice)

                if random_choice== "1":
                    fav_image_data = base64.b64decode(favorites_image.encode()) 
                    user_pref_image = fav_seed[0]                     
                        
                else:
                    fav_image_data = base64.b64decode(favorites_image2.encode())
                    user_pref_image = fav_seed2[0]
                   
                        
                image = Image.open(io.BytesIO(fav_image_data))
                image.save(str(user_pref_image) + ".png") 
            
                encoded_avatar_image = encode_image(str(user_pref_image) + ".png")
            
                no_prompt_needed = ""
                with st.spinner("Loading..."):
                    campaign_images(no_prompt_needed, init_image=encoded_avatar_image, style_preset= "fantasy-art")
                    st.session_state.generate_clicked = True
            
    # Let's offer the user some variations with additional prompting
    adjust_prompt= st.text_input("**Let's add our own variations with additional prompting**")
    
    if st.button("Adjust"):
        if not st.session_state.generate_clicked:
            st.write("You must Generate an avatar first!")
        elif not adjust_prompt:
            st.warning('Please enter a prompt before continuing.')
        else:
            with st.spinner("Loading..."):
                low_overlay = [0.25, 0.2]
                campaign_images(adjust_prompt, init_image=encoded_avatar_image, style_preset= "fantasy-art", img_strength=low_overlay)
                st.success("Completed!")
                # reset the session
                st.session_state.generate_clicked = None
        
    # --------------- delete the saved images so it does not fill-up the system now that we are done ----------
    # Get the current directory
    current_path = os.getcwd()

        # Get a list of all files in the current directory
    files = os.listdir(current_path)

    # Iterate over the files and delete the image files
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            file_path = os.path.join(current_path, file)
            os.remove(file_path)

        

if __name__ == '__main__':
    main()
