# import openai library

import openai
import os
from dotenv import load_dotenv;


# Import and configure and connect to open ai API key
load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

openai.api_key = api_key;
#connect to openai api and verify the connection
enginelist=openai.Engine.list()
models=openai.Model.list()

# print(enginelist);
# ------1

# //prepare the request
model="text-davinci-002"
prompt="Write a short story about a person who is a doctor."
temperature=0.7 #how trandowm the anser chatgpt generated 1 is highest(diverse and creative) 0.1 determine
max_tokens=150 #nax amount of token the response will have, no more than 100 in this case
n=3
top_p=1.0 

# //send teh request
response=openai.Completion.create(
    engine=model, prompt=prompt, n=1
)
# //process the responses and display the response
for i, option in enumerate(response.choices):
    generated_text=response.choices[0].text.strip()
    print(f"{i}: {generated_text}")

# -------1


# Import spacy
#  //install python -m download en_core_web_md first
# Model_apacy=spacy.load("en_core_web_md")
# Analysis=model_spacy(generated_text)
for token in Analysis:
    print(token.text, token.pos_, token.dep_)

for ent in Analysis.ents:
    print(ent.text, ent.label_)
