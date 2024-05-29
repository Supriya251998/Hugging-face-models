from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
import warnings
import requests
import os
import streamlit as st
load_dotenv(find_dotenv())
warnings.filterwarnings('ignore')
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def img2text(url):
    pipe=pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = pipe(url)[0]['generated_text']
    print(text)
    return text

import ollama

def generate_story(scenario):
    prompt = f"""You are a storyteller;
Generate a short story based on the {scenario} in less than 30 words;
Use the below output format;

context: {scenario}
Story:
"""
    response = ollama.chat(model='llama3', options={'temperature': 1}, messages=[{
        'role': 'user',
        'content': prompt,
    }])
    story = response['message']['content']
    print(story)
    return story

def text2speech(message,api_token):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_csmsc_conformer_fastspeech2"
    headers = {"Authorization": f"Bearer {api_token}"} 
    payloads = {
        'inputs': message
    }
 
    response = requests.post(API_URL, headers=headers, json=payloads)
    
    with open('audio.flac', 'wb') as file:
            file.write(response.content)
        
   ### with open ('audio.mp3','wb') as file:
        #file.write(response.content)
    #return response.content

    
def main():
    st.set_page_config(page_title="img 2 story")
    st.header("Turn image into a story")
    uploaded_file=st.file_uploader("Choose an image...",type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data =uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb')as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption='Uploaded image.',use_column_width=True)
        scenario=img2text(uploaded_file.name)
        story =generate_story(scenario)
        text2speech(story,HUGGINGFACEHUB_API_TOKEN)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")


if __name__ =='__main__':
    main()
