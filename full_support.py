import streamlit as st
import os
import cv2
import numpy as np
import joblib
from PIL import Image
import google.generativeai as genai
from pathlib import Path
from googletrans import Translator
from gtts import gTTS
import io
import base64

# Configure GenAI API key
genai.configure(api_key="Your_Api_key_here")

# Load the trained waste classification model
model = joblib.load(r'C:\Users\rathn\OneDrive\Documents\project_HACKHIVE\new_svm_model.pkl')

# Define a mapping of classes to recyclable or non-recyclable
recyclable_classes = {'paper', 'cardboard', 'plastic'}  # Adjust these class names based on your dataset
non_recyclable_classes = {'metal', 'glass', 'trash'}  # Adjust these class names based on your dataset

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = img.flatten()  # Flatten the image
    img = img.reshape(1, -1)  # Reshape for prediction
    return img

# Function to initialize the model
def initialize_model():
    generation_config = {"temperature": 0.7}  # Lower temperature for more focused responses
    return genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Function to generate waste-specific content based on prompts
def generate_waste_content(model, waste_type):
    prompt = f"How should we dispose of {waste_type}? Please provide environmentally friendly disposal methods."
    response = model.generate_content([prompt])
    
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            text_part = candidate.content.parts[0]
            if text_part.text:
                return text_part.text
            else:
                return "No valid content generated."
        else:
            return "No content parts found."
    else:
        return "No candidates found."

# Function to translate text into selected language
def translate_text(text, lang):
    translator = Translator()
    translation = translator.translate(text, dest=lang)
    return translation.text

# Function to convert text to speech and generate an audio file
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Streamlit app
def main():
    st.title("Waste Classification and Disposal Guidance")

    # Upload an image file
    uploaded_file = st.file_uploader("Choose a waste image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)

        # Display the image in the app
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image and classify it
        processed_image = load_and_preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_name = prediction[0]  # Directly use the predicted label
        st.write(f"Predicted Class: **{predicted_class_name}**")

        # Determine if the item is recyclable or not
        if predicted_class_name in recyclable_classes:
            waste_type = "recyclable waste"
            st.write("This item is **recyclable**.")
        elif predicted_class_name in non_recyclable_classes:
            waste_type = "non-recyclable waste"
            st.write("This item is **non-recyclable**.")
        else:
            waste_type = "unknown type of waste"
            st.write("Recyclability status is **unknown**.")

        # Initialize the GenAI model
        genai_model = initialize_model()

        # Generate waste-specific content
        content = generate_waste_content(genai_model, waste_type)
        st.write("Suggested Disposal Method:")
        st.write(content)

        # Generate and play the audio for the content
        audio_bytes = text_to_speech(content)
        st.audio(audio_bytes, format='audio/mp3')

        # Translation buttons for regional languages
        if st.button("Translate to Tamil"):
            tamil_translation = translate_text(content, 'ta')
            st.write(tamil_translation)
            tamil_audio_bytes = text_to_speech(tamil_translation, 'ta')
            st.audio(tamil_audio_bytes, format='audio/mp3')
        
        if st.button("Translate to Hindi"):
            hindi_translation = translate_text(content, 'hi')
            st.write(hindi_translation)
            hindi_audio_bytes = text_to_speech(hindi_translation, 'hi')
            st.audio(hindi_audio_bytes, format='audio/mp3')

        # Additional language translations can be added similarly...

if __name__ == "__main__":
    main()
