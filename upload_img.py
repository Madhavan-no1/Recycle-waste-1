
import streamlit as st
import os
import cv2
import numpy as np
import joblib
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("GENAI_API_KEY")

# Configure GenAI API key
if api_key is None:
    st.error("API key is not set. Please check your .env file.")
else:
    genai.configure(api_key=api_key)


# Load the trained waste classification model
model = joblib.load(r'C:\Users\rathn\OneDrive\Documents\project_HACKHIVE\new_svm_model.pkl')

# Define waste categories based on labels
recyclable_classes = {
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard',
    'glass', 'magazines', 'paper', 'paper_cups', 'plastic_cup_lids',
    'plastic_detergent_bottles', 'plastic_food_containers', 'plastic_soda_bottles',
    'steel_food_cans'
}

non_recyclable_classes = {
    'disposable_plastic_cutlery', 'plastic_shopping_bags', 'plastic_straws',
    'styrofoam_cups', 'styrofoam_food_containers', 'trash'
}

organic_classes = {
    'coffee_grounds', 'eggshells', 'food_waste', 'tea_bags'
}

other_classes = {
    'clothing', 'shoes'
}

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = img.flatten()  # Flatten the image
    img = img.reshape(1, -1)  # Reshape for prediction
    return img

# Function to initialize the GenAI model
def initialize_model():
    generation_config = {"temperature": 0.5}  # Adjusted temperature for focused responses
    return genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Function to generate waste-specific content with a focus on short, practical advice
def generate_short_disposal_content(model, waste_type):
    prompt = f"Provide a brief, practical guide for disposing of {waste_type} in India. Include bin color and any special instructions."
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

# Streamlit app
def main():
    st.title("Waste Classification and Disposal Guidance in India")

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

        # Determine the waste type and provide disposal instructions specific to India
        if predicted_class_name in recyclable_classes:
            waste_type = "recyclable waste"
            st.write("This item is **recyclable**. Place it in the **blue bin** for recycling.")
        elif predicted_class_name in non_recyclable_classes:
            waste_type = "non-recyclable waste"
            st.write("This item is **non-recyclable**. Dispose of it in the **red bin** for non-recyclable waste.")
        elif predicted_class_name in organic_classes:
            waste_type = "organic waste"
            st.write("This item is **organic**. Place it in the **green bin** for composting.")
        elif predicted_class_name in other_classes:
            waste_type = "special disposal waste"
            st.write("This item requires **special disposal**. Check local guidelines for proper disposal.")
        else:
            waste_type = "unknown type of waste"
            st.write("Recyclability status is **unknown**. Follow local guidelines for disposal.")

        # Initialize the GenAI model
        genai_model = initialize_model()

        # Generate short, practical waste-specific content
        content = generate_short_disposal_content(genai_model, waste_type)
        st.write("Suggested Disposal Method:")
        st.write(content)

if __name__ == "__main__":
    main()
