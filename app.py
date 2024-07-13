import os
import streamlit as st
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Function to download the model
def download_model(url, output):
    if not os.path.exists(output):
        st.write("Downloading model...")
        gdown.download(url, output, quiet=False)
        st.write("Model downloaded.")

# Model URL and path
model_url = 'https://drive.google.com/uc?id=18EeTVrZ6fnbSEJut9BEAZvTMUrFTZhsL'
model_path = 'model_15_88.h5'

# Download the model if not available
download_model(model_url, model_path)

# Load the model
def load_model_file():
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model_file()

# Labels
sorted_labels = ['Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp', 'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale']

# Function to perform image classification
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = sorted_labels[predicted_class]
    return predicted_label

# Streamlit app
def main():
    st.title('Image Classification')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        img_bytes = uploaded_file.read()
        st.image(img_bytes, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # Classify image
        st.write("Classifying...")
        if model is not None:
            predicted_label = classify_image(uploaded_file)
            st.write("Prediction:", predicted_label)
        else:
            st.write("Model could not be loaded.")

if __name__ == "__main__":
    main()
