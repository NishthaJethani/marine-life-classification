import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
import numpy as np
from PIL import Image

# URL to download the model from Google Drive
MODEL_URL = "https://drive.google.com/file/d/1Gk6JGKlnlx8ZjrzbNFUerRZCI_ccWcrY/view?usp=sharing"  # Replace YOUR_FILE_ID with the actual file ID
MODEL_PATH = "./model_15_88.h5"

# Function to download the model if not already downloaded
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success('Model downloaded successfully!')

# Load the trained model
def load_model_file():
    download_model()
    return load_model(MODEL_PATH, compile=False)

# Load model
model = load_model_file()

# Labels
sorted_labels = ['Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp', 'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale']

# Function to perform image classification
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Perform prediction
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
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # Classify image
        st.write("Classifying...")
        predicted_label = classify_image(uploaded_file)
        st.write("Prediction:", predicted_label)

if __name__ == "__main__":
    main()
