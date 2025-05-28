import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os

# Model file ID extracted from your link
model_url = "https://drive.google.com/uc?id=1gwCm1YTUHxLtGl6Gj5Ck4RXvRj3M37am"
model_path = "final_epoch13_855acc_fullmodel.h5"

# Download the model if it's not already present
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(model_url, model_path, quiet=False)

# Load the model once and cache it
@st.cache_resource
def load_dr_model():
    return load_model(model_path)

model = load_dr_model()
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']

# Streamlit App UI
st.title("🩺 Diabetic Retinopathy Detector")
st.markdown("Upload a retina image to detect the stage of Diabetic Retinopathy using AI.")

uploaded_file = st.file_uploader("Choose a retina image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image Preview', use_column_width=True)

    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        predictions = model.predict(img_array)
        result = class_names[np.argmax(predictions)]
        st.success(f"🧠 Prediction: {result}")
        st.bar_chart(predictions[0])
