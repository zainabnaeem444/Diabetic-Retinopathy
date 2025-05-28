import os
os.environ["STREAMLIT_HOME"] = "/tmp"  # Fix permission issue on Hugging Face

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Set page config
st.set_page_config(page_title="DR Detector", layout="centered")

# Load model
@st.cache_resource
def load_dr_model():
    model_path = "final_epoch13_855acc_fullmodel.h5"
    return load_model(model_path)

model = load_dr_model()
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']

# UI Title
st.title("🩺 Diabetic Retinopathy Detector")
st.markdown("Upload a retina image to predict the **stage of DR** using an AI model.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload retina image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼️ Retina Image Preview', use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]

        st.success(f"🧠 Prediction: **{prediction}**")
        st.bar_chart(preds[0])
