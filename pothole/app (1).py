from inference_sdk import InferenceHTTPClient
import streamlit as st
from PIL import Image
import requests
import io

# Use secrets for the API key
API_KEY = st.secrets["roboflow"]["api_key"]

CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",
    api_key=API_KEY
)

st.title("Pothole and Crack Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running detection..."):
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        result = CLIENT.infer(img_bytes.getvalue(), model_id="road-potholes-and-cracks-jmxtp/3")

    st.json(result)
