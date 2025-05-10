!pip install ultralytics
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# Set page config
st.set_page_config(page_title="Drone Detection", layout="centered")

st.title("🛸 Drone Detection using YOLOv11")
st.write("Upload an image to detect drones using a trained YOLO model.")

# Load YOLO model (make sure the .pt file is in the same directory or update the path)
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolo11", "custom", path="yolo11n.pt", force_reload=True)
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    with st.spinner("Running detection..."):
        results = model.predict(image, device=0, verbose=False)

        # Convert result image
        result_img = results[0].plot()  # result image with boxes
        st.image(result_img, caption="Detected Image", use_column_width=True)
