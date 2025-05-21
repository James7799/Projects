import numpy as np
import cv2
import torch
from ultralytics import YOLO
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import time

# Set page config
st.set_page_config(page_title="Pothole and crack detection", layout="centered")

st.title("Pothole and crack detection")
st.write("Upload an image to detect pothole and crack detection on the road")

# Load YOLO model with error handling
@st.cache_resource
def load_model():
    try:
        # Update this path to your actual model file
        model = torch.load('best.pt')  # Using YOLO interface instead of torch.hub
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Perform inference with progress
        with st.spinner("Running detection..."):
            start_time = time.time()
            
            # Run prediction (remove device=0 if not using GPU)
            results = model.predict(image, verbose=False)
            
            # Convert result image
            result_img = results[0].plot()  # result image with boxes
            st.image(result_img, caption="Detected Image", use_container_width=True)
            
            # Show performance metrics
            inference_time = time.time() - start_time
            st.success(f"Detection completed in {inference_time:.2f} seconds!")
            
            # Show detection details
            for result in results:
                st.write(f"Detected {len(result.boxes)} objects:")
                for box in result.boxes:
                    st.write(f"- {result.names[box.cls.item()]} with confidence {box.conf.item():.2f}")

    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
