

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
# Use try-except for imports
try:
    from roboflow import Roboflow
    import supervision as sv
    import cv2
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Rest of your app code...

# Set page config
st.set_page_config(layout="wide")
st.title("🚧 Pothole Detection System")

# Initialize Roboflow model
@st.cache_resource
def load_model():
    rf = Roboflow(api_key="yFWOQegLigSIyK7DbeZP")
    project = rf.workspace().project("road-potholes-and-cracks-jmxtp")
    return project.version(3).model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a road image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Save to temp file for prediction
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp:
        image.save(temp.name)
        
        # Make prediction
        result = model.predict(temp.name, confidence=40, overlap=30).json()
    
    # Process detections
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result)
    
    # Annotate image
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    annotated_image = mask_annotator.annotate(scene=image_np.copy(), detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=detections, 
        labels=labels
    )
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(annotated_image, caption="Detected Potholes", use_column_width=True)
    
    # Show detection summary
    st.subheader("Detection Summary")
    st.write(f"Found {len(detections)} potholes/cracks")
    st.json(result)
