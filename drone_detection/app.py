import cv2
import torch
import streamlit as st
from PIL import Image
from pathlib import Path
import os
import time
from ultralytics import YOLO

# Cloud-friendly configuration
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)

# Streamlit app setup
st.set_page_config(page_title="Road Damage Detection", layout="centered")
st.title("🛣️ Pothole & Crack Detection")

@st.cache_resource
def load_model():
    try:
        # Try loading from local weights first
        if Path('weights/best.pt').exists():
            return YOLO('weights/best.pt')
        
        # Fallback to pretrained model
        st.warning("Using standard YOLOv8 model (custom weights not found)")
        return YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# File upload section
col1, col2 = st.columns(2)
with col1:
    img_file = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])
with col2:
    weight_file = st.file_uploader("Upload custom weights (.pt)", type=["pt"])

# Handle weight upload
if weight_file:
    with st.spinner("Updating model..."):
        with open('weights/best.pt', 'wb') as f:
            f.write(weight_file.getbuffer())
        st.cache_resource.clear()
        st.rerun()

# Process image
if img_file and model:
    img = Image.open(img_file).convert("RGB")
    with st.expander("Original Image", expanded=True):
        st.image(img, use_container_width=True)
    
    with st.spinner("Detecting road damage..."):
        start = time.time()
        results = model(img)
        inference_time = time.time() - start
        
        with st.expander("Detection Results", expanded=True):
            st.image(
                results[0].plot(), 
                caption=f"Processed in {inference_time:.2f}s | Confidence threshold: 0.5",
                use_container_width=True
            )
        
        # Display detection metrics
        detections = results[0].boxes
        if len(detections) == 0:
            st.success("✅ No road damage detected")
        else:
            st.warning(f"🚨 Detected {len(detections)} road damage instances:")
            for cls_id in detections.cls.unique():
                cls_name = results[0].names[int(cls_id)]
                count = sum(detections.cls == cls_id)
                conf = detections[detections.cls == cls_id].conf.mean()
                st.write(f"- {count} {cls_name}(s) with average confidence {conf:.2f}")
