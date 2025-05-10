import cv2
import torch
import streamlit as st
from PIL import Image
from pathlib import Path
import os
import time
import gdown

# Configuration
os.environ['YOLO_CONFIG_DIR'] = str(Path.home() / '.config' / 'Ultralytics')
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)

# Streamlit setup
st.set_page_config(page_title="Pothole Detection", layout="centered")
st.title("🕳️ Pothole and Crack Detection")
st.write("Upload an image to detect road damage")

@st.cache_resource
def load_model():
    try:
        model_path = Path("weights/best.pt")
        model_path.parent.mkdir(exist_ok=True)
        
        # Download weights if missing (Google Drive example)
        if not model_path.exists():
            with st.spinner("Downloading model weights..."):
                gdown.download(
                    "https://drive.google.com/uc?id=1zEyenhof2r5WfcaMtcpSqcRMqgGm1b9O",
                    str(model_path),
                    quiet=False
                )
        
        # Load model
        model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path=str(model_path),
            trust_repo=True
        )
        return model
        
    except Exception as e:
        st.error(f"""Model loading failed: {str(e)}
        
        Solutions:
        1. For local use: Place 'best.pt' in /weights/
        2. For cloud: Update Google Drive ID
        3. Check console logs""")
        return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose road image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)
        
        with st.spinner("Analyzing road surface..."):
            start = time.time()
            results = model(img)
            st.image(
                results.render()[0], 
                caption=f"Detections ({time.time()-start:.2f}s)",
                use_container_width=True
            )
            
            # Display results
            counts = results.pandas().xyxy[0]['name'].value_counts()
            st.write("**Detection Summary:**")
            for obj, count in counts.items():
                st.write(f"- {count} {obj}{'s' if count > 1 else ''}")
                
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
