import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import requests
import io

# Page settings
st.set_page_config(page_title="Pothole and Crack Detection", layout="centered")
st.title("🕳️ Pothole and Crack Detection")
st.write("Upload an image to detect potholes and cracks using Roboflow model.")

# Initialize Roboflow client using secret API key
api_key = st.secrets["roboflow"]["api_key"]
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display original image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save image to bytes buffer
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        # Inference
        with st.spinner("Detecting..."):
            result = CLIENT.infer(image_bytes, model_id="road-potholes-and-cracks-jmxtp/3")

        # Show JSON result
        st.subheader("🧾 Detection Result")
        st.json(result)

        # Display detections on image if available
        if "image" in result and result["image"].get("url"):
            image_url = result["image"]["url"]
            response = requests.get(image_url)
            if response.status_code == 200:
                annotated_image = Image.open(io.BytesIO(response.content))
                st.image(annotated_image, caption="Detected Objects", use_container_width=True)
            else:
                st.warning("Could not load annotated image from Roboflow.")
        else:
            st.warning("No detections found in the image.")

    except Exception as e:
        st.error(f"Error during detection: {str(e)}")