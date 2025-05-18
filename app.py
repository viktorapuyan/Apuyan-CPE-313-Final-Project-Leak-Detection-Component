import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLOv11 model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Ensure this file is in the same directory
    return model

model = load_model()

# Streamlit UI
st.title("🚨 Pipeline Leak Detection")
st.write("Upload a pipeline image to detect if there's a leak.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    results = model(image)

    # Get class names and confidence
    labels = results[0].names
    detections = results[0].boxes

    if detections is not None and len(detections.cls) > 0:
        leak_detected = False
        for cls_id in detections.cls:
            label = labels[int(cls_id)]
            if label.lower() == "leak":
                leak_detected = True
                break

        if leak_detected:
            st.success("✅ Leak Detected!")
        else:
            st.info("❌ No Leak Detected.")
    else:
        st.info("❌ No objects detected.")
