import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load YOLO ONNX model
@st.cache_resource
def load_model():
    return ort.InferenceSession("best.onnx")

session = load_model()

# Image preprocessing
def preprocess_image(image: Image.Image, target_size=(640, 640)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    img_array = np.transpose(img_array, (2, 0, 1))           # (HWC → CHW)
    img_array = np.expand_dims(img_array, axis=0)            # Add batch dim
    return img_array

# Leak detection function
def detect_leak(image_array, session, threshold=0.50, leak_class_index=1):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Inference
    outputs = session.run([output_name], {input_name: image_array})
    detections = outputs[0]  # shape: (1, 8, 8400)

    # Postprocess
    detections = np.squeeze(detections).T  # shape: (8400, 8)
    objectness = detections[:, 4]
    class_scores = detections[:, 5 + leak_class_index]  # index 6 if leak is class 1
    final_conf = objectness * class_scores

    # Determine if any leak predictions exist above threshold
    leak_found = np.any(final_conf > threshold)
    return leak_found, np.max(final_conf)

# Streamlit UI
st.title("🚨 Pipeline Leak Detection")
st.write("Upload a pipeline image to determine if a leak is present.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    image_array = preprocess_image(image)
    leak_found, confidence = detect_leak(image_array, session)

    # Display results
    if leak_found:
        st.success(f"✅ Leak Detected (Confidence: {confidence:.2f})")
    else:
        st.info(f"❌ No Leak Detected (Max Confidence: {confidence:.2f})")
