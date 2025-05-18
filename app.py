import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load the ONNX model
session = ort.InferenceSession("best (1).onnx")

# Define preprocessing function
def preprocess_image(image: Image.Image, target_size=(640, 640)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    img_array = np.transpose(img_array, (2, 0, 1))  # (HWC → CHW)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

# Streamlit UI
st.title("Pipeline Leak Detection (Object Detection)")
st.write("Upload an image to detect pipeline leaks.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_data = preprocess_image(image)

    # Get model input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference 
    outputs = session.run([output_name], {input_name: input_data})
    detections = outputs[0]  # shape: (1, 8, 8400)

    # Remove batch dimension
    detections = np.squeeze(detections)  # shape: (8, 8400)

    # Transpose to (8400, 8) so each row is a detection
    detections = detections.T

    # Confidence threshold
    conf_threshold = 0.25  # You can lower this if your model is under-confident

    # Objectness * class_1 score (assuming class 1 is "leak")
    objectness = detections[:, 4]
    class_1_score = detections[:, 6]  # Index 6 = second class ("leak")
    final_conf = objectness * class_1_score

    # Find detections above threshold
    leak_detections = detections[final_conf > conf_threshold]

if len(leak_detections) > 0:
    st.success("Prediction: Leak Detected")
else:
    st.info("Prediction: No Leak Detected")
