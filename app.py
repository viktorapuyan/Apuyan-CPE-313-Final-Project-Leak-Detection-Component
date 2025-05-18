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

    # YOLO-style: filter detections by confidence threshold
    confidence_threshold = 0.5
    detections = np.squeeze(detections)  # shape becomes (8, 8400)

    # Assuming class confidence is in 5th position (index 4 or 5)
    scores = detections[4]  # Adjust if needed
    high_conf_indices = np.where(scores > confidence_threshold)[0]

    if len(high_conf_indices) > 0:
        st.success("Prediction: Leak Detected")
    else:
        st.info("Prediction: No Leak Detected")
