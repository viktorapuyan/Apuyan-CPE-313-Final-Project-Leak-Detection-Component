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
    img_array = np.transpose(img_array, (2, 0, 1))  # Convert to (C, H, W)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

# Streamlit UI
st.title("Pipeline Leak Detection")
st.write("Upload an image to determine if there is a pipeline leak.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_data = preprocess_image(image)

    # Get input & output names for the ONNX model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    result = outputs[0]

    # Print debug info
    st.write("Model Output:", result)
    st.write("Output Shape:", result.shape)

    # Handle output types
    if result.ndim == 2 and result.shape[1] == 2:
        # e.g., [[0.1, 0.9]] → Softmax or logits
        prediction = int(np.argmax(result, axis=1)[0])
    elif result.ndim == 2 and result.shape[1] == 1:
        # e.g., [[0.87]] → sigmoid output
        prediction = int(result[0][0] > 0.5)
    else:
        st.error("Unexpected model output shape. Check your model's output layer.")
        st.stop()

# Display result
label = "Leak" if prediction == 1 else "No Leak"
st.subheader(f"Prediction: {label}")

