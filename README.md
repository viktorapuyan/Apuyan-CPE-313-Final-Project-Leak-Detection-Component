# Apuyan - CPE 313 Final Project: Pipeline Leak Detection Component

## 📌 Project Overview

This project is a **deep learning-based pipeline leak detection system** developed as a final project for CPE 313. It uses a fine-tuned **YOLOv11** (You Only Look Once) object detection model to classify pipeline images as either containing a **leak** or **no leak**. The application is deployed as an interactive web app using **Streamlit**, making it easy for users to upload pipeline images and receive real-time detection results.

---

## 🚀 Live Demo

Try the deployed app here:  
🔗 [https://apuyan-cpe-313-final-project-leak-detection-component-ce9esgu3.streamlit.app/](https://apuyan-cpe-313-final-project-leak-detection-component-ce9esgu3.streamlit.app/)

---

## 🧠 How It Works

1. The user uploads a pipeline image (JPG, JPEG, or PNG) through the web interface.
2. The image is passed to a **YOLOv11** model (`best.pt`) trained to detect leaks in pipeline images.
3. The model performs inference and outputs detection results including class labels and confidence scores.
4. The app displays whether a **Leak** or **No Leak** was detected based on the model's predictions.

---

## ✨ Features

- 📷 Upload pipeline images directly through the browser
- 🤖 Real-time leak detection powered by YOLOv11
- ✅ Clear visual feedback: "Leak Detected!" or "No Leak Detected"
- ⚡ Fast inference using a pre-trained and fine-tuned YOLO model
- 🌐 Accessible via a public Streamlit web app

---

## 🛠️ Tech Stack

| Component        | Technology                        |
|------------------|-----------------------------------|
| Web Framework    | [Streamlit](https://streamlit.io) |
| Object Detection | [Ultralytics YOLOv11](https://docs.ultralytics.com) |
| Image Processing | [Pillow (PIL)](https://pillow.readthedocs.io), [OpenCV](https://opencv.org) |
| Numerical Computing | [NumPy](https://numpy.org)     |
| Model Runtime    | [ONNX Runtime](https://onnxruntime.ai) |

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit application
├── best.pt                 # Trained YOLOv11 model weights
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level packages
├── images/
│   ├── leak/               # Sample images with leaks
│   └── no leak/            # Sample images without leaks
└── Adornado Apuyan - Deep Learning-based leak detection PPT.pdf
```

---

## ⚙️ Running Locally

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/viktorapuyan/Apuyan-CPE-313-Final-Project-Leak-Detection-Component.git
   cd Apuyan-CPE-313-Final-Project-Leak-Detection-Component
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`.

---

## 🖼️ Sample Images

Sample pipeline images for testing are available in the `images/` folder:
- `images/leak/` — Images showing pipeline leaks
- `images/no leak/` — Images showing pipelines without leaks

---

## 👤 Author

**Viktor Adornado Apuyan**  
CPE 313 — Final Project  
Deep Learning-Based Pipeline Leak Detection
