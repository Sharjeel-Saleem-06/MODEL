import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO Model
@st.cache_resource
def load_model():
    model_path = r"E:\MODEL\LTV_HTV\last.pt"
    model = YOLO(model_path)  # Load YOLOv8 model
    return model

model = load_model()

# Streamlit UI
st.title("🚀 YOLO Object Detection App")
st.sidebar.title("🔍 Upload Image for Detection")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert image for model
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Run YOLO model
    results = model(img_array)

    # Display the image with detections
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Draw bounding boxes
    for result in results:
        boxes = result.boxes  # Detected objects
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box

    # Show results
    st.image(img_array, caption="🎯 Detection Result", use_column_width=True)
    st.write("📌 Detected Objects:", [model.names[int(cls)] for cls in result.boxes.cls])
