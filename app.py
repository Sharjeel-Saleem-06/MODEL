import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import requests

# GitHub Raw URL for the model (Replace with your GitHub link)
MODEL_URL = "https://github.com/Sharjeel-Saleem-06/MODEL/blob/main/LTV_HTV.pt"
MODEL_PATH = "LTV_HTV.pt"

# Function to download model from GitHub if not available locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from GitHub...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            raise RuntimeError(f"❌ Failed to download model. HTTP Status: {response.status_code}")

# Load YOLO model
def load_model():
    download_model()
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model: {e}")

model = load_model()

# Function to perform inference on an image
def predict(image):
    try:
        img_array = np.array(image)
        results = model(img_array)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = model.names[int(box.cls[0])]
                if confidence < 0.7:
                    continue
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(img_array, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        result_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        return result_image
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

# Flag to control camera feed
stop_camera = False

# Function to handle live camera feed
def process_camera_feed():
    global stop_camera
    stop_camera = False
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("⚠️ Camera not working!")
    
    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = model.names[int(box.cls[0])]
                if confidence < 0.7:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb
    
    cap.release()

# Function to stop the camera
def stop_camera_feed():
    global stop_camera
    stop_camera = True

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Autopilot Pro")
    
    gr.Markdown("### Upload an image or use your camera for real-time object detection.")
    
    with gr.Tab("📷 Upload Image"):
        upload_box = gr.Image(label="Upload Image", type="pil")
        output_image = gr.Image(label="Detection Result", type="pil")
        upload_box.change(predict, inputs=upload_box, outputs=output_image)
    
    with gr.Tab("📹 Live Camera"):
        camera_output = gr.Image(label="Live Detection", streaming=True)
        start_button = gr.Button("Start Camera")
        stop_button = gr.Button("Stop Camera")
        
        start_button.click(process_camera_feed, outputs=camera_output, show_progress=False)
        stop_button.click(stop_camera_feed)

# Launch the app
demo.launch()
