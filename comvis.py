import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO

model_path = "best.pt"
model = YOLO(model_path)


st.title("Implementasi Deteksi Sendal dan Sepatu Hilang di Masjid")
st.write("Model di download dari https://universe.roboflow.com/fadhil-1aylr/sendal-gw-hilang/dataset/2# by fadhillah rahmad kurnia")
st.write("Model ini menggunakan YOLOv12 untuk mendeteksi sandal dan sepatu yang hilang di masjid.")

frame_window = st.image([])


def detect_and_display():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam tidak bisa diakses.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model(frame)[0] 

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]

                if conf > 0.3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

    cap.release()
    cv2.destroyAllWindows()

if st.button("Mulai Deteksi"):
    detect_and_display()
