# Posture-AI

An AI-powered real-time posture detection system that analyzes human body alignment using computer vision and pose estimation.

---

## Overview

Posture-AI detects and evaluates human posture in real time using YOLO-based pose estimation and custom geometric analysis.
It is designed to identify incorrect posture patterns (like slouching or neck misalignment) with high efficiency and minimal latency.

---

## Key Features

* Real-time posture detection using webcam/video feed
* YOLOv8-based human pose estimation
* Geometric analysis of neck and spine alignment
* Detects incorrect posture and misalignment
* Logging system for posture data
* Smooth performance (~25 FPS)

---

## What Makes This Project Strong

* Developed a **real-time posture detection system** using YOLO & OpenCV
* Designed **geometric algorithms** for neck & spine alignment-based classification
* Implemented **variance-based filtering** to reduce noise and false positives
* Achieved **~25 FPS real-time performance** for reliable detection

---

## Tech Stack

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* PyTorch

---

## Project Structure

```
Posture-AI/
│── src/                 # Core logic and processing
│── README.md
│── .gitignore
```

---

## Model Weights (Important)

Due to GitHub file size limits, model weights are not included.

Download YOLOv8 weights from:
https://github.com/ultralytics/ultralytics

Place in root directory:

* `yolov8n.pt`
* `yolov8n-pose.pt`

---

## Running the Project

```bash
python src/main.py
```


## Performance

* ~25 FPS real-time inference
* Optimized for low-latency detection
* Reduced false positives using filtering techniques

---

## Future Improvements

* Deploy as a web/mobile app
* Add posture correction suggestions
* Advanced analytics dashboard
* Multi-person posture tracking

---
