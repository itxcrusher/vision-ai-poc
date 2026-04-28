# Vision AI POC — People Counter

Real-time people detection and counting using a USB webcam, YOLOv8, and OpenCV.

## Requirements

- Python 3.9+
- USB webcam

## Setup

```
pip install -r requirements.txt
```

The YOLOv8 nano model (`yolov8n.pt`) downloads automatically on first run (~6 MB).

## Run

```
python detect.py
```

If you have multiple cameras and the default (index 0) is wrong, pass the camera index:

```
python detect.py 1
```

## What you see

- Live webcam feed in a window
- Green bounding box around each detected person
- Confidence score above each box
- Live people count in the top-left corner

Press **Q** to quit.
