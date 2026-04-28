# Vision AI POC — Zone Analytics

Real-time people detection, zone entry counting, and dwell time tracking using a USB webcam, YOLOv8, and OpenCV.

## Requirements

- Python 3.9+
- USB or built-in webcam

## Setup

```bash
pip install -r requirements.txt
```

The YOLOv8 nano model (`yolov8n.pt`) downloads automatically on first run (~6 MB).

## Run

```bash
python detect.py
```

Pass a camera index if the default (0) is wrong:

```bash
python detect.py 1
```

## How to use

1. The webcam feed opens in a window
2. **Click and drag** on the video to draw a zone rectangle
3. The stats panel (top-left) updates in real time:
   - **In zone** — people currently inside the zone
   - **Total entries** — cumulative count of zone entries this session
   - **Avg dwell** — average time people spend inside the zone (seconds)
4. Each tracked person gets a persistent ID:
   - **Green box** = outside zone
   - **Blue-orange box** = inside zone, with live dwell time shown
5. Press **R** to reset the zone and all stats
6. Press **Q** to quit

## Notes

- Works on Windows, macOS, and Linux — any OS-recognised camera (built-in or USB)
- Person tracking uses YOLOv8 + ByteTrack for persistent IDs across frames
- Dwell time is recorded when a person leaves the zone or exits the frame
