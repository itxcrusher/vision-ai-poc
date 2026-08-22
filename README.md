# Vision AI — Zone Analytics

Real-time people detection, per-zone counting, and dwell-time tracking with YOLOv8 + ByteTrack + OpenCV. Person-only detection (no age/gender demographics). Runs either as a local GUI demo or as a headless, server-configured client for a managed camera.

## Requirements

- Python 3.9+
- A USB/built-in webcam, or an RTSP/HTTP camera stream

## Setup

```bash
pip install -r requirements.txt
```

The YOLOv8 nano model (`yolov8n.pt`) downloads automatically on first run (~6 MB).

## Demo mode (local GUI)

```bash
python detect.py            # default camera (index 0)
python detect.py 1          # a specific camera index
```

- Click corner 1, then click corner 2 to draw a zone.
- The overlay shows each tracked person (persistent ID) and the live in-zone count.
- Press **R** to reset the zone, **Q** to quit.

## Server mode (headless, production-shaped)

When these env vars are set, the detector runs headless, pulls THIS camera's zones from the server, honors IGNORE zones, and posts the analytics + trigger contract.

| Variable | Purpose |
|---|---|
| `VISION_API_URL` | API base, e.g. `https://api.example.com` |
| `VISION_DEVICE_TOKEN` | the device JWT (same token the player gets when it pairs) |
| `VISION_CAMERA_ID` | this camera's id in the managing account |
| `VISION_CAMERA_SOURCE` | camera index or RTSP/HTTP URL (or pass as argv) |
| `VISION_HEADLESS` | `1` (default in server mode) runs with no window; `0` shows the overlay |
| `VISION_BUSY_THRESHOLD` | people in a zone to emit `occupancy_busy` (default 3) |
| `VISION_QUEUE_THRESHOLD` | people in a QUEUE zone to emit `behaviour_queue` (default 2) |
| `VISION_ZONE_REFRESH_SEC` | how often to re-fetch zones (default 60) |

```bash
VISION_API_URL=https://api.example.com \
VISION_DEVICE_TOKEN=... \
VISION_CAMERA_ID=... \
VISION_CAMERA_SOURCE="rtsp://user:pass@host/stream" \
python detect.py
```

What it does in server mode:

- **Fetches zones** from `GET /zones?cameraId=...` (device-authed), refreshed periodically.
- **IGNORE zones (privacy):** any person whose centroid falls inside a `kind=IGNORE` zone is dropped before counting, posting, or drawing, so sensitive areas (restrooms, back-of-house) are never analysed. This honors the customer consent/notice templates' masking promise.
- **Per-zone counting + dwell** via point-in-polygon against the normalised (0-1) zone polygons.
- **Posts** to `POST /vision-events` with `cameraId`, catalog `detections` (`occupancy_busy` / `occupancy_quiet` / `behaviour_queue`) that drive triggers, and `zoneEvents[]` (`zoneId`, `peopleInZone`, `dwellTimeSeconds`) that populate the dashboard, heatmaps, and reports. Throttled (~1/s) with backoff on failure.

## Notes

- Person tracking is YOLOv8 + ByteTrack (persistent IDs across frames).
- Detection is person-only; there is no age/gender/demographic model.
- RTSP uses TCP transport and auto-reconnects, so a 24/7 store stream self-heals.
- Not yet wired (production hardening): device-token refresh on a 401 (`POST /token/refresh`), a managed service wrapper (systemd) + a hardware accelerator (Hailo/Coral/Jetson) for multi-camera 24/7 inference.
