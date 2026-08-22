"""Vision AI detector.

Two run modes from one script:

* DEMO (default, GUI): open a camera, draw ONE zone with two clicks, see live
  counts/dwell. For local demos on a laptop.  `python detect.py [camera_index]`

* SERVER (headless): when VISION_API_URL + VISION_DEVICE_TOKEN +
  VISION_CAMERA_ID are set, the detector fetches THIS camera's zones from the
  device-authed endpoint GET /zones?cameraId=..., honors IGNORE zones
  (privacy masking), counts people + dwell per zone, and POSTs the analytics +
  trigger contract to /vision-events. Runs without a display so it can
  live on an appliance / Pi. No GUI calls unless VISION_HEADLESS=0.

Detection is person-only (YOLOv8n + ByteTrack). It does NOT do age/gender
demographics; those claims were removed from the product. What it produces is
people-counting, dwell, and zone occupancy, which is what the dashboard,
analytics, and occupancy/queue triggers consume.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
PERSON_CLASS = 0
CONF_THRESHOLD = 0.4
WIN = "Vision AI POC"  # plain ASCII window title avoids Windows callback issues

# ── Server integration ──────────────────────────────────────────────
# The device token is the same JWT the paired device client obtains when it pairs; the
# camera id is this physical camera's id in the managing account.
SRV_API_URL = os.environ.get("VISION_API_URL", "").rstrip("/")
SRV_DEVICE_TOKEN = os.environ.get("VISION_DEVICE_TOKEN", "")
SRV_CAMERA_ID = os.environ.get("VISION_CAMERA_ID", "")
SRV_POST_INTERVAL = float(os.environ.get("VISION_POST_INTERVAL", "1.0"))  # server throttles ~1/s
SRV_ZONE_REFRESH_SEC = float(os.environ.get("VISION_ZONE_REFRESH_SEC", "60"))
SRV_BUSY_THRESHOLD = int(os.environ.get("VISION_BUSY_THRESHOLD", "3"))    # people in a zone => "busy"
SRV_QUEUE_THRESHOLD = int(os.environ.get("VISION_QUEUE_THRESHOLD", "2"))  # people in a QUEUE zone => queue event
DWELL_LOG_CAP = 500  # bound per-zone dwell history so a 24/7 run can't grow unbounded

SERVER_MODE = bool(SRV_API_URL and SRV_DEVICE_TOKEN and SRV_CAMERA_ID)
HEADLESS = os.environ.get("VISION_HEADLESS", "1" if SERVER_MODE else "0") == "1"

# Server-fetched zones for this camera. Each: {id, name, kind, direction,
# sensitivity, polygon:[{x,y}] normalised 0-1}. Refreshed periodically.
zones = []
_zones_fetched_at = 0.0

# Demo-mode single mouse-drawn zone (pixel rect); only used when not SERVER_MODE.
_corner1 = None
demo_zone = None  # (x1, y1, x2, y2) in pixels

# Per-zone tracking across frames.
zone_members = {}    # {zoneId: {track_id: entry_ts}}
zone_dwell_log = {}  # {zoneId: [completed dwell seconds]}
_last_post = 0.0
_post_fail_streak = 0


def _fetch_zones():
    """Pull this camera's zones from the device-authed endpoint. Best-effort:
    on any failure the previous zones are kept and detection continues."""
    global zones, _zones_fetched_at
    if not SERVER_MODE:
        return
    _zones_fetched_at = time.time()
    url = f"{SRV_API_URL}/zones?cameraId={urllib.parse.quote(SRV_CAMERA_ID)}"
    req = urllib.request.Request(
        url, method="GET",
        headers={"Authorization": "Bearer " + SRV_DEVICE_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        fetched = payload.get("zones") or []
        # Keep only zones with a usable normalised polygon.
        clean = [z for z in fetched if isinstance(z.get("polygon"), list) and len(z["polygon"]) >= 3]
        zones = clean
        print(f"[server] loaded {len(zones)} zone(s) for camera {SRV_CAMERA_ID}")
    except Exception as exc:  # network/auth/parse — never stop detection
        print(f"[server] zone fetch failed (keeping {len(zones)} cached): {exc}")


def _point_in_poly(px, py, polygon):
    """Ray-casting point-in-polygon for a normalised (0-1) polygon [{x,y}]."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].get("x", 0.0), polygon[i].get("y", 0.0)
        xj, yj = polygon[j].get("x", 0.0), polygon[j].get("y", 0.0)
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _active_zones(w, h):
    """The zones to evaluate this frame: the server-fetched zones, or (demo mode)
    the single mouse-drawn rectangle expressed as a normalised COUNTING zone."""
    if SERVER_MODE:
        return zones
    if demo_zone is None:
        return []
    x1, y1, x2, y2 = demo_zone
    return [{
        "id": "demo",
        "name": "Demo zone",
        "kind": "COUNTING",
        "direction": None,
        "sensitivity": 50,
        "polygon": [
            {"x": x1 / w, "y": y1 / h}, {"x": x2 / w, "y": y1 / h},
            {"x": x2 / w, "y": y2 / h}, {"x": x1 / w, "y": y2 / h},
        ],
    }]


def _post_vision_event(detections, zone_events):
    """POST the trigger + analytics contract while there is activity. Throttled;
    never raises into the loop. Includes cameraId + zoneEvents so the dashboard,
    heatmaps, and reports populate (not just triggers)."""
    global _last_post, _post_fail_streak
    if not SERVER_MODE or not detections:
        return
    now = time.time()
    # Back off when the API is failing so a dead endpoint can't hammer the loop.
    interval = SRV_POST_INTERVAL * (1 + min(_post_fail_streak, 5))
    if now - _last_post < interval:
        return
    _last_post = now
    body = json.dumps({
        "cameraId": SRV_CAMERA_ID,
        "detections": detections,
        "zoneEvents": zone_events,
    }).encode("utf-8")
    req = urllib.request.Request(
        SRV_API_URL + "/vision-events", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + SRV_DEVICE_TOKEN},
    )
    try:
        urllib.request.urlopen(req, timeout=3).close()
        _post_fail_streak = 0
    except urllib.error.HTTPError as exc:
        _post_fail_streak += 1
        # 401 => the device token expired; a production client should refresh it
        # via POST /token/refresh. Logged here so it is visible in ops.
        print(f"[server] vision-event POST {exc.code}: {exc.reason}")
    except Exception as exc:
        _post_fail_streak += 1
        print(f"[server] vision-event POST failed: {exc}")


def _mouse(event, x, y, flags, param):
    global _corner1, demo_zone
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if _corner1 is None:
        _corner1 = (x, y)
    else:
        x1, y1 = min(_corner1[0], x), min(_corner1[1], y)
        x2, y2 = max(_corner1[0], x), max(_corner1[1], y)
        if x2 - x1 > 10 and y2 - y1 > 10:
            demo_zone = (x1, y1, x2, y2)
        _corner1 = None
        zone_members.clear()
        zone_dwell_log.clear()


def _resolve_source():
    # USB index (`python detect.py 1`) or a stream URL (rtsp://.., http://..),
    # also accepted via VISION_CAMERA_SOURCE so it can run headless on an appliance.
    arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VISION_CAMERA_SOURCE", "")
    if not arg:
        return 0
    return int(arg) if str(arg).isdigit() else arg


def _open_capture(source):
    if isinstance(source, str) and source.lower().startswith("rtsp"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    return cv2.VideoCapture(source)


def main():
    source = _resolve_source()
    model = YOLO(MODEL_NAME)

    cap = _open_capture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera/stream {source!r}")
        sys.exit(1)

    if SERVER_MODE:
        print(f"[server] server mode, camera {SRV_CAMERA_ID}, headless={HEADLESS}")
        _fetch_zones()
    else:
        print("Demo mode. Click corner 1, click corner 2 to draw a zone. R = reset, Q = quit.")

    if not HEADLESS:
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, _mouse)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    print("Stream read failed; reconnecting in 3s...")
                    cap.release()
                    time.sleep(3)
                    cap = _open_capture(source)
                    continue
                print("ERROR: Failed to read frame")
                break

            now = time.time()
            h, w = frame.shape[:2]

            # Periodically refresh zones so dashboard edits take effect live.
            if SERVER_MODE and now - _zones_fetched_at >= SRV_ZONE_REFRESH_SEC:
                _fetch_zones()

            active = _active_zones(w, h)
            ignore_polys = [z["polygon"] for z in active if z.get("kind") == "IGNORE"]
            counted = [z for z in active if z.get("kind") != "IGNORE"]

            results = model.track(
                frame, classes=[PERSON_CLASS], conf=CONF_THRESHOLD,
                persist=True, verbose=False,
            )

            detected_ids = set()
            # zoneId -> set of track ids inside this frame
            inside_now = {z["id"]: set() for z in counted}

            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue
                for box, tid in zip(result.boxes, result.boxes.id.int().tolist()):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    nx, ny = cx / w, cy / h

                    # PRIVACY: drop any person whose centroid is in an IGNORE zone
                    # BEFORE counting/posting/drawing. Sensitive areas stay unseen.
                    if any(_point_in_poly(nx, ny, poly) for poly in ignore_polys):
                        continue

                    detected_ids.add(tid)
                    in_any = False
                    for z in counted:
                        if _point_in_poly(nx, ny, z["polygon"]):
                            in_any = True
                            inside_now[z["id"]].add(tid)
                            members = zone_members.setdefault(z["id"], {})
                            if tid not in members:
                                members[tid] = now

                    if not HEADLESS:
                        color = (30, 140, 255) if in_any else (0, 200, 80)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame, (cx, cy), 4, color, -1)
                        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Close out dwell for anyone who left a zone (or left frame) and build
            # the per-zone analytics + the catalog detection types for triggers.
            detection_types = set()
            zone_events = []
            for z in counted:
                zid = z["id"]
                members = zone_members.setdefault(zid, {})
                here = inside_now.get(zid, set())
                for tid in list(members):
                    if tid not in here:
                        dur = now - members.pop(tid)
                        log = zone_dwell_log.setdefault(zid, [])
                        log.append(dur)
                        if len(log) > DWELL_LOG_CAP:
                            del log[: len(log) - DWELL_LOG_CAP]

                count = len(here)
                dwells = [now - t for t in members.values()]
                avg_dwell = round(sum(dwells) / len(dwells), 1) if dwells else 0.0
                zone_events.append({"zoneId": zid, "peopleInZone": count, "dwellTimeSeconds": avg_dwell})

                # Occupancy detection types the trigger catalog understands.
                if count >= SRV_BUSY_THRESHOLD:
                    detection_types.add("occupancy_busy")
                elif count > 0:
                    detection_types.add("occupancy_quiet")
                if z.get("kind") == "QUEUE" and count >= SRV_QUEUE_THRESHOLD:
                    detection_types.add("behaviour_queue")

            detections = [{"type": t, "confidence": 0.9} for t in sorted(detection_types)]
            # Only post when there is activity (a non-empty counted zone).
            if any(ev["peopleInZone"] > 0 for ev in zone_events):
                _post_vision_event(detections, zone_events)

            if not HEADLESS:
                for z in counted:
                    poly_px = [(int(p["x"] * w), int(p["y"] * h)) for p in z["polygon"]]
                    import numpy as np  # local import: only needed for the GUI overlay
                    pts = np.array(poly_px, dtype=np.int32)
                    cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
                    cv2.putText(frame, z.get("name", "zone"), (poly_px[0][0] + 4, poly_px[0][1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
                for z in active:
                    if z.get("kind") == "IGNORE":
                        poly_px = [(int(p["x"] * w), int(p["y"] * h)) for p in z["polygon"]]
                        import numpy as np
                        pts = np.array(poly_px, dtype=np.int32)
                        cv2.polylines(frame, [pts], True, (60, 60, 200), 2)
                        cv2.putText(frame, "IGNORE", (poly_px[0][0] + 4, poly_px[0][1] + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 200), 1)
                total_in = sum(ev["peopleInZone"] for ev in zone_events)
                cv2.rectangle(frame, (0, 0), (240, 30), (0, 0, 0), -1)
                cv2.putText(frame, f"In zones: {total_in}", (10, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
                if _corner1:
                    cv2.drawMarker(frame, _corner1, (255, 200, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.imshow(WIN, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    zone_members.clear()
                    zone_dwell_log.clear()
                    globals()["demo_zone"] = None
    except KeyboardInterrupt:
        print("Interrupted; shutting down.")
    finally:
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
