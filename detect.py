import sys
import time
import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
PERSON_CLASS = 0
CONF_THRESHOLD = 0.4
WIN = "Vision AI POC"          # plain ASCII — avoids Windows mouse-callback issues

# Zone state: two-click drawing (click corner 1, click corner 2)
_corner1 = None                # first click position
zone = None                    # final zone: (x1, y1, x2, y2)

# Per-session tracking
in_zone = {}    # track_id -> entry_time
dwell_log = []  # completed dwell durations in seconds
total_entries = 0


def _mouse(event, x, y, flags, param):
    global _corner1, zone, total_entries
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if _corner1 is None:
        _corner1 = (x, y)
    else:
        x1 = min(_corner1[0], x)
        y1 = min(_corner1[1], y)
        x2 = max(_corner1[0], x)
        y2 = max(_corner1[1], y)
        if x2 - x1 > 10 and y2 - y1 > 10:   # ignore accidental double-clicks
            zone = (x1, y1, x2, y2)
        _corner1 = None
        in_zone.clear()
        dwell_log.clear()
        total_entries = 0


def _inside(cx, cy):
    if zone is None:
        return False
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def _reset():
    global _corner1, zone, total_entries
    _corner1 = None
    zone = None
    in_zone.clear()
    dwell_log.clear()
    total_entries = 0


def main():
    global total_entries

    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        sys.exit(1)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, _mouse)

    print("Camera opened.")
    print("  Click once = set corner 1,  click again = set corner 2  (draws the zone)")
    print("  R = reset zone and stats    Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        now = time.time()
        h, w = frame.shape[:2]

        results = model.track(
            frame,
            classes=[PERSON_CLASS],
            conf=CONF_THRESHOLD,
            persist=True,
            verbose=False,
        )

        detected_ids = set()

        for result in results:
            if result.boxes.id is None:
                continue
            for box, tid in zip(result.boxes, result.boxes.id.int().tolist()):
                detected_ids.add(tid)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                inside = _inside(cx, cy)

                if inside:
                    if tid not in in_zone:
                        in_zone[tid] = now
                        total_entries += 1
                    dwell = now - in_zone[tid]
                    color = (30, 140, 255)
                    label = f"ID:{tid}  {dwell:.1f}s"
                else:
                    if tid in in_zone:
                        dwell_log.append(now - in_zone.pop(tid))
                    color = (0, 200, 80)
                    label = f"ID:{tid}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # People who left frame while still tracked inside zone
        for tid in set(in_zone) - detected_ids:
            dwell_log.append(now - in_zone.pop(tid))

        # Draw confirmed zone
        if zone:
            overlay = frame.copy()
            cv2.rectangle(overlay, (zone[0], zone[1]), (zone[2], zone[3]),
                          (255, 200, 0), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]),
                          (255, 200, 0), 2)
            cv2.putText(frame, "ZONE", (zone[0] + 6, zone[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        # Show first-click crosshair marker
        if _corner1:
            cv2.drawMarker(frame, _corner1, (255, 200, 0),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, "Corner 1 set - click corner 2",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 200, 0), 1)

        # Stats panel
        in_zone_now = len([t for t in in_zone if t in detected_ids])
        avg_dwell = sum(dwell_log) / len(dwell_log) if dwell_log else 0.0

        lines = [
            f"In zone:       {in_zone_now}",
            f"Total entries: {total_entries}",
            f"Avg dwell:     {avg_dwell:.1f}s",
        ]
        cv2.rectangle(frame, (0, 0), (260, 16 + 28 * len(lines)), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 20 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 100), 2)

        # Hint
        if zone is None and _corner1 is None:
            cv2.putText(frame, "Click to set zone corner 1",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (180, 180, 180), 1)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            _reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
