import sys
import time
import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
PERSON_CLASS = 0
CONF_THRESHOLD = 0.4

# Zone state (set by mouse drag)
zone = None
_drawing = False
_drag_start = None

# Per-session tracking state
in_zone = {}    # track_id -> entry_time (only people currently inside zone)
dwell_log = []  # completed dwell durations in seconds
total_entries = 0


def _mouse(event, x, y, flags, param):
    global zone, _drawing, _drag_start
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _drag_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        zone = (_min2(_drag_start[0], x), _min2(_drag_start[1], y),
                max(_drag_start[0], x), max(_drag_start[1], y))
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        zone = (_min2(_drag_start[0], x), _min2(_drag_start[1], y),
                max(_drag_start[0], x), max(_drag_start[1], y))


def _min2(a, b):
    return a if a < b else b


def _inside(cx, cy):
    if zone is None:
        return False
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def _reset():
    global zone, total_entries
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

    win = "Vision AI POC — Zone Analytics"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _mouse)

    print("Camera opened.")
    print("  Click and drag on the video to draw a zone.")
    print("  R = reset zone and stats   Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        now = time.time()

        # model.track gives persistent IDs across frames (ByteTrack)
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
                    color = (30, 140, 255)          # blue-orange: in zone
                    label = f"ID:{tid}  {dwell:.1f}s"
                else:
                    if tid in in_zone:
                        dwell_log.append(now - in_zone.pop(tid))
                    color = (0, 200, 80)            # green: outside zone
                    label = f"ID:{tid}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # People who disappeared from frame while still tracked in zone
        for tid in set(in_zone) - detected_ids:
            dwell_log.append(now - in_zone.pop(tid))

        # Draw zone rectangle with semi-transparent fill
        if zone:
            overlay = frame.copy()
            cv2.rectangle(overlay, (zone[0], zone[1]), (zone[2], zone[3]),
                          (255, 200, 0), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]),
                          (255, 200, 0), 2)
            cv2.putText(frame, "ZONE", (zone[0] + 6, zone[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        # Stats panel (top-left)
        in_zone_now = len([t for t in in_zone if t in detected_ids])
        avg_dwell = sum(dwell_log) / len(dwell_log) if dwell_log else 0.0

        lines = [
            f"In zone:       {in_zone_now}",
            f"Total entries: {total_entries}",
            f"Avg dwell:     {avg_dwell:.1f}s",
        ]
        panel_h = 16 + 28 * len(lines)
        cv2.rectangle(frame, (0, 0), (260, panel_h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 20 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 100), 2)

        # Hint when no zone drawn
        if zone is None:
            cv2.putText(frame, "Click and drag to draw a zone",
                        (10, frame.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            _reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
