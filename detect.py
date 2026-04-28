import sys
import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"  # nano — fast, downloads automatically on first run
PERSON_CLASS = 0            # COCO class 0 = person
CONF_THRESHOLD = 0.4


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        sys.exit(1)

    print(f"Camera {camera_index} opened. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        results = model(frame, classes=[PERSON_CLASS], conf=CONF_THRESHOLD, verbose=False)

        count = 0
        for result in results:
            for box in result.boxes:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)
                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 80),
                    1,
                )

        label = f"People: {count}"
        cv2.rectangle(frame, (0, 0), (200, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)

        cv2.imshow("Vision AI POC — People Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
