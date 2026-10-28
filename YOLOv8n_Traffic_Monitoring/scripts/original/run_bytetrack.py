import cv2
import csv
from ultralytics import YOLO

MODEL_PATH = r"C:\Clg\VehicleDetection\Traffic_Research\results\training\baseline_yolov8n_50epochs\weights\best.pt"
VIDEO_PATH = r"C:\Clg\VehicleDetection\Traffic_Research\datasets\GRAM\Videos\video_train\test_1.mov"

OUTPUT_VIDEO = r"C:\Clg\VehicleDetection\Traffic_Research\results\tracking\bytetrack\videos\bytetrack.mp4"
OUTPUT_CSV = r"C:\Clg\VehicleDetection\Traffic_Research\results\tracking\bytetrack\labels\bytetrack.csv"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

with open(OUTPUT_CSV, "w", newline="") as f:

    csv_writer = csv.writer(f)

    csv_writer.writerow([
        "frame",
        "track_id",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ])

    frame_number = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(
                boxes, ids, classes, confidences
            ):

                x1, y1, x2, y2 = box

                csv_writer.writerow([
                    frame_number,
                    int(track_id),
                    int(cls),
                    float(conf),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ])

        annotated = result.plot()

        writer.write(annotated)

        cv2.imshow("ByteTrack", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("ByteTrack completed.")