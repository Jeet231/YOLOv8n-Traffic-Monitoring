import cv2
import csv

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


MODEL_PATH = r"C:\Clg\VehicleDetection\Traffic_Research\results\training\baseline_yolov8n_50epochs\weights\best.pt"

VIDEO_PATH = r"C:\Clg\VehicleDetection\Traffic_Research\datasets\GRAM\Videos\video_train\test_1.mov"

OUTPUT_VIDEO = r"C:\Clg\VehicleDetection\Traffic_Research\results\tracking\deepsort\videos\deepsort.mp4"

OUTPUT_CSV = r"C:\Clg\VehicleDetection\Traffic_Research\results\tracking\deepsort\labels\deepsort.csv"


model = YOLO(MODEL_PATH)

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
)

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

        # YOLO detection only
        results = model(
            frame,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(
                boxes,
                classes,
                confidences
            ):

                x1, y1, x2, y2 = box

                width_box = x2 - x1
                height_box = y2 - y1

                detections.append(
                    (
                        [x1, y1, width_box, height_box],
                        float(conf),
                        int(cls)
                    )
                )

        # DeepSORT
        tracks = tracker.update_tracks(
            detections,
            frame=frame
        )

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = ltrb

            # Class information
            class_id = track.get_det_class()

            csv_writer.writerow([
                frame_number,
                int(track_id),
                class_id,
                "",
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ])

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        writer.write(frame)

        cv2.imshow("DeepSORT", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("DeepSORT completed.")