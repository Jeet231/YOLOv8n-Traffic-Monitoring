"""Portable YOLOv8n + DeepSORT video runner (adapted from the project's original script)."""
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Track vehicles with YOLOv8n and DeepSORT")
    parser.add_argument("--weights", type=Path, default=ROOT / "results/training/baseline_yolov8n_50epochs/weights/best.pt")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tracking/deepsort/deepsort.mp4")
    parser.add_argument("--csv", type=Path, default=ROOT / "results/tracking/deepsort/deepsort.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"Model weights not found: {args.weights}")
    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video: {args.output}")
    model = YOLO(str(args.weights))
    tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)
    try:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
            frame_number = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1
                result = model(frame, conf=args.conf, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for (x1,y1,x2,y2), cls, conf in zip(boxes, classes, confs):
                        detections.append(([float(x1),float(y1),float(x2-x1),float(y2-y1)],float(conf),int(cls)))
                tracks = tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    x1,y1,x2,y2 = track.to_ltrb()
                    tid = track.track_id
                    cls = track.get_det_class()
                    csv_writer.writerow([frame_number, tid, cls, "",float(x1),float(y1),float(x2),float(y2)])
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0),2)
                    cv2.putText(frame, f"ID: {tid}", (int(x1), max(0,int(y1)-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
                writer.write(frame)
    finally:
        cap.release()
        writer.release()
    print(f"Video: {args.output}\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
