"""Portable YOLOv8n + ByteTrack video runner (adapted from the project's original script)."""
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Track vehicles with YOLOv8n and ByteTrack")
    parser.add_argument("--weights", type=Path, default=ROOT / "results/training/baseline_yolov8n_50epochs/weights/best.pt")
    parser.add_argument("--video", type=Path, required=True, help="Path to a local video")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tracking/bytetrack/bytetrack.mp4")
    parser.add_argument("--csv", type=Path, default=ROOT / "results/tracking/bytetrack/bytetrack.csv")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config or local YAML path")
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
                result = model.track(frame, tracker=args.tracker, persist=True, conf=args.conf, verbose=False)[0]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    for box, tid, cls, conf in zip(boxes, ids, classes, confidences):
                        csv_writer.writerow([frame_number, int(tid), int(cls), float(conf), *[float(v) for v in box]])
                writer.write(result.plot())
    finally:
        cap.release()
        writer.release()
    print(f"Video: {args.output}\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
