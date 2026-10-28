
from pathlib import Path
import csv
import json
import time
from collections import Counter

import cv2
from ultralytics import YOLO

PROJECT = Path(r"C:\Clg\VehicleDetection\Traffic_Research")
VIDEO = r"C:\Clg\VehicleDetection\Traffic_Research\datasets\GRAM\Videos\video_train\test_1.mov"
WEIGHTS = PROJECT / "results" / "training" / "baseline_yolov8n_50epochs" / "weights" / "best.pt"
OUT = PROJECT / "results" / "tracking" / "ocsort_metrics"
OUT.mkdir(parents=True, exist_ok=True)

FRAME_CSV = OUT / "frame_metrics.csv"
SUMMARY_JSON = OUT / "summary_metrics.json"

model = YOLO(str(WEIGHTS))

frame_rows = []
class_counts = Counter()
all_track_ids = set()
frames_processed = 0
frames_with_tracks = 0
detection_total = 0
inference_ms = []
start = time.perf_counter()

results = model.track(
    source=str(VIDEO),
    tracker="ocsort.yaml",
    stream=True,
    persist=True,
    imgsz=480,
    device=0,
    conf=0.25,
    iou=0.5,
    vid_stride=2,
    save=True,
    project=str(OUT),
    name="annotated_video",
    exist_ok=True,
    verbose=False
)

with FRAME_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sampled_frame", "detections", "tracked_detections",
            "track_ids", "classes", "inference_ms"
        ]
    )
    writer.writeheader()

    for frame_no, result in enumerate(results, start=1):
        frames_processed += 1
        boxes = result.boxes
        count = len(boxes) if boxes is not None else 0
        detection_total += count

        ids_this_frame = []
        classes_this_frame = Counter()
        tracked_count = 0

        if boxes is not None and count:
            class_ids = boxes.cls.int().cpu().tolist()
            names = result.names

            for class_id in class_ids:
                label = names.get(class_id, str(class_id))
                classes_this_frame[label] += 1
                class_counts[label] += 1

            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                ids_this_frame = ids
                tracked_count = len(ids)
                all_track_ids.update(ids)
                frames_with_tracks += 1

        infer = result.speed.get("inference", 0.0)
        inference_ms.append(float(infer))

        writer.writerow({
            "sampled_frame": frame_no,
            "detections": count,
            "tracked_detections": tracked_count,
            "track_ids": ";".join(map(str, ids_this_frame)),
            "classes": json.dumps(dict(classes_this_frame)),
            "inference_ms": round(float(infer), 3)
        })

        if frame_no % 500 == 0:
            print(f"Processed {frame_no} sampled frames")

elapsed = time.perf_counter() - start
avg_inference = (
    sum(inference_ms) / len(inference_ms) if inference_ms else 0.0
)

summary = {
    "video": str(VIDEO),
    "tracker": "ocsort.yaml",
    "weights": str(WEIGHTS),
    "frames_processed_sampled": frames_processed,
    "video_frame_stride": 2,
    "total_detections": detection_total,
    "average_detections_per_sampled_frame": (
        detection_total / frames_processed if frames_processed else 0
    ),
    "frames_with_track_ids": frames_with_tracks,
    "distinct_track_ids_observed": len(all_track_ids),
    "class_detection_counts": dict(class_counts),
    "average_model_inference_ms": round(avg_inference, 3),
    "average_model_inference_fps": (
        round(1000 / avg_inference, 2) if avg_inference > 0 else None
    ),
    "end_to_end_processing_seconds": round(elapsed, 2),
    "end_to_end_sampled_frames_per_second": (
        round(frames_processed / elapsed, 2) if elapsed > 0 else None
    ),
    "note": (
        "Detection and tracking diagnostics only. MOTA, IDF1, HOTA, "
        "false positives, false negatives and ID switches require "
        "aligned ground-truth track annotations."
    )
}

SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n===== OC-SORT RUN SUMMARY =====")
print(json.dumps(summary, indent=2))
print(f"\nFrame-level CSV: {FRAME_CSV}")
print(f"Summary JSON:    {SUMMARY_JSON}")
