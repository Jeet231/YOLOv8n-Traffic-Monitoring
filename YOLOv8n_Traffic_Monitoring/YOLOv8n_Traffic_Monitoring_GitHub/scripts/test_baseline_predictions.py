from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "results"
        / "training"
        / "baseline_yolov8n_50epochs"
        / "weights"
        / "best.pt"
    )

    source_path = (
        project_root
        / "prepared_data"
        / "GRAM"
        / "images"
        / "test"
    )

    output_path = (
        project_root
        / "results"
        / "baseline_predictions"
    )

    model = YOLO(str(model_path))

    model.predict(
        source=str(source_path),
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        save_txt=True,
        project=str(output_path),
        name="yolov8n_baseline",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()