from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[1]

    data_yaml = (
        project_root
        / "prepared_data"
        / "GRAM"
        / "data.yaml"
    )

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        project=str(project_root / "results" / "training"),
        name="baseline_yolov8n_50epochs",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()