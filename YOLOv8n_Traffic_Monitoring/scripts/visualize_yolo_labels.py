
from pathlib import Path
import random

from PIL import Image, ImageDraw
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "prepared_data" / "GRAM"

SPLIT = "train"
SAMPLE_COUNT = 10


def yolo_to_bbox(
    center_x,
    center_y,
    box_width,
    box_height,
    image_width,
    image_height,
):
    x1 = (center_x - box_width / 2) * image_width
    y1 = (center_y - box_height / 2) * image_height
    x2 = (center_x + box_width / 2) * image_width
    y2 = (center_y + box_height / 2) * image_height

    return (
        int(x1),
        int(y1),
        int(x2),
        int(y2),
    )


def main():
    with (DATA_ROOT / "data.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    class_names = config["names"]

    image_dir = DATA_ROOT / "images" / SPLIT
    label_dir = DATA_ROOT / "labels" / SPLIT

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "dataset_checks"
        / "yolo_visualizations"
        / SPLIT
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = list(image_dir.glob("*"))
    random.seed(42)
    random.shuffle(images)

    saved = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        draw = ImageDraw.Draw(image)

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = map(
                float,
                parts[1:],
            )

            bbox = yolo_to_bbox(
                center_x,
                center_y,
                box_width,
                box_height,
                image_width,
                image_height,
            )

            draw.rectangle(
                bbox,
                outline="red",
                width=3,
            )

            class_name = class_names[class_id]

            draw.text(
                (bbox[0], max(0, bbox[1] - 15)),
                class_name,
                fill="yellow",
            )

        output_path = output_dir / image_path.name
        image.save(output_path)

        saved += 1

        if saved >= SAMPLE_COUNT:
            break

    print(f"Saved {saved} YOLO visualization(s).")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()