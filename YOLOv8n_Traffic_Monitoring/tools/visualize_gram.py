
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
GRAM_ROOT = PROJECT_ROOT / "datasets" / "GRAM"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "gram_visualizations"

DATASETS = {
    "M-30": {
        "images": GRAM_ROOT / "Images" / "M-30",
        "annotations": GRAM_ROOT / "Annotations" / "M-30" / "xml",
    },
    "M-30-HD": {
        "images": GRAM_ROOT / "Images" / "M-30-HD",
        "annotations": GRAM_ROOT / "Annotations" / "M-30-HD" / "xml",
    },
    "Urban1": {
        "images": GRAM_ROOT / "Images" / "Urban1",
        "annotations": GRAM_ROOT / "Annotations" / "Urban1" / "xml",
    },
}


def resolve_image(xml_file, image_dir):
    """
    GRAM uses XML numbering that corresponds to the numeric
    portion of image filenames.

    Example:
        1.xml -> image000001.jpg
    """
    try:
        frame_number = int(xml_file.stem)
    except ValueError:
        return None

    candidates = [
        image_dir / f"image{frame_number:06d}.jpg",
        image_dir / f"image{frame_number:06d}.jpeg",
        image_dir / f"image{frame_number:06d}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def draw_annotations(image, xml_file):
    root = ET.parse(xml_file).getroot()
    draw = ImageDraw.Draw(image)

    for obj in root.findall("object"):
        class_name = obj.findtext("class", default="unknown")
        object_id = obj.findtext("ID", default="unknown")
        trajectory = obj.findtext("trajectory", default="")

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        try:
            xmin = int(float(bbox.findtext("xmin")))
            ymin = int(float(bbox.findtext("ymin")))
            xmax = int(float(bbox.findtext("xmax")))
            ymax = int(float(bbox.findtext("ymax")))
        except (TypeError, ValueError):
            continue

        label = f"{class_name} | ID: {object_id}"

        if trajectory:
            label += f" | T: {trajectory}"

        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline="red",
            width=3,
        )

        text_y = max(0, ymin - 18)
        draw.text(
            (xmin, text_y),
            label,
            fill="yellow",
        )

    return image


def visualize_dataset(dataset_name, sample_count=5):
    config = DATASETS[dataset_name]
    image_dir = config["images"]
    annotation_dir = config["annotations"]

    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(
        annotation_dir.glob("*.xml"),
        key=lambda path: int(path.stem)
        if path.stem.isdigit()
        else path.stem,
    )

    saved_count = 0

    for xml_file in xml_files:
        image_path = resolve_image(xml_file, image_dir)

        if image_path is None:
            print(f"Skipped: no image found for {xml_file.name}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            print(f"Could not open {image_path}: {error}")
            continue

        image = draw_annotations(image, xml_file)

        output_path = output_dir / image_path.name
        image.save(output_path)

        print(f"Saved: {output_path}")
        saved_count += 1

        if saved_count >= sample_count:
            break

    print(
        f"\n{dataset_name}: saved "
        f"{saved_count} visualization(s)"
    )


if __name__ == "__main__":
    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'=' * 60}")

        visualize_dataset(dataset_name, sample_count=5)

    print("\nVisualization complete.")