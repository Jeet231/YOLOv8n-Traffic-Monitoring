
from pathlib import Path
from collections import Counter
import csv
import shutil
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAM_ROOT = PROJECT_ROOT / "datasets" / "GRAM"
OUTPUT_ROOT = PROJECT_ROOT / "prepared_data" / "GRAM"

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

# Use one of the following options:
# "all"        = combine all three datasets
# "M-30"       = prepare only M-30
# "M-30-HD"    = prepare only M-30-HD
# "Urban1"     = prepare only Urban1
DATASET_SELECTION = "all"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def resolve_image(xml_file, image_dir):
    """
    Resolve:
        1.xml -> image000001.jpg

    XML file 0.xml is expected to have no corresponding image
    because image000000.jpg does not exist in the dataset.
    """
    if not xml_file.stem.isdigit():
        return None

    frame_number = int(xml_file.stem)

    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"image{frame_number:06d}{extension}"

        if candidate.exists():
            return candidate

    return None


def get_text(parent, tag, default=""):
    element = parent.find(tag)

    if element is None or element.text is None:
        return default

    return element.text.strip()


def parse_xml(xml_file):
    """
    Return parsed object information from one XML file.
    """
    root = ET.parse(xml_file).getroot()

    width = int(float(get_text(root.find("size"), "width", "0")))
    height = int(float(get_text(root.find("size"), "height", "0")))

    objects = []

    for obj in root.findall("object"):
        class_name = get_text(obj, "class", "unknown")
        object_id = get_text(obj, "ID", "")
        trajectory = get_text(obj, "trajectory", "")

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        try:
            xmin = float(get_text(bbox, "xmin"))
            ymin = float(get_text(bbox, "ymin"))
            xmax = float(get_text(bbox, "xmax"))
            ymax = float(get_text(bbox, "ymax"))
        except ValueError:
            continue

        objects.append({
            "class_name": class_name,
            "object_id": object_id,
            "trajectory": trajectory,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return width, height, objects


def validate_bbox(obj, width, height):
    """
    Validate and clip a bounding box to image boundaries.
    """
    xmin = max(0.0, min(obj["xmin"], width - 1))
    ymin = max(0.0, min(obj["ymin"], height - 1))
    xmax = max(0.0, min(obj["xmax"], width - 1))
    ymax = max(0.0, min(obj["ymax"], height - 1))

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax


def bbox_to_yolo(xmin, ymin, xmax, ymax, width, height):
    center_x = ((xmin + xmax) / 2.0) / width
    center_y = ((ymin + ymax) / 2.0) / height

    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return center_x, center_y, box_width, box_height


def selected_datasets():
    if DATASET_SELECTION == "all":
        return list(DATASETS.keys())

    if DATASET_SELECTION not in DATASETS:
        raise ValueError(
            f"Unknown dataset selection: {DATASET_SELECTION}"
        )

    return [DATASET_SELECTION]


def collect_records(dataset_names):
    """
    Collect valid image/XML pairs and discover all classes.
    """
    records = []
    class_counter = Counter()
    skipped_pairs = []

    for dataset_name in dataset_names:
        config = DATASETS[dataset_name]

        annotation_files = sorted(
            config["annotations"].glob("*.xml"),
            key=lambda path: int(path.stem)
            if path.stem.isdigit()
            else path.stem,
        )

        print(f"\nCollecting dataset: {dataset_name}")

        for xml_file in tqdm(annotation_files):
            image_path = resolve_image(
                xml_file,
                config["images"],
            )

            if image_path is None:
                skipped_pairs.append({
                    "dataset": dataset_name,
                    "xml": xml_file.name,
                    "reason": "image_not_found",
                })
                continue

            try:
                width, height, objects = parse_xml(xml_file)
            except Exception as error:
                skipped_pairs.append({
                    "dataset": dataset_name,
                    "xml": xml_file.name,
                    "reason": f"xml_error: {error}",
                })
                continue

            if width <= 0 or height <= 0:
                skipped_pairs.append({
                    "dataset": dataset_name,
                    "xml": xml_file.name,
                    "reason": "invalid_dimensions",
                })
                continue

            for obj in objects:
                if validate_bbox(obj, width, height) is not None:
                    class_counter[obj["class_name"]] += 1

            records.append({
                "dataset": dataset_name,
                "xml_path": xml_file,
                "image_path": image_path,
                "frame_id": int(xml_file.stem),
                "width": width,
                "height": height,
                "objects": objects,
            })

    return records, class_counter, skipped_pairs


def create_class_mapping(class_counter):
    class_names = sorted(class_counter.keys())

    class_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(class_names)
    }

    print("\nClass mapping:")

    for class_name, class_id in class_to_id.items():
        print(
            f"  {class_id}: {class_name} "
            f"({class_counter[class_name]} objects)"
        )

    return class_to_id


def chronological_split(records):
    """
    Split each dataset independently in chronological order.
    """
    split_records = []

    for dataset_name in sorted(set(r["dataset"] for r in records)):
        dataset_records = [
            r for r in records
            if r["dataset"] == dataset_name
        ]

        dataset_records.sort(key=lambda r: r["frame_id"])

        total = len(dataset_records)

        train_end = int(total * TRAIN_RATIO)
        val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

        for index, record in enumerate(dataset_records):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"

            record_copy = record.copy()
            record_copy["split"] = split
            split_records.append(record_copy)

    return split_records


def write_yolo_label(record, class_to_id, label_path):
    """
    Write one YOLO label file.
    """
    lines = []

    for obj in record["objects"]:
        valid_bbox = validate_bbox(
            obj,
            record["width"],
            record["height"],
        )

        if valid_bbox is None:
            continue

        xmin, ymin, xmax, ymax = valid_bbox

        if obj["class_name"] not in class_to_id:
            continue

        class_id = class_to_id[obj["class_name"]]

        center_x, center_y, box_width, box_height = bbox_to_yolo(
            xmin,
            ymin,
            xmax,
            ymax,
            record["width"],
            record["height"],
        )

        lines.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def write_tracking_rows(record, tracking_writer):
    """
    Write tracking ground truth separately from YOLO labels.
    """
    for obj in record["objects"]:
        valid_bbox = validate_bbox(
            obj,
            record["width"],
            record["height"],
        )

        if valid_bbox is None:
            continue

        xmin, ymin, xmax, ymax = valid_bbox

        tracking_writer.writerow({
            "dataset": record["dataset"],
            "split": record["split"],
            "frame_id": record["frame_id"],
            "image_name": record["image_path"].name,
            "class_name": obj["class_name"],
            "object_id": obj["object_id"],
            "trajectory": obj["trajectory"],
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "image_width": record["width"],
            "image_height": record["height"],
        })


def main():
    dataset_names = selected_datasets()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    records, class_counter, skipped_pairs = collect_records(
        dataset_names
    )

    if not records:
        raise RuntimeError("No valid records found.")

    class_to_id = create_class_mapping(class_counter)

    split_records = chronological_split(records)

    split_counts = Counter(
        record["split"] for record in split_records
    )

    print("\nSplit counts:")

    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {split_counts[split_name]}")

    tracking_path = OUTPUT_ROOT / "tracking_ground_truth.csv"

    tracking_fields = [
        "dataset",
        "split",
        "frame_id",
        "image_name",
        "class_name",
        "object_id",
        "trajectory",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "image_width",
        "image_height",
    ]

    with tracking_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as tracking_file:
        tracking_writer = csv.DictWriter(
            tracking_file,
            fieldnames=tracking_fields,
        )

        tracking_writer.writeheader()

        for record in tqdm(split_records, desc="Preparing images"):
            split = record["split"]

            image_output_dir = (
                OUTPUT_ROOT / "images" / split
            )

            label_output_dir = (
                OUTPUT_ROOT / "labels" / split
            )

            image_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            label_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_stem = (
                f"{record['dataset']}_"
                f"{record['frame_id']:06d}"
            )

            image_output_path = (
                image_output_dir /
                f"{output_stem}{record['image_path'].suffix}"
            )

            label_output_path = (
                label_output_dir /
                f"{output_stem}.txt"
            )

            if not image_output_path.exists():
                shutil.copy2(
                    record["image_path"],
                    image_output_path,
                )

            write_yolo_label(
                record,
                class_to_id,
                label_output_path,
            )

            write_tracking_rows(
                record,
                tracking_writer,
            )

    names = [
        name for name, _ in sorted(
            class_to_id.items(),
            key=lambda item: item[1],
        )
    ]

    data_yaml = {
        "path": str(OUTPUT_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }

    yaml_path = OUTPUT_ROOT / "data.yaml"

    with yaml_path.open("w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(
            data_yaml,
            yaml_file,
            sort_keys=False,
        )

    class_mapping_path = OUTPUT_ROOT / "class_mapping.yaml"

    with class_mapping_path.open("w", encoding="utf-8") as mapping_file:
        yaml.safe_dump(
            class_to_id,
            mapping_file,
            sort_keys=False,
        )

    skipped_path = OUTPUT_ROOT / "skipped_records.csv"

    with skipped_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as skipped_file:
        skipped_fields = ["dataset", "xml", "reason"]

        writer = csv.DictWriter(
            skipped_file,
            fieldnames=skipped_fields,
        )

        writer.writeheader()
        writer.writerows(skipped_pairs)

    print("\nPreparation complete.")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Tracking CSV: {tracking_path}")
    print(f"YOLO configuration: {yaml_path}")
    print(f"Skipped records: {len(skipped_pairs)}")


if __name__ == "__main__":
    main()