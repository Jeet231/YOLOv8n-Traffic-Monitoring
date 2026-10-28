
from pathlib import Path
from collections import Counter
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "prepared_data" / "GRAM"


def main():
    yaml_path = DATA_ROOT / "data.yaml"

    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    class_count = int(config["nc"])
    class_names = config["names"]

    print("Number of classes:", class_count)
    print("Class names:", class_names)

    total_labels = 0
    total_objects = 0
    invalid_labels = []
    empty_labels = 0

    for split in ["train", "val", "test"]:
        image_dir = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split

        images = list(image_dir.glob("*"))
        labels = list(label_dir.glob("*.txt"))

        print(f"\nSplit: {split}")
        print("Images:", len(images))
        print("Labels:", len(labels))

        image_stems = {image.stem for image in images}
        label_stems = {label.stem for label in labels}

        missing_labels = image_stems - label_stems

        if missing_labels:
            print(
                "Images without label files:",
                len(missing_labels),
            )

        for label_file in labels:
            total_labels += 1

            lines = label_file.read_text(
                encoding="utf-8"
            ).splitlines()

            if not lines:
                empty_labels += 1
                continue

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()

                if len(parts) != 5:
                    invalid_labels.append(
                        (
                            str(label_file),
                            line_number,
                            "expected 5 values",
                        )
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    invalid_labels.append(
                        (
                            str(label_file),
                            line_number,
                            "non-numeric value",
                        )
                    )
                    continue

                center_x, center_y, width, height = values

                if not 0 <= class_id < class_count:
                    invalid_labels.append(
                        (
                            str(label_file),
                            line_number,
                            "invalid class ID",
                        )
                    )

                if not all(
                    0 <= value <= 1
                    for value in values
                ):
                    invalid_labels.append(
                        (
                            str(label_file),
                            line_number,
                            "coordinates outside 0-1",
                        )
                    )

                if width <= 0 or height <= 0:
                    invalid_labels.append(
                        (
                            str(label_file),
                            line_number,
                            "non-positive box size",
                        )
                    )

                total_objects += 1

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print("Label files:", total_labels)
    print("Objects:", total_objects)
    print("Empty label files:", empty_labels)
    print("Invalid entries:", len(invalid_labels))

    if invalid_labels:
        print("\nFirst invalid entries:")

        for item in invalid_labels[:20]:
            print(item)

        raise RuntimeError(
            "YOLO label validation failed."
        )

    print("\nYOLO label validation passed.")


if __name__ == "__main__":
    main()