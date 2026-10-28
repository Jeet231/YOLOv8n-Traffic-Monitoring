
from pathlib import Path
import xml.etree.ElementTree as ET
import re


PROJECT_ROOT = Path(__file__).resolve().parent
GRAM_ROOT = PROJECT_ROOT / "datasets" / "GRAM"

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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_xml_filename(xml_file):

    root = ET.parse(xml_file).getroot()

    filename_element = root.find("filename")

    if filename_element is None:
        return None

    return (filename_element.text or "").strip()


def numeric_part(filename):

    """
    Extract numeric portions from a filename.

    Examples:
        0.jpg            -> 0
        image000001.jpg  -> 000001
    """

    stem = Path(filename).stem

    match = re.search(r"\d+", stem)

    if match:
        return int(match.group())

    return None


def check_dataset(dataset_name, paths):

    image_dir = paths["images"]
    annotation_dir = paths["annotations"]

    print("\n" + "=" * 70)
    print("DATASET:", dataset_name)
    print("=" * 70)

    image_files = sorted(
        file for file in image_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    xml_files = sorted(annotation_dir.rglob("*.xml"))

    image_by_name = {
        file.name.lower(): file
        for file in image_files
    }

    image_by_stem = {
        file.stem.lower(): file
        for file in image_files
    }

    image_by_number = {}

    for image in image_files:

        number = numeric_part(image.name)

        if number is not None:
            image_by_number.setdefault(number, []).append(image)

    exact_matches = 0
    stem_matches = 0
    numeric_candidates = 0
    unmatched = 0

    examples = []

    for xml_file in xml_files:

        internal_filename = get_xml_filename(xml_file)

        if not internal_filename:
            unmatched += 1
            continue

        filename_key = Path(internal_filename).name.lower()
        stem_key = Path(internal_filename).stem.lower()

        matched_image = image_by_name.get(filename_key)

        match_type = None

        if matched_image is not None:

            exact_matches += 1
            match_type = "exact"

        else:

            matched_image = image_by_stem.get(stem_key)

            if matched_image is not None:

                stem_matches += 1
                match_type = "stem"

            else:

                number = numeric_part(internal_filename)

                candidates = image_by_number.get(number, [])

                if len(candidates) == 1:

                    numeric_candidates += 1
                    matched_image = candidates[0]
                    match_type = "numeric candidate"

                else:

                    unmatched += 1
                    match_type = "unmatched"

        if len(examples) < 20:

            examples.append(
                (
                    xml_file.name,
                    internal_filename,
                    matched_image.name if matched_image else "NONE",
                    match_type,
                )
            )

    print("Total XML files:", len(xml_files))
    print("Total images:", len(image_files))
    print("Exact matches:", exact_matches)
    print("Stem matches:", stem_matches)
    print("Numeric candidates:", numeric_candidates)
    print("Unmatched XML files:", unmatched)

    print("\nExample matching results:")

    for xml_name, internal_name, image_name, match_type in examples:

        print(
            f"XML: {xml_name:20s} | "
            f"Internal: {internal_name:20s} | "
            f"Image: {image_name:25s} | "
            f"Type: {match_type}"
        )


for dataset_name, paths in DATASETS.items():
    check_dataset(dataset_name, paths)