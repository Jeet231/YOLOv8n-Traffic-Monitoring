
from pathlib import Path
import xml.etree.ElementTree as ET


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


def inspect_names(dataset_name, paths):

    image_dir = paths["images"]
    annotation_dir = paths["annotations"]

    print("\n" + "=" * 70)
    print("DATASET:", dataset_name)
    print("=" * 70)

    images = sorted(
        file for file in image_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    xml_files = sorted(annotation_dir.glob("*.xml"))

    print("\nFirst 15 image filenames:")

    for image in images[:15]:
        print(" ", image.name)

    print("\nFirst 15 XML filenames and internal image names:")

    for xml_file in xml_files[:15]:

        try:
            root = ET.parse(xml_file).getroot()

            filename_element = root.find("filename")

            if filename_element is not None:
                internal_filename = (filename_element.text or "").strip()
            else:
                internal_filename = "NO_FILENAME_TAG"

            print(
                f" XML file: {xml_file.name:20s}"
                f" Internal filename: {internal_filename}"
            )

        except ET.ParseError as error:
            print("XML parsing error:", xml_file, error)


for dataset_name, paths in DATASETS.items():
    inspect_names(dataset_name, paths)