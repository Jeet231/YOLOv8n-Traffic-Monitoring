
from pathlib import Path
import xml.etree.ElementTree as ET


# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Print XML tree
# --------------------------------------------------

def print_xml_structure(element, level=0, max_depth=5):
    """
    Print XML tags and text values.
    Limits depth to avoid excessively long output.
    """

    if level > max_depth:
        return

    indentation = "  " * level
    text = (element.text or "").strip()

    if text:
        print(f"{indentation}{element.tag}: {text}")
    else:
        print(f"{indentation}{element.tag}")

    for child in element:
        print_xml_structure(child, level + 1, max_depth)


# --------------------------------------------------
# 3. Inspect one XML file
# --------------------------------------------------

def inspect_one_xml(dataset_name, annotation_dir):

    xml_files = sorted(annotation_dir.glob("*.xml"))

    if not xml_files:
        print(f"No XML files found for {dataset_name}")
        return

    xml_file = xml_files[0]

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    print("Number of XML files:", len(xml_files))
    print("Inspecting:", xml_file.name)

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        print("\nXML root:", root.tag)
        print("\nXML structure:")

        print_xml_structure(root)

    except ET.ParseError as error:
        print("XML parsing error:", error)


# --------------------------------------------------
# 4. Inspect datasets
# --------------------------------------------------

for dataset_name, paths in DATASETS.items():

    annotation_dir = paths["annotations"]

    print("\nChecking:", dataset_name)
    print("Annotation directory:", annotation_dir)
    print("Exists:", annotation_dir.exists())

    if annotation_dir.exists():
        inspect_one_xml(dataset_name, annotation_dir)