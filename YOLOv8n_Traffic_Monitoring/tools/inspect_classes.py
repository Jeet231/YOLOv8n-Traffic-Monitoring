
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent
GRAM_ROOT = PROJECT_ROOT / "datasets" / "GRAM"

DATASETS = {
    "M-30": GRAM_ROOT / "Annotations" / "M-30" / "xml",
    "M-30-HD": GRAM_ROOT / "Annotations" / "M-30-HD" / "xml",
    "Urban1": GRAM_ROOT / "Annotations" / "Urban1" / "xml",
}


for dataset_name, annotation_dir in DATASETS.items():

    class_counter = Counter()
    object_counter = 0
    invalid_xml_count = 0

    for xml_file in annotation_dir.glob("*.xml"):

        try:
            root = ET.parse(xml_file).getroot()

            for object_element in root.findall("object"):

                class_element = object_element.find("class")

                if class_element is not None:

                    class_name = (class_element.text or "").strip()

                    if class_name:
                        class_counter[class_name] += 1
                        object_counter += 1

        except ET.ParseError:
            invalid_xml_count += 1

    print("\n" + "=" * 60)
    print("DATASET:", dataset_name)
    print("=" * 60)

    print("Total objects:", object_counter)
    print("Invalid XML files:", invalid_xml_count)

    print("\nClasses:")

    for class_name, count in sorted(class_counter.items()):

        print(f"  {class_name}: {count}")