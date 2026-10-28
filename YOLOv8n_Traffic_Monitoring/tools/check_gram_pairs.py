
from pathlib import Path


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


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def check_dataset(dataset_name, paths):

    image_dir = paths["images"]
    annotation_dir = paths["annotations"]

    print("\n" + "=" * 70)
    print("DATASET:", dataset_name)
    print("=" * 70)

    if not image_dir.exists():
        print("Image directory does not exist:", image_dir)
        return

    if not annotation_dir.exists():
        print("Annotation directory does not exist:", annotation_dir)
        return

    image_files = sorted(
        file for file in image_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    xml_files = sorted(annotation_dir.rglob("*.xml"))

    print("Image count:", len(image_files))
    print("XML count:", len(xml_files))

    image_stems = {
        file.stem
        for file in image_files
    }

    xml_stems = {
        file.stem
        for file in xml_files
    }

    images_without_xml = image_stems - xml_stems
    xml_without_images = xml_stems - image_stems

    print("Images without matching XML:", len(images_without_xml))
    print("XML files without matching image:", len(xml_without_images))

    if images_without_xml:
        print("\nExample images without XML:")
        for name in sorted(images_without_xml)[:10]:
            print(" ", name)

    if xml_without_images:
        print("\nExample XML files without images:")
        for name in sorted(xml_without_images)[:10]:
            print(" ", name)


for dataset_name, paths in DATASETS.items():
    check_dataset(dataset_name, paths)