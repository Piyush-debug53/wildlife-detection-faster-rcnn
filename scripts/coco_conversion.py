import os
import json
import re
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "val", "test"]

categories = [
    {"id": 1, "name": "bear"},
    {"id": 2, "name": "deer"},
    {"id": 3, "name": "dog"},
    {"id": 4, "name": "elephant"},
    {"id": 5, "name": "tiger"}
]

def natural_sort_key(filename):
    """
    Sort filenames naturally:
    bear (2).jpg before bear (10).jpg
    Bear_0313.jpg before Bear_1020.jpg
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', filename)
    ]

for split in SPLITS:
    print(f"\nProcessing {split}...")

    image_dir = os.path.join(BASE_DIR, split, "images")
    label_dir = os.path.join(BASE_DIR, split, "labels")

    if not os.path.exists(image_dir):
        print(f"{split} images folder not found. Skipping.")
        continue

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Natural sorting (FIXED)
    image_files = sorted(image_files, key=natural_sort_key)

    print("Number of images:", len(image_files))

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id = 0
    annotation_id = 0

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_name = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        # Read image size
        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        # Process label if exists
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])  # already shifted to 1–5

                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height

                x_min = x_center - box_width / 2
                y_min = y_center - box_height / 2

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

    output_path = os.path.join(BASE_DIR, split, "annotations.json")

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"Saved: {output_path}")

print("\nCOCO Conversion Completed Successfully.")