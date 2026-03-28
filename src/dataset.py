import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
import random


class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train

        # Transforms
        self.to_tensor = T.ToTensor()

        annotation_path = os.path.join(root, "annotations.json")

        with open(annotation_path, "r") as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]

        # Map image_id -> list of annotations
        self.image_id_to_annotations = {}

        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]

        img_path = os.path.join(self.root, "images", img_info["file_name"])

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Get annotations
        anns = self.image_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        # Convert to tensor
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # -------------------------------
        #  DATA AUGMENTATION (SAFE)
        # -------------------------------
        if self.train and random.random() < 0.5:
            # Horizontal flip
            img = T.functional.hflip(img)

            width = img.width

            if len(boxes) > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        # Convert image to tensor AFTER augmentation
        img = self.to_tensor(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        return img, target

    def __len__(self):
        return len(self.images)

