import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import AnimalDataset
from torchmetrics.detection.means_ap import MeanAveragePrecision
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np


def collate_fn(batch):
    return tuple(zip(*batch))


def box_iou(box1, box2):
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    val_dataset = AnimalDataset("val")

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # MODEL
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    num_classes = 6

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes
    )

    model.load_state_dict(torch.load("animal_detector2.pth", map_location=device))
    model.to(device)
    model.eval()

    # Metric (IMPORTANT: class_metrics=True)
    metric = MeanAveragePrecision(
        iou_thresholds=[0.5],
        class_metrics=True
    )

    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, targets in val_loader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = []
            gts = []

            for output, target in zip(outputs, targets):

                preds.append({
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu(),
                })

                gts.append({
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                })

                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()

                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                for i, gt_box in enumerate(gt_boxes):

                    best_iou = 0
                    best_label = 0

                    for j, pred_box in enumerate(pred_boxes):

                        iou = box_iou(gt_box, pred_box)

                        if iou > best_iou and iou >= 0.5:
                            best_iou = iou
                            best_label = pred_labels[j]

                    all_true.append(gt_labels[i])
                    all_pred.append(best_label)

            metric.update(preds, gts)

    results = metric.compute()

    overall_map50 = results["map_50"].item()
    map_per_class = results["map_per_class"]

    precision = precision_score(all_true, all_pred, average=None, zero_division=0)
    recall = recall_score(all_true, all_pred, average=None, zero_division=0)

    cm = confusion_matrix(all_true, all_pred)

    print("\nOverall mAP@50:", overall_map50)

    print("\nPer Class mAP@50:")
    for i, m in enumerate(map_per_class):
        print(f"Class {i}: {m.item():.4f}")

    print("\nPrecision per class:")
    for i, p in enumerate(precision):
        print(f"Class {i}: {p:.4f}")

    print("\nRecall per class:")
    for i, r in enumerate(recall):
        print(f"Class {i}: {r:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()