import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import AnimalDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import os


# -------------------------------
# Collate Function
# -------------------------------
def collate_fn(batch):
    return tuple(zip(*batch))


# -------------------------------
# IoU Function
# -------------------------------
def box_iou_matrix(gt_boxes, pred_boxes):
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            x1 = max(gt[0], pred[0])
            y1 = max(gt[1], pred[1])
            x2 = min(gt[2], pred[2])
            y2 = min(gt[3], pred[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)

            area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
            area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])

            union = area_gt + area_pred - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0

    return iou_matrix


# -------------------------------
# Confusion Matrix Plot
# -------------------------------
def plot_confusion_matrix(cm, class_names):
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", color="black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    val_path = os.path.join(BASE_DIR, "data", "val")
    model_path = "animal_detector_final.pth"

    val_dataset = AnimalDataset(val_path)

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    # MODEL
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    num_classes = 6
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Metric
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)

    CONF_THRESH = 0.5
    IOU_THRESH = 0.5

    all_true = []
    all_pred = []

    debug_count = 0  # LIMIT DEBUG OUTPUT

    with torch.no_grad():
        for images, targets in val_loader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = []
            gts = []

            for idx, (output, target) in enumerate(zip(outputs, targets)):

                # ---------------- DEBUG (LIMITED) ----------------
                if debug_count < 5:
                    print("\n----- DEBUG SAMPLE -----")
                    print("GT labels:", target["labels"].cpu().numpy())
                    print("Pred labels:", output["labels"].cpu().numpy())

                    img = images[idx].cpu().permute(1, 2, 0).numpy()
                    plt.imshow(img)
                    plt.title(f"Pred: {output['labels'].cpu().numpy()}")
                    plt.axis("off")
                    plt.show()

                    debug_count += 1

                # -----------------------
                # Apply confidence threshold
                # -----------------------
                keep = output["scores"] > CONF_THRESH

                pred_boxes = output["boxes"][keep].cpu().numpy()
                pred_labels = output["labels"][keep].cpu().numpy()

                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()

                preds.append({
                    "boxes": torch.tensor(pred_boxes),
                    "scores": torch.ones(len(pred_boxes)),
                    "labels": torch.tensor(pred_labels)
                })

                gts.append({
                    "boxes": target["boxes"],
                    "labels": target["labels"]
                })

                if len(gt_boxes) == 0:
                    continue

                if len(pred_boxes) == 0:
                    for gt_label in gt_labels:
                        all_true.append(gt_label)
                        all_pred.append(0)
                    continue

                # -----------------------
                # Hungarian Matching
                # -----------------------
                iou_mat = box_iou_matrix(gt_boxes, pred_boxes)

                cost_matrix = 1 - iou_mat
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                matched_gt = set()
                matched_pred = set()

                for r, c in zip(row_ind, col_ind):
                    if iou_mat[r, c] >= IOU_THRESH:
                        all_true.append(gt_labels[r])
                        all_pred.append(pred_labels[c])

                        matched_gt.add(r)
                        matched_pred.add(c)

                for i in range(len(gt_labels)):
                    if i not in matched_gt:
                        all_true.append(gt_labels[i])
                        all_pred.append(0)

                for j in range(len(pred_labels)):
                    if j not in matched_pred:
                        all_true.append(0)
                        all_pred.append(pred_labels[j])

            metric.update(preds, gts)

    # -----------------------
    # Metrics
    # -----------------------
    results = metric.compute()

    print("\nOverall mAP@50:", results["map_50"].item())

    print("\nPer Class mAP@50:")
    for i, m in enumerate(results["map_per_class"]):
        print(f"Class {i}: {m.item():.4f}")

    print("\nUnique predicted labels:", set(all_pred))

    print("\nSample GT vs Pred:")
    for i in range(min(10, len(all_true))):
        print(all_true[i], "→", all_pred[i])

    # -----------------------
    # Confusion Matrix
    # -----------------------
    cm = confusion_matrix(all_true, all_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # FINAL CORRECT CLASS NAMES
    class_names = ["bg", "bear", "deer", "dog", "elephant", "tiger"]

    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()

# import torch
# import torchvision
# from torch.utils.data import DataLoader
# from dataset import AnimalDataset
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import linear_sum_assignment


# # -------------------------------
# # Collate Function
# # -------------------------------
# def collate_fn(batch):
#     return tuple(zip(*batch))

# # -------------------------------
# # IoU Function
# # -------------------------------
# def box_iou_matrix(gt_boxes, pred_boxes):
#     iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

#     for i, gt in enumerate(gt_boxes):
#         for j, pred in enumerate(pred_boxes):
#             x1 = max(gt[0], pred[0])
#             y1 = max(gt[1], pred[1])
#             x2 = min(gt[2], pred[2])
#             y2 = min(gt[3], pred[3])

#             inter = max(0, x2 - x1) * max(0, y2 - y1)

#             area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
#             area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])

#             union = area_gt + area_pred - inter
#             iou_matrix[i, j] = inter / union if union > 0 else 0

#     return iou_matrix


# # -------------------------------
# # Confusion Matrix Plot
# # -------------------------------
# def plot_confusion_matrix(cm, class_names):
#     cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(cm_norm)

#     ax.set_xticks(np.arange(len(class_names)))
#     ax.set_yticks(np.arange(len(class_names)))

#     ax.set_xticklabels(class_names, rotation=45, ha="right")
#     ax.set_yticklabels(class_names)

#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Normalized Confusion Matrix")

#     for i in range(len(class_names)):
#         for j in range(len(class_names)):
#             ax.text(j, i, f"{cm_norm[i, j]:.2f}",
#                     ha="center", va="center", color="black")

#     fig.colorbar(im)
#     plt.tight_layout()
#     plt.show()


# # -------------------------------
# # MAIN FUNCTION
# # -------------------------------
# def main():

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     val_dataset = AnimalDataset("../data/val")

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=4,
#         shuffle=False,
#         collate_fn=collate_fn
#     )

#     # MODEL
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

#     num_classes = 6
#     in_features = model.roi_heads.box_predictor.cls_score.in_features

#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
#         in_features, num_classes
#     )

#     model.load_state_dict(torch.load("../animal_detector2.pth", map_location=device))
#     model.to(device)
#     model.eval()

#     # Metric
#     metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)

#     CONF_THRESH = 0.5
#     IOU_THRESH = 0.5

#     all_true = []
#     all_pred = []

#     with torch.no_grad():
#         for images, targets in val_loader:

#             images = [img.to(device) for img in images]
#             outputs = model(images)

#             preds = []
#             gts = []

#             for output, target in zip(outputs, targets):

#                 # -----------------------
#                 # Apply confidence threshold
#                 # -----------------------
#                 keep = output["scores"] > CONF_THRESH

#                 pred_boxes = output["boxes"][keep].cpu().numpy()
#                 pred_labels = output["labels"][keep].cpu().numpy()

#                 gt_boxes = target["boxes"].numpy()
#                 gt_labels = target["labels"].numpy()

#                 preds.append({
#                     "boxes": torch.tensor(pred_boxes),
#                     "scores": torch.ones(len(pred_boxes)),
#                     "labels": torch.tensor(pred_labels)
#                 })

#                 gts.append({
#                     "boxes": target["boxes"],
#                     "labels": target["labels"]
#                 })

#                 if len(gt_boxes) == 0:
#                     continue

#                 if len(pred_boxes) == 0:
#                     for gt_label in gt_labels:
#                         all_true.append(gt_label)
#                         all_pred.append(0)
#                     continue

#                 # -----------------------
#                 # Hungarian Matching
#                 # -----------------------
#                 iou_mat = box_iou_matrix(gt_boxes, pred_boxes)

#                 cost_matrix = 1 - iou_mat
#                 row_ind, col_ind = linear_sum_assignment(cost_matrix)

#                 matched_gt = set()
#                 matched_pred = set()

#                 # Assign matches
#                 for r, c in zip(row_ind, col_ind):
#                     if iou_mat[r, c] >= IOU_THRESH:
#                         all_true.append(gt_labels[r])
#                         all_pred.append(pred_labels[c])

#                         matched_gt.add(r)
#                         matched_pred.add(c)

#                 # Unmatched GT → FN
#                 for i in range(len(gt_labels)):
#                     if i not in matched_gt:
#                         all_true.append(gt_labels[i])
#                         all_pred.append(0)

#                 # Unmatched Predictions → FP
#                 for j in range(len(pred_labels)):
#                     if j not in matched_pred:
#                         all_true.append(0)
#                         all_pred.append(pred_labels[j])

#             metric.update(preds, gts)

#     # -----------------------
#     # Metrics
#     # -----------------------
#     results = metric.compute()

#     print("\nOverall mAP@50:", results["map_50"].item())

#     print("\nPer Class mAP@50:")
#     for i, m in enumerate(results["map_per_class"]):
#         print(f"Class {i}: {m.item():.4f}")

#     # -----------------------
#     # Confusion Matrix
#     # -----------------------
#     cm = confusion_matrix(all_true, all_pred)

#     print("\nConfusion Matrix:")
#     print(cm)

#     class_names = ["bg", "bear", "deer", "dog", "elephant", "tiger"]

#     plot_confusion_matrix(cm, class_names)


# if __name__ == "__main__":
#     main()

# import torch
# import torchvision
# from torch.utils.data import DataLoader
# from dataset import AnimalDataset
# from torchmetrics.detection.means_ap import MeanAveragePrecision
# from sklearn.metrics import confusion_matrix, precision_score, recall_score
# import numpy as np


# def collate_fn(batch):
#     return tuple(zip(*batch))


# def box_iou(box1, box2):
#     area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
#     area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     inter = max(0, x2-x1) * max(0, y2-y1)
#     union = area1 + area2 - inter

#     return inter / union if union > 0 else 0


# def main():

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     val_dataset = AnimalDataset("val")

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=4,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=collate_fn
#     )

#     # MODEL
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

#     num_classes = 6

#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
#         in_features,
#         num_classes
#     )

#     model.load_state_dict(torch.load("animal_detector2.pth", map_location=device))
#     model.to(device)
#     model.eval()

#     # Metric (IMPORTANT: class_metrics=True)
#     metric = MeanAveragePrecision(
#         iou_thresholds=[0.5],
#         class_metrics=True
#     )

#     all_true = []
#     all_pred = []

#     with torch.no_grad():
#         for images, targets in val_loader:

#             images = [img.to(device) for img in images]
#             outputs = model(images)

#             preds = []
#             gts = []

#             for output, target in zip(outputs, targets):

#                 preds.append({
#                     "boxes": output["boxes"].cpu(),
#                     "scores": output["scores"].cpu(),
#                     "labels": output["labels"].cpu(),
#                 })

#                 gts.append({
#                     "boxes": target["boxes"],
#                     "labels": target["labels"],
#                 })

#                 gt_boxes = target["boxes"].numpy()
#                 gt_labels = target["labels"].numpy()

#                 pred_boxes = output["boxes"].cpu().numpy()
#                 pred_labels = output["labels"].cpu().numpy()

#                 for i, gt_box in enumerate(gt_boxes):

#                     best_iou = 0
#                     best_label = 0

#                     for j, pred_box in enumerate(pred_boxes):

#                         iou = box_iou(gt_box, pred_box)

#                         if iou > best_iou and iou >= 0.5:
#                             best_iou = iou
#                             best_label = pred_labels[j]

#                     all_true.append(gt_labels[i])
#                     all_pred.append(best_label)

#             metric.update(preds, gts)

#     results = metric.compute()

#     overall_map50 = results["map_50"].item()
#     map_per_class = results["map_per_class"]

#     precision = precision_score(all_true, all_pred, average=None, zero_division=0)
#     recall = recall_score(all_true, all_pred, average=None, zero_division=0)

#     cm = confusion_matrix(all_true, all_pred)

#     print("\nOverall mAP@50:", overall_map50)

#     print("\nPer Class mAP@50:")
#     for i, m in enumerate(map_per_class):
#         print(f"Class {i}: {m.item():.4f}")

#     print("\nPrecision per class:")
#     for i, p in enumerate(precision):
#         print(f"Class {i}: {p:.4f}")

#     print("\nRecall per class:")
#     for i, r in enumerate(recall):
#         print(f"Class {i}: {r:.4f}")

#     print("\nConfusion Matrix:")
#     print(cm)


# if __name__ == "__main__":
#     main()