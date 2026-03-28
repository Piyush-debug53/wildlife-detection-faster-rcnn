import torch
import torchvision
import cv2
import os

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "animal_detector_final.pth"
VAL_IMAGES_PATH = "../data/val/images"

RAW_OUTPUT_PATH = "inference_raw"
PRED_OUTPUT_PATH = "inference_pred"

CONFIDENCE_THRESHOLD = 0.4
SHOW_IMAGES = True   # True for display

class_names = [
    "__background__",
    "bear",
    "deer",
    "dog",
    "elephant",
    "tiger"
]

# ----------------------------
# Create Output Folders
# ----------------------------
os.makedirs(RAW_OUTPUT_PATH, exist_ok=True)
os.makedirs(PRED_OUTPUT_PATH, exist_ok=True)

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Load Model
# ----------------------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

num_classes = 6
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")

# ----------------------------
# Inference Loop
# ----------------------------
image_files = os.listdir(VAL_IMAGES_PATH)[:10]

for img_name in image_files:

    print(f"\nProcessing image: {img_name}")

    img_path = os.path.join(VAL_IMAGES_PATH, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Failed to load {img_name}")
        continue

    # ----------------------------
    # Save RAW image
    # ----------------------------
    raw_save_path = os.path.join(RAW_OUTPUT_PATH, img_name)
    cv2.imwrite(raw_save_path, image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(image_rgb).to(device)

    # ----------------------------
    # Inference
    # ----------------------------
    with torch.no_grad():
        outputs = model([img_tensor])

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    # ----------------------------
    # Draw predictions
    # ----------------------------
    for box, label, score in zip(boxes, labels, scores):

        if score >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box.int().cpu().numpy()
            label = int(label.item())

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{class_names[label]}: {score:.2f}"
            cv2.putText(image, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

    # ----------------------------
    # Save predicted image
    # ----------------------------
    pred_save_path = os.path.join(PRED_OUTPUT_PATH, img_name)
    cv2.imwrite(pred_save_path, image)

    # ----------------------------
    # Display
    # ----------------------------
    if SHOW_IMAGES:
        cv2.imshow("Prediction", image)
        cv2.waitKey(0)

# ----------------------------
# Cleanup
# ----------------------------
if SHOW_IMAGES:
    cv2.destroyAllWindows()

print("\n✅ Inference completed. Check output folders.")