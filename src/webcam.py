import torch
import torchvision
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "animal_detector_final.pth"   # new trained model
CONFIDENCE_THRESHOLD = 0.7   # start with 0.4

class_names = [
    "__background__",
    "bear",
    "deer",
    "dog",
    "elephant",
    "tiger"
]

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# ----------------------------
# Model (MUST match training)
# ----------------------------
model = fasterrcnn_resnet50_fpn(weights=None)

num_classes = 6
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = torchvision.transforms.ToTensor()

# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    boxes = outputs[0]["boxes"]
    labels = outputs[0]["labels"]
    scores = outputs[0]["scores"]

    for box, label, score in zip(boxes, labels, scores):

        if score >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box.int().cpu().numpy()

            label_name = class_names[label]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label_name}: {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Animal Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()