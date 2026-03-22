import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import AnimalDataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm


def collate_fn(batch):
    return tuple(zip(*batch))


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    train_dataset = AnimalDataset("train")
    val_dataset = AnimalDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=10,  # keep 0 on Windows
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=10,
        collate_fn=collate_fn
    )

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    num_classes = 6  # 5 animals + background

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes
    )

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    num_epochs = 10

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        progress_bar = tqdm(train_loader, leave=True)

        for images, targets in progress_bar:

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            # Update single progress bar
            progress_bar.set_postfix({
                "Loss": f"{losses.item():.4f}",
                "Cls": f"{loss_dict['loss_classifier']:.4f}",
                "Box": f"{loss_dict['loss_box_reg']:.4f}"
            })

        avg_train_loss = running_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.train()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "animal_detector.pth")
    print("\nTraining Finished Successfully. Model Saved.")


if __name__ == "__main__":
    main()