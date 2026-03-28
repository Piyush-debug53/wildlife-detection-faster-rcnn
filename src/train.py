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

    # -------------------------------
    # DATASET
    # -------------------------------
    train_dataset = AnimalDataset("../data/train", train=True)
    val_dataset = AnimalDataset("../data/val", train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,   # safer for Windows
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # -------------------------------
    # MODEL
    # -------------------------------
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    num_classes = 6  # bg + 5 animals

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes
    )

    model.to(device)

    # -------------------------------
    # STAGE 1: Freeze backbone
    # -------------------------------
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.003,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    num_epochs = 30

    print("\n Starting Training...")

    for epoch in range(num_epochs):

        # -------------------------------
        # Unfreeze after 10 epochs
        # -------------------------------
        if epoch == 10:
            print("\n Unfreezing backbone...")

            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.001,   # lower LR after unfreeze
                momentum=0.9,
                weight_decay=0.0005
            )

            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=5,
                gamma=0.1
            )

        # -------------------------------
        # TRAIN
        # -------------------------------
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

            progress_bar.set_postfix({
                "Loss": f"{losses.item():.4f}",
                "Cls": f"{loss_dict['loss_classifier']:.4f}",
                "Box": f"{loss_dict['loss_box_reg']:.4f}"
            })

        avg_train_loss = running_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")

        # -------------------------------
        # VALIDATION
        # -------------------------------
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

        # Step LR scheduler
        lr_scheduler.step()

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    torch.save(model.state_dict(), "animal_detector_final.pth")
    print("\n✅ Training Finished Successfully. Model Saved.")


if __name__ == "__main__":
    main()
    