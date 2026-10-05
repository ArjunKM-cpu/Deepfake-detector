"""
Model training module for AI Image Authenticity Checker & Deepfake Detector.
Performs transfer learning using EfficientNet-B0 with validation monitoring and model saving.
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import get_device, set_seed, ensure_dir, load_efficientnet_b0, generate_synthetic_dataset
from src.data_preprocessing import create_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains model for a single epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates model on validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def train_model(
    data_dir="dataset",
    output_dir="models",
    model_name="deepfake_detector.pth",
    epochs=10,
    batch_size=16,
    lr=1e-4,
    auto_synthetic=True,
    device=None
):
    """
    Main training function.

    Args:
        data_dir (str): Base dataset directory.
        output_dir (str): Path to save trained model checkpoint.
        model_name (str): Filename for saved model weights.
        epochs (int): Number of training epochs.
        batch_size (int): DataLoader batch size.
        lr (float): Learning rate.
        auto_synthetic (bool): Automatically generate synthetic dataset if missing/empty.
        device (torch.device, optional): Compute device.
    """
    if device is None:
        device = get_device()

    set_seed(42)
    ensure_dir(output_dir)

    # Check and create dataloaders
    dataloaders = create_dataloaders(data_dir=data_dir, batch_size=batch_size)
    if dataloaders is None or dataloaders["train"] is None or len(dataloaders["train"].dataset) == 0:
        if auto_synthetic:
            print("[Warning] Dataset empty or missing. Generating synthetic dataset for demonstration...")
            generate_synthetic_dataset(base_dir=data_dir)
            dataloaders = create_dataloaders(data_dir=data_dir, batch_size=batch_size)
        else:
            raise ValueError(f"No valid dataset found in '{data_dir}'. Please populate dataset/train/real and dataset/train/fake.")

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"\n[Info] Starting training on device: {device}")
    print(f"  - Train dataset size: {len(train_loader.dataset)} images")
    if val_loader:
        print(f"  - Validation dataset size: {len(val_loader.dataset)} images")

    # Model, loss, optimizer
    model = load_efficientnet_b0(pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    save_path = os.path.join(output_dir, model_name)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = train_loss, train_acc

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        # Save model checkpoint if improved
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_to_idx": {"REAL": 0, "MANIPULATED": 1}
            }, save_path)
            print(f"  --> Saved best model checkpoint to '{save_path}' (Val Acc: {val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[Success] Training complete in {total_time/60:.2f} minutes.")
    print(f"  - Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  - Model saved to: {save_path}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    return model, save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Deepfake Detector")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for trained models")
    parser.add_argument("--model_name", type=str, default="deepfake_detector.pth", help="Model filename")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
