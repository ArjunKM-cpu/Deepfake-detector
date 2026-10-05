"""
Evaluation module for AI Image Authenticity Checker & Deepfake Detector.
Computes accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, and exports metrics/plots.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_device, ensure_dir, load_efficientnet_b0, IDX_TO_CLASS
from src.data_preprocessing import create_dataloaders


def evaluate_model(
    model_path="models/deepfake_detector.pth",
    data_dir="dataset",
    output_dir="evaluation_results",
    split="test",
    device=None
):
    """
    Evaluates trained deepfake detection model on dataset split and saves metrics/plots.

    Args:
        model_path (str): Path to trained model checkpoint (.pth).
        data_dir (str): Base dataset directory.
        output_dir (str): Output directory for metrics and visualization plots.
        split (str): Split to evaluate ('test' or 'val').
        device (torch.device, optional): Target compute device.

    Returns:
        dict: Evaluated metrics dictionary.
    """
    if device is None:
        device = get_device()

    ensure_dir(output_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at '{model_path}'. Please run training first.")

    # Load model architecture and state dict
    model = load_efficientnet_b0(pretrained=False, num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load dataloaders
    dataloaders = create_dataloaders(data_dir=data_dir, batch_size=16)
    if dataloaders is None:
        raise ValueError(f"Dataset directory '{data_dir}' is empty or invalid.")

    eval_loader = dataloaders.get("test") if split == "test" and dataloaders.get("test") else dataloaders.get("val")
    if eval_loader is None or len(eval_loader.dataset) == 0:
        print(f"[Warning] Specified split '{split}' empty. Falling back to training loader for evaluation demonstration.")
        eval_loader = dataloaders.get("train")

    print(f"[Info] Evaluating model on {len(eval_loader.dataset)} samples...")

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for class 1 (MANIPULATED)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(y_true))
    }

    # Print evaluation summary
    print("\n" + "="*50)
    print(" MODEL EVALUATION RESULTS ")
    print("="*50)
    print(f"Total Samples Evaluated: {len(y_true)}")
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Precision:       {prec * 100:.2f}%")
    print(f"Recall:          {rec * 100:.2f}%")
    print(f"F1-Score:        {f1 * 100:.2f}%")
    print(f"ROC-AUC Score:   {auc:.4f}")
    print("Confusion Matrix:")
    print(f"  [[TN={cm[0,0] if cm.shape==(2,2) else 0}, FP={cm[0,1] if cm.shape==(2,2) else 0}],")
    print(f"   [FN={cm[1,0] if cm.shape==(2,2) else 0}, TP={cm[1,1] if cm.shape==(2,2) else 0}]]")
    print("="*50 + "\n")

    # Save metrics JSON
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save summary text file
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Deepfake Detection Evaluation Summary\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Precision: {prec * 100:.2f}%\n")
        f.write(f"Recall: {rec * 100:.2f}%\n")
        f.write(f"F1-Score: {f1 * 100:.2f}%\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "MANIPULATED"],
        yticklabels=["REAL", "MANIPULATED"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # Plot ROC Curve if applicable
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_plot_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_plot_path, dpi=300)
        plt.close()

    print(f"[Success] Saved metrics to '{json_path}' and plots to '{output_dir}/'.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Deepfake Detector")
    parser.add_argument("--model_path", type=str, default="models/deepfake_detector.pth", help="Model file path")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Results directory")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate ('test' or 'val')")

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split
    )
