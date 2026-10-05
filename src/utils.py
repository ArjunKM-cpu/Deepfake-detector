"""
Utility functions for AI Image Authenticity Checker & Deepfake Detector.
Handles device configuration, image validation, seed setting, directory management,
model loading, and synthetic dataset creation for testing and demo purposes.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageDraw


# Class label constants
CLASS_NAMES = ["REAL", "MANIPULATED"]
CLASS_TO_IDX = {"REAL": 0, "MANIPULATED": 1}
IDX_TO_CLASS = {0: "REAL", 1: "MANIPULATED"}


def get_device():
    """
    Automatically detects and returns CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Info] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Info] Using CPU for computation.")
    return device


def set_seed(seed=42):
    """
    Sets random seeds across random, numpy, and torch for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Info] Random seed set to: {seed}")


def ensure_dir(dir_path):
    """
    Creates a directory if it does not already exist.
    """
    os.makedirs(dir_path, exist_ok=True)


def validate_image_file(image_input):
    """
    Validates if an image input (file path, PIL Image, or bytes) is valid and non-corrupted.

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    try:
        if isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                return False, f"File does not exist: {image_input}"
            if os.path.getsize(image_input) == 0:
                return False, "File is empty (0 bytes)."
            with Image.open(image_input) as img:
                img.verify()
            # Reopen after verify to ensure image content can be decoded
            with Image.open(image_input) as img:
                img.load()
        elif isinstance(image_input, Image.Image):
            image_input.load()
        elif isinstance(image_input, bytes):
            import io
            with Image.open(io.BytesIO(image_input)) as img:
                img.verify()
                img.load()
        else:
            return False, "Unsupported image input type."
        return True, ""
    except Exception as e:
        return False, f"Corrupted or invalid image: {str(e)}"


def load_efficientnet_b0(pretrained=True, num_classes=2):
    """
    Loads EfficientNet-B0 pretrained model and updates classification head for binary classification.

    Args:
        pretrained (bool): Whether to load ImageNet pretrained weights.
        num_classes (int): Output classes count (default 2 for REAL vs MANIPULATED).

    Returns:
        torch.nn.Module: Configured EfficientNet-B0 network.
    """
    if pretrained:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            model = models.efficientnet_b0(pretrained=True)
    else:
        model = models.efficientnet_b0(weights=None)

    # Replace classifier head (Dropout -> Linear)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def generate_synthetic_dataset(base_dir="dataset", samples_per_split={"train": 30, "validation": 10, "test": 10}):
    """
    Generates a synthetic sample dataset in `dataset/` directory.
    Useful for demonstration, initial testing, and verifying the ML pipeline end-to-end.
    
    - Real images: Smooth geometric shapes and natural color gradients.
    - Fake images: Geometric shapes with artificial noise blocks, splice boxes, and color artifacts.
    """
    print(f"[Info] Generating synthetic dataset in '{base_dir}'...")
    set_seed(42)
    classes = ["real", "fake"]
    
    for split, count in samples_per_split.items():
        for cls in classes:
            target_dir = os.path.join(base_dir, split, cls)
            ensure_dir(target_dir)
            
            for i in range(count):
                img_path = os.path.join(target_dir, f"sample_{cls}_{i+1:03d}.jpg")
                if os.path.exists(img_path):
                    continue
                
                # Base 224x224 RGB image
                img = Image.new("RGB", (224, 224), color=(
                    random.randint(180, 240),
                    random.randint(180, 240),
                    random.randint(180, 240)
                ))
                draw = ImageDraw.Draw(img)
                
                # Draw natural background patterns
                draw.ellipse([20, 20, 200, 200], fill=(
                    random.randint(50, 150),
                    random.randint(50, 150),
                    random.randint(100, 200)
                ))
                draw.rectangle([50, 50, 170, 170], fill=(
                    random.randint(100, 220),
                    random.randint(100, 220),
                    random.randint(50, 150)
                ))

                # For fake class, insert synthetic manipulation artifacts
                if cls == "fake":
                    arr = np.array(img)
                    x1, y1 = random.randint(60, 110), random.randint(60, 110)
                    patch_size = random.randint(45, 75)
                    # High-frequency digital noise patch (deepfake splice artifact simulation)
                    noise = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
                    arr[y1:y1+patch_size, x1:x1+patch_size] = noise
                    # Color space distortion box
                    arr[20:70, 20:70, 0] = np.clip(arr[20:70, 20:70, 0].astype(int) + 100, 0, 255).astype(np.uint8)
                    img = Image.fromarray(arr)

                img.save(img_path, "JPEG", quality=92)
                
    print("[Info] Synthetic dataset generation complete.")


if __name__ == "__main__":
    device = get_device()
    set_seed(42)
    generate_synthetic_dataset()
