"""
Data preprocessing, augmentation, and robust PyTorch Dataset & DataLoader utilities.
Handles corrupted image filtering, ImageNet normalization, and data loading splits.
"""

import os
import glob
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils import validate_image_file, CLASS_TO_IDX

# Set up logger for dataset validation
logger = logging.getLogger("DataPreprocessing")
logger.setLevel(logging.INFO)


def get_transforms(image_size=(224, 224)):
    """
    Returns image transformation pipelines for training and evaluation.

    Args:
        image_size (tuple): Target height and width (default 224x224).

    Returns:
        tuple: (train_transforms, eval_transforms)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform


class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image datasets supporting directory layout:
      split/
        real/
        fake/

    Validates corrupted/invalid images during initialization and gracefully handles
    runtime errors during loading.
    """
    def __init__(self, split_dir, transform=None):
        """
        Args:
            split_dir (str): Path to split directory (e.g., dataset/train).
            transform (callable, optional): PyTorch torchvision transform.
        """
        self.split_dir = split_dir
        self.transform = transform
        self.samples = []

        if not os.path.exists(split_dir):
            logger.warning(f"Split directory '{split_dir}' does not exist.")
            return

        # Target class subdirectories mapping
        # Maps 'real' -> 0 (REAL), 'fake' -> 1 (MANIPULATED)
        folder_mapping = {
            "real": CLASS_TO_IDX["REAL"],
            "fake": CLASS_TO_IDX["MANIPULATED"]
        }

        supported_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

        for class_folder, label in folder_mapping.items():
            folder_path = os.path.join(split_dir, class_folder)
            if not os.path.exists(folder_path):
                logger.warning(f"Class directory '{folder_path}' missing.")
                continue

            # Gather image files
            files = []
            for ext in supported_extensions:
                files.extend(glob.glob(os.path.join(folder_path, ext)))
                files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

            # Validate each image
            valid_count = 0
            for file_path in files:
                is_valid, err = validate_image_file(file_path)
                if is_valid:
                    self.samples.append((file_path, label))
                    valid_count += 1
                else:
                    logger.warning(f"Skipping invalid/corrupted image '{file_path}': {err}")

            logger.info(f"Loaded {valid_count} valid images from '{folder_path}' (Label: {label})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            logger.error(f"Error loading image '{path}': {e}. Returning zero tensor.")
            # Fallback to zero tensor to avoid breaking batch loading
            dummy_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            return dummy_tensor, label


def create_dataloaders(data_dir="dataset", batch_size=16, num_workers=0):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        data_dir (str): Base dataset directory containing train/validation/test folders.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.

    Returns:
        dict: Dictionary of DataLoaders {'train': train_loader, 'val': val_loader, 'test': test_loader}
              or None if train set is empty.
    """
    train_transform, eval_transform = get_transforms()

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    train_ds = DeepfakeDataset(train_dir, transform=train_transform)
    val_ds = DeepfakeDataset(val_dir, transform=eval_transform)
    test_ds = DeepfakeDataset(test_dir, transform=eval_transform)

    if len(train_ds) == 0:
        logger.error(f"No valid training samples found in '{train_dir}'.")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if len(test_ds) > 0 else None

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Data Preprocessing Pipeline")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    args = parser.parse_args()

    dataloaders = create_dataloaders(data_dir=args.data_dir)
    if dataloaders:
        print("[Success] DataLoaders created successfully:")
        for key, loader in dataloaders.items():
            if loader:
                print(f"  - {key}: {len(loader.dataset)} images ({len(loader)} batches)")
    else:
        print("[Warning] No data found. Run synthetic dataset generator in utils.py first.")
