"""
Prediction module for AI Image Authenticity Checker & Deepfake Detector.
Loads trained model, preprocesses target image, and returns real-time prediction and confidence scores.
"""

import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F

from src.utils import get_device, validate_image_file, load_efficientnet_b0, IDX_TO_CLASS
from src.data_preprocessing import get_transforms


def predict_image(image_input, model_path="models/deepfake_detector.pth", device=None):
    """
    Predicts whether an image is REAL or MANIPULATED using the trained EfficientNet-B0 model.

    Args:
        image_input (str, PIL.Image.Image, or bytes): Input image to classify.
        model_path (str): Path to trained model checkpoint file (.pth).
        device (torch.device, optional): Compute device (CPU/GPU).

    Returns:
        dict: Prediction results containing:
              - 'class_name': 'REAL' or 'MANIPULATED'
              - 'confidence': float (percentage 0-100)
              - 'probabilities': dict with probabilities for REAL and MANIPULATED
              - 'status': 'success'
              - 'image_pil': loaded PIL Image object for downstream Grad-CAM
    """
    # Validate image input
    is_valid, err_msg = validate_image_file(image_input)
    if not is_valid:
        raise ValueError(f"Invalid image input: {err_msg}")

    if device is None:
        device = get_device()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model checkpoint not found at '{model_path}'. "
            "Please train the model first using src/train.py or via the Streamlit interface."
        )

    # Load image as PIL RGB
    if isinstance(image_input, (str, os.PathLike)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, bytes):
        import io
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        raise TypeError("Unsupported image input type.")

    # Load model
    model = load_efficientnet_b0(pretrained=False, num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Preprocess image
    _, eval_transform = get_transforms()
    input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)

    prob_real = float(probs[0].item())
    prob_fake = float(probs[1].item())

    pred_idx = int(torch.argmax(probs).item())
    predicted_label = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx].item() * 100.0)

    result = {
        "class_name": predicted_label,
        "confidence": confidence,
        "probabilities": {
            "REAL": prob_real * 100.0,
            "MANIPULATED": prob_fake * 100.0
        },
        "status": "success",
        "image_pil": pil_img,
        "model": model,
        "input_tensor": input_tensor
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict image authenticity using deepfake detector")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model_path", type=str, default="models/deepfake_detector.pth", help="Path to model file")

    args = parser.parse_args()

    try:
        res = predict_image(args.image_path, model_path=args.model_path)
        print("\n" + "="*40)
        print(" PREDICTION RESULT ")
        print("="*40)
        print(f"Predicted Class:  {res['class_name']}")
        print(f"Confidence Score: {res['confidence']:.2f}%")
        print(f"Probabilities:    REAL = {res['probabilities']['REAL']:.2f}%, MANIPULATED = {res['probabilities']['MANIPULATED']:.2f}%")
        print("="*40 + "\n")
    except Exception as e:
        print(f"[Error] Prediction failed: {e}")
