"""
Explainable AI (XAI) module using Grad-CAM for EfficientNet-B0.
Generates class activation heatmaps and overlays them onto input images to visualize suspicious regions.
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.utils import get_device, validate_image_file, load_efficientnet_b0
from src.data_preprocessing import get_transforms


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation tailored for
    EfficientNet-B0 architectures.
    """
    def __init__(self, model, target_layer=None):
        """
        Args:
            model (torch.nn.Module): EfficientNet-B0 model.
            target_layer (torch.nn.Module, optional): Convolutional layer to extract features from.
                                                      Defaults to model.features[-1].
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            # EfficientNet-B0 final conv layer is the last module in model.features
            self.target_layer = self.model.features[-1]
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates normalized Grad-CAM heatmap tensor (1, 1, H, W) for specified class.

        Args:
            input_tensor (torch.Tensor): Preprocessed input image tensor of shape (1, C, H, W).
            target_class (int, optional): Class index to explain. If None, uses predicted class.

        Returns:
            np.ndarray: Normalized 2D heatmap numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations during Grad-CAM backward pass.")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # Shape: (1, 1, H_feature, W_feature)

        # Apply ReLU to retain only positive influence
        cam = F.relu(cam)

        # Interpolate to match input spatial dimensions (224x224)
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Convert to numpy and normalize to [0, 1]
        heatmap = cam.squeeze().cpu().numpy()
        min_val, max_val = np.min(heatmap), np.max(heatmap)
        if max_val > min_val:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap


def overlay_heatmap(pil_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap onto a original PIL image.

    Args:
        pil_image (PIL.Image.Image): Original input image.
        heatmap (np.ndarray): 2D heatmap array normalized to [0, 1].
        alpha (float): Transparency factor for heatmap overlay (0.0 - 1.0).
        colormap (int): OpenCV colormap enum (e.g. cv2.COLORMAP_JET).

    Returns:
        tuple: (overlay_pil_img, heatmap_rgb_pil_img)
    """
    orig_w, orig_h = pil_image.size
    orig_np = np.array(pil_image.convert("RGB"))

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    color_mapped = cv2.applyColorMap(heatmap_uint8, colormap)
    color_mapped_rgb = cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    overlay_np = cv2.addWeighted(orig_np, 1.0 - alpha, color_mapped_rgb, alpha, 0)

    overlay_img = Image.fromarray(overlay_np)
    heatmap_img = Image.fromarray(color_mapped_rgb)

    return overlay_img, heatmap_img


def explain_prediction(image_input, model_path="models/deepfake_detector.pth", target_class=None, device=None):
    """
    Pipeline function to run Grad-CAM explanation on an image.

    Args:
        image_input (str, PIL.Image.Image, or bytes): Target image.
        model_path (str): Path to trained model checkpoint.
        target_class (int, optional): Target class index.
        device (torch.device, optional): Compute device.

    Returns:
        dict: Explanation results including overlay Image, raw heatmap Image, and target class label.
    """
    if device is None:
        device = get_device()

    # Load image
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

    # Preprocess
    _, eval_transform = get_transforms()
    input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

    # Grad-CAM
    grad_cam = GradCAM(model)
    heatmap = grad_cam.generate_heatmap(input_tensor, target_class=target_class)
    overlay_img, heatmap_img = overlay_heatmap(pil_img, heatmap)

    return {
        "overlay_image": overlay_img,
        "heatmap_image": heatmap_img,
        "heatmap_raw": heatmap
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanation for image")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--model_path", type=str, default="models/deepfake_detector.pth", help="Model path")
    parser.add_argument("--output_path", type=str, default="grad_cam_overlay.png", help="Saved output image path")

    args = parser.parse_args()

    try:
        res = explain_prediction(args.image_path, model_path=args.model_path)
        res["overlay_image"].save(args.output_path)
        print(f"[Success] Saved Grad-CAM overlay to '{args.output_path}'")
    except Exception as e:
        print(f"[Error] Grad-CAM generation failed: {e}")
