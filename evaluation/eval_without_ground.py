import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import piq  # pip install piq
from skimage import img_as_float
import os

def load_image_tensor(image_path):
    """Load image and convert to 4D tensor (1, 3, H, W) for PIQ."""
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

def compute_no_ref_metrics(image_path):
    """
    Compute NIQE, BRISQUE, PIQE using PIQ.
    """
    img_tensor = load_image_tensor(image_path)

    niqe_score = piq.niqe(img_tensor)
    piqe_score = piq.piqe(img_tensor)
    brisque_score = piq.brisque(img_tensor)

    return {
        'NIQE': niqe_score.item(),
        'PIQE': piqe_score.item(),
        'BRISQUE': brisque_score.item()
    }

def compute_cei(original_path, enhanced_path):
    """
    Compute Contrast Enhancement Index (CEI).
    """
    def contrast(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray)

    img_orig = cv2.imread(original_path)
    img_enh = cv2.imread(enhanced_path)

    c_orig = contrast(img_orig)
    c_enh = contrast(img_enh)

    return c_enh / (c_orig + 1e-8)  # avoid divide by zero

def evaluate_image(enhanced_path, original_path=None):
    """
    Compute all no-reference IQA metrics, and optionally CEI if original provided.
    """
    result = compute_no_ref_metrics(enhanced_path)

    if original_path:
        cei_val = compute_cei(original_path, enhanced_path)
        result['CEI'] = cei_val

    return result

if __name__ == '__main__':
    enhanced = 'enhanced_dark.png'
    original = 'original_dark.png'  # optional, can be None

    scores = evaluate_image(enhanced, original_path=original)

    print("No-Reference IQA Results:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")