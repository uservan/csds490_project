import numpy as np
import cv2
from PIL import Image


def compute_simple_quality(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    std_dev = np.std(gray)

    score = 100 - (0.7 * laplacian_var + 0.3 * std_dev)

    return max(score, 0)


def compute_cei(original_img, enhanced_img):
    def contrast(image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        return np.std(gray)

    img_orig_np = np.array(original_img.convert("RGB"))
    img_enh_np = np.array(enhanced_img.convert("RGB"))

    c_orig = contrast(img_orig_np)
    c_enh = contrast(img_enh_np)

    return c_enh / (c_orig + 1e-8)


def evaluate_image(enhanced_img, original_img=None):
    result = {}
    result["SIMPLE_SCORE"] = compute_simple_quality(enhanced_img)

    if original_img:
        result["CEI"] = compute_cei(original_img, enhanced_img)

    return result


if __name__ == "__main__":
    enhanced_img = Image.open("enhanced_dark.png")
    original_img = Image.open("original_dark.png")

    scores = evaluate_image(enhanced_img, original_img)

    print("Evaluation Results:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
