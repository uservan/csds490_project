import numpy as np
import cv2
from PIL import Image


def compute_simple_quality(pil_img):
    # 1. 转灰度图
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. 清晰度：Laplacian 方差
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 3. 对比度：标准差
    std_dev = np.std(gray)

    # 4. 归一化（假设清晰度最大 1000，对比度最大 128，都归一化到 [0, 1]）
    norm_lap = min(lap_var / 1000.0, 1.0)
    norm_std = min(std_dev / 128.0, 1.0)

    # 5. 加权得分（清晰度权重更大）
    score = (0.7 * norm_lap + 0.3 * norm_std) * 100

    return round(score, 2)


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
