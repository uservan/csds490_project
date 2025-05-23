import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def compute_mse(img1, img2):
    """Compute Mean Squared Error (MSE) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    return mse


def compute_psnr(img1, img2, max_pixel=255.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float("inf")  # Perfect match
    psnr = 10 * np.log10((max_pixel**2) / mse)
    return psnr


def compute_ssim(img1, img2):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    Automatically converts RGB images to grayscale before comparison.
    """
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value


def evaluate_image_pair_from_pil(enhanced_img, reference_img):
    """
    Evaluate a pair of PIL Images using MSE, PSNR, and SSIM.
    """
    # Convert PIL images to NumPy arrays (RGB)
    enhanced = np.array(enhanced_img.convert("RGB"), dtype=np.float32)
    reference = np.array(reference_img.convert("RGB"), dtype=np.float32)

    # Compute metrics
    mse_val = compute_mse(enhanced, reference)
    psnr_val = compute_psnr(enhanced, reference)
    ssim_val = compute_ssim(enhanced, reference)

    return {"MSE": mse_val, "PSNR": psnr_val, "SSIM": ssim_val}


# Example usage
if __name__ == "__main__":
    enhanced = Image.open("enhanced.png")
    reference = Image.open("reference.png")

    results = evaluate_image_pair_from_pil(enhanced, reference)
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
