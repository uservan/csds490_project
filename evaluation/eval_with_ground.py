import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def compute_mse(img1, img2):
    """Compute Mean Squared Error (MSE) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    return mse

def compute_psnr(img1, img2, max_pixel=255.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def compute_ssim(img1, img2):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    Automatically converts RGB images to grayscale before comparison.
    """
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value

def evaluate_image_pair(enhanced_path, reference_path):
    """
    Evaluate a single pair of images using MSE, PSNR, and SSIM.
    """
    # Load images
    enhanced = cv2.imread(enhanced_path)
    reference = cv2.imread(reference_path)

    # Convert to float32
    enhanced = enhanced.astype(np.float32)
    reference = reference.astype(np.float32)

    # Compute metrics
    mse_val = compute_mse(enhanced, reference)
    psnr_val = compute_psnr(enhanced, reference)
    ssim_val = compute_ssim(enhanced, reference)

    return {
        'MSE': mse_val,
        'PSNR': psnr_val,
        'SSIM': ssim_val
    }

if __name__ == '__main__':
    result = evaluate_image_pair('enhanced.png', 'reference.png')
    print("Evaluation Results:")
    for metric, value in result.items():
        print(f"{metric}: {value:.4f}")