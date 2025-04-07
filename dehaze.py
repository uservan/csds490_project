import cv2
import numpy as np

def invert_image(img):
    """Invert an image (simulate hazy image)"""
    return 255 - img

def get_dark_channel(img, size=15):
    """
    Compute the dark channel prior of the image.
    Input should be a float32 image in [0, 1].
    """
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def get_atmosphere(img, dark_channel, top_percent=0.001):
    """
    Estimate the atmospheric light A from the brightest pixels in the dark channel.
    """
    h, w = img.shape[:2]
    num_pixels = int(max(h * w * top_percent, 1))
    dark_vec = dark_channel.reshape(-1)
    indices = np.argsort(dark_vec)[-num_pixels:]
    brightest = img.reshape(-1, 3)[indices]
    A = np.mean(brightest, axis=0)
    return A

def get_transmission(img, A, omega=0.95, size=15):
    """
    Estimate the transmission map using the dark channel.
    """
    norm_img = img / A
    transmission = 1 - omega * get_dark_channel(norm_img, size)
    return np.clip(transmission, 0.15, 0.9)  # Safe range

def recover_image(img, t, A):
    """
    Recover the dehazed image using the estimated transmission and atmospheric light.
    """
    t = t[..., np.newaxis]  # Make shape (H, W, 1) for broadcasting
    J = (img - A) / t + A
    return np.clip(J, 0, 1)

def dehaze(img):
    """
    Main dehazing function. Input is a uint8 image, output is uint8 dehazed image.
    """
    img = img.astype(np.float32) / 255.0  # Normalize
    dark = get_dark_channel(img)
    A = get_atmosphere(img, dark)
    t = get_transmission(img, A)
    J = recover_image(img, t, A)
    return (J * 255).astype(np.uint8)

def enhance_low_light_image(img):
    """
    Enhance a low-light image using the dehazing-based method.
    """
    inverted = invert_image(img)
    dehazed = dehaze(inverted)
    enhanced = invert_image(dehazed)
    return enhanced

def dehaze_and_enhance(input_path):
    """
    Dehaze and enhance the image.
    """
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Image not found. Check the path.")

    # Ensure image is in uint8 and in [0, 255]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    result = enhance_low_light_image(image)
    return result
# Example usage
if __name__ == "__main__":
    
    input_path = 'data/lol_dataset/test15/low/22.png'  # Replace with your image path
    image = cv2.imread(input_path)
    result = dehaze_and_enhance(input_path)

    # Save and display result
    cv2.imwrite("enhanced.jpg", result)
    cv2.imshow("Original", image)
    cv2.imshow("Enhanced", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()