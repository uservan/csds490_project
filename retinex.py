import cv2
import numpy as np


def single_scale_retinex(img, sigma):
    img = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    retinex = np.log(img) - np.log(blur + 1.0)
    return retinex

def multi_scale_retinex(img, sigmas):
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    retinex /= len(sigmas)
    return retinex

def simplest_color_balance(img, low_clip=1, high_clip=99):
    result = np.zeros_like(img)
    for i in range(3):
        channel = img[:, :, i]
        low_val = np.percentile(channel, low_clip)
        high_val = np.percentile(channel, high_clip)
        channel = np.clip((channel - low_val) * 255.0 / (high_val - low_val + 1e-6), 0, 255)
        result[:, :, i] = channel
    return result.astype(np.uint8)

def single_retinex_enhance(img_bgr, sigma, if_exp):
    img = img_bgr.astype(np.float32)
    img_retinex = np.zeros_like(img)

    for i in range(3):  # R, G, B channel
        img_retinex[:, :, i] = single_scale_retinex(img[:, :, i], sigma)

    # Normalize
    if if_exp:
        img_retinex = simplest_color_balance(np.exp(img_retinex))
    else:
        img_retinex = simplest_color_balance(img_retinex)
    return img_retinex

def multi_retinex_enhance(img_bgr, sigmas, if_exp):
    img = img_bgr.astype(np.float32)
    img_retinex = np.zeros_like(img)

    for i in range(3):  # R, G, B channel
        img_retinex[:, :, i] = multi_scale_retinex(img[:, :, i], sigmas)

    # Normalize
    if if_exp:
        img_retinex = simplest_color_balance(np.exp(img_retinex))
    else:
        img_retinex = simplest_color_balance(img_retinex)
    return img_retinex

def ssr_enhance(img):
    """ enhance the image using SSR

    :param img: original image
    :return: enhanced image
    """
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # parameter used in SSR
    sigma = 80
    if_exp = False

    result = single_retinex_enhance(img, sigma, if_exp)

    return result

def msr_enhance(img):
    """ enhance the image using MSR

    :param img: original image
    :return: enhanced image
    """
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # parameter used in MSR
    sigmas = [15, 80, 250]
    if_exp = False

    result = multi_retinex_enhance(img, sigmas, if_exp)

    return result

# Example usage
if __name__ == "__main__":
    input_path = "22.png"  # Replace with your image path
    image = cv2.imread(input_path)
    result_ssr = ssr_enhance(image)
    result_msr = msr_enhance(image)

    # Save and display result
    # cv2.imwrite("enhanced.jpg", result)
    cv2.imshow("Original", image)
    cv2.imshow("Enhanced_SSR", result_ssr)
    cv2.imshow("Enhanced_MSR", result_msr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()