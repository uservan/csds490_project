import cv2
import numpy as np

def ahe(image, win_size):
    padded = cv2.copyMakeBorder(image, win_size // 2, win_size // 2, win_size // 2, win_size // 2, cv2.BORDER_REFLECT)
    output = np.zeros_like(image)
    height, weight = image.shape

    for i in range(height):
        for j in range(weight):
            local = padded[i:i+win_size, j:j+win_size]

            equalized = cv2.equalizeHist(local)
            output[i, j] = equalized[win_size // 2, win_size // 2]

    return output

def he_enhance(img):
    img_r = img[:, :, 2]
    img_g = img[:, :, 1]
    img_b = img[:, :, 0]

    he_img_r = cv2.equalizeHist(img_r)
    he_img_g = cv2.equalizeHist(img_g)
    he_img_b = cv2.equalizeHist(img_b)
    he_img = cv2.merge((he_img_b, he_img_g, he_img_r))

    return he_img

def ahe_enhance(img):
    # parameter used in AHE
    win_size = 64

    img_r = img[:, :, 2]
    img_g = img[:, :, 1]
    img_b = img[:, :, 0]

    # adaptive histogram equalization(AHE)
    ahe_img_r = ahe(img_r, win_size)
    ahe_img_g = ahe(img_g, win_size)
    ahe_img_b = ahe(img_b, win_size)
    ahe_img = cv2.merge((ahe_img_b, ahe_img_g, ahe_img_r))

    return ahe_img

def clahe_enhance(img):
    # parameter used in CLAHE
    clipLimit = 3

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    img_r = img[:, :, 2]
    img_g = img[:, :, 1]
    img_b = img[:, :, 0]
    # contrast limited histogram equalization(CLAHE)
    clahe_img_r = clahe.apply(img_r)
    clahe_img_g = clahe.apply(img_g)
    clahe_img_b = clahe.apply(img_b)
    clahe_img = cv2.merge((clahe_img_b, clahe_img_g, clahe_img_r))

    return clahe_img

# Example usage
if __name__ == "__main__":
    input_path = "22.png"  # Replace with your image path
    image = cv2.imread(input_path)
    result_he = he_enhance(image)
    result_ahe = ahe_enhance(image)
    result_clahe = clahe_enhance(image)

    # Save and display result
    # cv2.imwrite("enhanced.jpg", result)
    cv2.imshow("Original", image)
    cv2.imshow("Enhanced_HE", result_he)
    cv2.imshow("Enhanced_AHE", result_ahe)
    cv2.imshow("Enhanced_CLAHE", result_clahe)
    cv2.waitKey(0)
    cv2.destroyAllWindows()