import cv2
import numpy as np
from scipy.spatial import distance
from scipy.ndimage import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from bm3d import bm3d


def get_sparse_neighbor(p, n, m):
    """Returns a dictionary, where the keys are index of 4-neighbor of `p` in the sparse matrix,
       and values are tuples (i, j, x), where `i`, `j` are index of neighbor in the normal matrix,
       and x is the direction of neighbor.

    :param p: index in the sparse matrix
            n: number of rows in the original matrix
            m: number of columns in the original matrix
    :return: dictionary containing indices of 4-neighbors of `p`
    """
    i, j = p // m, p % m
    d = {}
    if i - 1 >= 0:
        d[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        d[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        d[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        d[i * m + j + 1] = (i, j + 1, 1)
    return d

def create_spacial_affinity_kernel(spatial_sigma, size = 15):
    """Create a kernel that will be used to compute the HE spatial affinity based Gaussian weights.

    :param spatial_sigma: spatial standard deviation
            size: size of the kernel
    :return: `size` * `size` kernel
    """
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

    return kernel

def compute_smoothness_weights(T, x, kernel, eps):
    """Compute the smoothness weights used in refining the illumination map optimization problem.

    :param L: the illumination map to be refined
            x: the direction of the weights. Can either be x=1 for horizontal or x=0 for vertical.
            kernel: Gaussian kernel
    :return: smoothness weights according to direction x. same dimension as `L`.
    """
    delta_T = cv2.Sobel(T, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    G = convolve(np.ones_like(T), kernel, mode='constant')
    weight = G / (np.abs(convolve(delta_T, kernel, mode='constant')) + eps)
    return weight / (np.abs(delta_T) + eps)

def refine_illumination_map_linear(T, gamma, lambda_, kernel):
    """Refine the illumination map based on the optimization problem described in the LIME paper.
       This function use the sped-up solver presented in the paper.

    :param T: the illumination map to be refined
            gamma: gamma correction factor
            lambda_: coefficient to balance the terms in the optimization problem
            kernel: Gaussian kernel
    :return: refined illumination map
    """

    eps = 1e-5
    # compute smoothness weights
    wx = compute_smoothness_weights(T, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(T, x=0, kernel=kernel, eps=eps)

    n, m = T.shape
    T_1d = T.copy().flatten()

    # compute the five-point spatially inhomogeneous Laplacian matrix
    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    # solve the linear system
    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    T_refined = spsolve(csr_matrix(A), T_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

    # gamma correction
    T_refined = np.clip(T_refined, eps, 1) ** gamma

    return T_refined

def correct_underexposure(img, gamma, lambda_, kernel, denoise):
    """correct underexposudness using the retinex based algorithm presented in LIME paper.

    :param img: original image
            gamma: gamma correction factor
            lambda_: coefficient to balance the terms in the optimization problem
            kernel: Gaussian kernel
    :return: enhanced image
    """

    # first estimation of the illumination map
    T = np.max(img, axis=-1)
    # illumination refinement
    T_refined = refine_illumination_map_linear(T, gamma, lambda_, kernel)

    # correct image underexposure
    T_refined_3d = np.repeat(T_refined[..., None], 3, axis=-1)
    img_enhanced = img / T_refined_3d

    if denoise:
        img_enhanced = np.clip(img_enhanced * 255, 0, 255).astype("uint8")
        result_yuv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2YUV).astype(np.float32) / 255.0
        y, u, v = cv2.split(result_yuv)
        y_denoised = bm3d(y, sigma_psd=0.05)
        y_denoised = y_denoised.astype(np.float32)

        result_denoised = cv2.merge([y_denoised, u, v])
        result_denoised = cv2.cvtColor(result_denoised, cv2.COLOR_YUV2BGR)

        # result = img_enhanced * T_refined_3d + result_denoised * (1-T_refined_3d)
        img_enhanced = img_enhanced.astype(np.float32) / 255.0
        result = img_enhanced * T_refined_3d + result_denoised * (1-T_refined_3d)

    else:
        result = img_enhanced

    return result

def lime_enhance(img):
    """ enhance the image using LIME

    :param img: original image
    :return: enhanced image
    """
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # parameter used in LIME
    gamma = 0.6
    lambda_ = 0.15
    sigma = 3
    denoise = True

    # create spacial affinity kernel
    kernel = create_spacial_affinity_kernel(sigma)

    # enhance the image
    img_normalized = img.astype(float) / 255.
    img_enhanced = correct_underexposure(img_normalized, gamma, lambda_, kernel, denoise)

    result = np.clip(img_enhanced * 255, 0, 255).astype("uint8")

    return result


# Example usage
if __name__ == "__main__":
    input_path = "22.png"  # Replace with your image path
    image = cv2.imread(input_path)
    result = lime_enhance(image)

    # Save and display result
    # cv2.imwrite("enhanced.jpg", result)
    cv2.imshow("Original", image)
    cv2.imshow("Enhanced", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()