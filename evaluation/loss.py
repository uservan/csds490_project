import torch
import torch.nn as nn


def exposure_control_loss(image, patch_size=16):
    mean = nn.functional.avg_pool2d(image, patch_size)
    return torch.mean((mean - 0.6) ** 2)  # Target exposure (e.g. 0.6)


def tv_loss(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    )


def contrast_loss(img):
    if img.shape[1] == 3:
        # Convert to grayscale using luminance-preserving weights
        grayscale = (
            0.2989 * img[:, 0, :, :]
            + 0.5870 * img[:, 1, :, :]
            + 0.1140 * img[:, 2, :, :]
        )
    elif img.shape[1] == 1:
        # Already grayscale
        grayscale = img[:, 0, :, :]
    else:
        raise ValueError(f"Unexpected number of channels: {img.shape[1]}")

    std = torch.std(grayscale, dim=[1, 2], keepdim=True)
    return -torch.mean(std)  # Maximize contrast


def unsupervised_loss(enhanced_img):
    loss_exposure = exposure_control_loss(enhanced_img)
    loss_tv = tv_loss(enhanced_img)
    loss_contrast = contrast_loss(enhanced_img)
    return loss_exposure + 0.5 * loss_tv + 0.3 * loss_contrast
