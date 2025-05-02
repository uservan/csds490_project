import torch
from common import GENERATOR_MAPPING, GENERATOR_WEIGHTS, WEIGHTS_PATH, DataSets, device
from custom_datasets import TO_TENSOR
from cv2.typing import MatLike
from eval import cv2_to_pil, pil_to_cv2
from torchvision.transforms.functional import to_pil_image


def gan_method(dataset: DataSets, img: MatLike) -> MatLike:
    # TODO maybe not spin up and down generator every time?
    generator = GENERATOR_MAPPING[dataset]
    generator.load_state_dict(torch.load(WEIGHTS_PATH / dataset / GENERATOR_WEIGHTS))

    # Generator's expect PIL
    pil_image = cv2_to_pil(img)
    tensor = TO_TENSOR(pil_image)

    tensor = tensor.to(device)
    tensor_shape = tensor.shape
    is_three_D = len(tensor.shape) == 3

    if is_three_D:
        tensor = tensor.view(1, *tensor_shape)

    generated_image = generator(tensor)

    if is_three_D:
        generated_image = generated_image.view(*tensor_shape)

    generated_pil = to_pil_image(generated_image)

    return pil_to_cv2(generated_pil)
