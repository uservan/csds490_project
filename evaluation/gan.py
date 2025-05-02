import torch
from common import GENERATOR_MAPPING, WEIGHTS_PATH, DataSets
from cv2.typing import MatLike


def gan_method(dataset: DataSets, img: MatLike) -> MatLike:
    # TODO maybe not spin up and down generator every time?
    generator = GENERATOR_MAPPING[dataset]
    generator.load_state_dict(torch.load(WEIGHTS_PATH / dataset / GENERATOR_WEIGHTS))
    return generator(img)
