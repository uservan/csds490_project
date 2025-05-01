from pathlib import Path

import torch
import torch.nn as nn
from common import DataSets
from cv2.typing import MatLike
from models import dark_face_Generator, lolGenerator, lolv2Generator

GENERATOR_MAPPING: dict[DataSets, nn.Module] = {
    "lol_dataset": lolGenerator(),
    "Dark_Face": dark_face_Generator(),
    "LOL-V2": lolv2Generator(),
}

WEIGHTS_PATH = Path.cwd().parent / "weights"
GENERATOR_WEIGHTS = "generator_weights.pth"
DISCRIMINATOR_WEIGHTS = "discriminator_weight.pth"


def gan_method(dataset: DataSets, img: MatLike) -> MatLike:
    # TODO maybe not spin up and down generator every time?
    generator = GENERATOR_MAPPING[dataset]
    generator.load_state_dict(torch.load(WEIGHTS_PATH / dataset / GENERATOR_WEIGHTS))
    return generator(img)
