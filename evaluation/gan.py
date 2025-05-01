from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from cv2.typing import MatLike
from eval import load_data
from models import lolGenerator

type DataSets = Literal["lol_dataset", "Dark_Face", "LOL-V2"]
# data = load_data()

# print(data["lol_dataset"])

GENERATOR_MAPPING = {
    'lol_dataset': lolGenerator()
}

WEIGHTS_PATH = Path.cwd().parent / "weights"
GENERATOR_WEIGHTS = "generator_weights.pth"
DISCRIMINATOR_WEIGHTS = "discriminator_weight.pth"


# def generic_method(img: MatLike, generator: nn.Module) -> MatLike:
#     return generator(img)


# def lol_method(img: MatLike) -> MatLike:
#     # TODO apply lol generator
#     generator = lolGenerator()
#     generator.load_state_dict(torch.load(Path('')))
#     return generic_method(img, generator)


# def dark_face_method(img: MatLike) -> MatLike:
#     # TODO apply dark_face generator
#     pass


# def lolv2_method(img: MatLike) -> MatLike:
#     # TODO apply lolv2 generator
#     pass


def gan_method(dataset: DataSets, img: MatLike) -> MatLike:
    # match dataset:
    #     case "lol_dataset":
    #         return lol_method(img)
    #     case "Dark_Face":
    #         return dark_face_method(img)
    #     case "LOL-V2":
    #         return lolv2_method(img)
    #     case _:
    #         raise ValueError(f"dataset {dataset} not in valid list")


    # TODO maybe not spin up and down generator everytime?
    generator = GENERATOR_MAPPING[dataset]
    generator.load_state_dict(
        torch.load(WEIGHTS_PATH / dataset / GENERATOR_WEIGHTS)
    )
    return generator(img)
