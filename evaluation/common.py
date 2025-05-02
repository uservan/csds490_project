from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from models import (
    dark_faceDiscriminator,
    dark_faceGenerator,
    lolDiscriminator,
    lolGenerator,
    lolv2Discriminator,
    lolv2Generator,
)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
torch.set_default_device(device)
torch.cuda.empty_cache()


type GroundTruthDataSets = Literal["lol_dataset", "LOL-v2"]
type NoGroundTruthDataSets = Literal["Dark_Face"]

type DataSets = GroundTruthDataSets | NoGroundTruthDataSets

GENERATOR_MAPPING: dict[DataSets, nn.Module] = {
    "lol_dataset": lolGenerator().to(device),
    "Dark_Face": dark_faceGenerator().to(device),
    "LOL-v2": lolv2Generator().to(device),
}

DISCRIMINATOR_MAPPING: dict[DataSets, nn.Module] = {
    "lol_dataset": lolDiscriminator().to(device),
    "Dark_Face": dark_faceDiscriminator().to(device),
    "LOL-v2": lolv2Discriminator().to(device),
}

WEIGHTS_PATH = Path.cwd().parent / "weights"
GENERATOR_WEIGHTS = "generator_weights.pth"
DISCRIMINATOR_WEIGHTS = "discriminator_weight.pth"
